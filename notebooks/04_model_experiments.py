# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %%
# imports, paths and Loading data
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path().resolve().parent
sys.path.append(str(PROJECT_ROOT))

csv_path = PROJECT_ROOT / "data" / "processed" / "ims_features_sample.csv"
assert csv_path.exists(), f"Missing features file: {csv_path}"

df = pd.read_csv(csv_path)
print("Loaded:", df.shape)
print("Label counts:\n", df["label"].value_counts(dropna=False))

# %%
#Leakage-safe split by file
# Prevent leakage: split by file_name (no shared windows between train/val)
files = df["file_name"].unique()

rng = np.random.default_rng(42)
files = rng.permutation(files)

split = int(0.75 * len(files))
train_files = set(files[:split])
val_files   = set(files[split:])

train_df = df[df["file_name"].isin(train_files)].copy()
val_df   = df[df["file_name"].isin(val_files)].copy()

print("\nTrain rows:", train_df.shape, "| Val rows:", val_df.shape)
print("Train label counts:\n", train_df["label"].value_counts())
print("Val label counts:\n", val_df["label"].value_counts())

# %%
# Drop metadata + clean infinities
DROP_COLS = ["test_dir", "file_name", "label", "file_index", "window_index"]
DROP_COLS = [c for c in DROP_COLS if c in train_df.columns]

X_train = train_df.drop(columns=DROP_COLS)
y_train = train_df["label"].astype(int).values

X_val = val_df.drop(columns=DROP_COLS)
y_val = val_df["label"].astype(int).values

# Replace inf/-inf with NaN so imputers handle them
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_val   = X_val.replace([np.inf, -np.inf], np.nan)

print("\nX_train:", X_train.shape, "X_val:", X_val.shape)
print("Train class balance:", np.bincount(y_train))
print("Val class balance:", np.bincount(y_val))


# %%
# Better evaluation + threshold tuning (KEY FIX)
def summarize_probs(y_prob, name="model"):
    print(f"{name} prob summary: "
          f"min={y_prob.min():.4f}, p50={np.median(y_prob):.4f}, "
          f"p90={np.quantile(y_prob, 0.9):.4f}, max={y_prob.max():.4f}")

def evaluate_model(name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    pr  = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

    print(f"\n=== {name} @ thr={threshold:.2f} ===")
    print(f"ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}")
    print("Pred counts:", np.bincount(y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))

    return {"name": name, "threshold": threshold, "roc_auc": roc, "pr_auc": pr}

def pick_threshold_for_recall(y_true, y_prob, target_recall=0.70):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds length = n-1; precision/recall length = n
    best = None  # (precision, recall, threshold)
    for i, thr in enumerate(thresholds):
        if recall[i] >= target_recall:
            cand = (precision[i], recall[i], thr)
            if best is None or cand[0] > best[0]:
                best = cand
    return best


# %%
# Logistic Regression — add imbalance handling
# Logistic Regression: class_weight balanced + pipeline
lr = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))
])

lr.fit(X_train, y_train)
lr_prob = lr.predict_proba(X_val)[:, 1]
summarize_probs(lr_prob, "LogReg")

# Evaluate at default and a more realistic threshold
evaluate_model("LogReg balanced", y_val, lr_prob, threshold=0.50)
evaluate_model("LogReg balanced", y_val, lr_prob, threshold=0.30)

best_lr = pick_threshold_for_recall(y_val, lr_prob, target_recall=0.70)
print("Best LR threshold (recall>=0.70):", best_lr)
if best_lr:
    evaluate_model("LogReg tuned", y_val, lr_prob, threshold=float(best_lr[2]))

# %%
# Random Forest — add imbalance handling + consistent imputer
imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_val_imp   = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)

# safety if any all-NaN columns exist
X_train_imp = X_train_imp.fillna(0)
X_val_imp   = X_val_imp.fillna(0)

rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample",  # better for RF
    min_samples_leaf=2
)
rf.fit(X_train_imp, y_train)
rf_prob = rf.predict_proba(X_val_imp)[:, 1]
summarize_probs(rf_prob, "RF")

evaluate_model("RF balanced", y_val, rf_prob, threshold=0.50)
evaluate_model("RF balanced", y_val, rf_prob, threshold=0.25)

best_rf = pick_threshold_for_recall(y_val, rf_prob, target_recall=0.70)
print("Best RF threshold (recall>=0.70):", best_rf)
if best_rf:
    evaluate_model("RF tuned", y_val, rf_prob, threshold=float(best_rf[2]))


# %%
# XGBoost — add scale_pos_weight + optional threshold tuning
results = []

try:
    from xgboost import XGBClassifier

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / max(pos, 1)
    print("scale_pos_weight:", scale_pos_weight)

    xgb = XGBClassifier(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_lambda=1.0,
        gamma=0.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    xgb.fit(X_train_imp, y_train)  # use same imputed matrices
    xgb_prob = xgb.predict_proba(X_val_imp)[:, 1]
    summarize_probs(xgb_prob, "XGB")

    results.append(evaluate_model("XGBoost balanced", y_val, xgb_prob, threshold=0.50))
    results.append(evaluate_model("XGBoost balanced", y_val, xgb_prob, threshold=0.20))

    best_xgb = pick_threshold_for_recall(y_val, xgb_prob, target_recall=0.70)
    print("Best XGB threshold (recall>=0.70):", best_xgb)
    if best_xgb:
        results.append(evaluate_model("XGBoost tuned", y_val, xgb_prob, threshold=float(best_xgb[2])))

except Exception as e:
    print("\n[XGBoost skipped] Install with: pip install xgboost")
    print("Reason:", e)

try:
    from xgboost import XGBClassifier

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / max(pos, 1)
    print("scale_pos_weight:", scale_pos_weight)

    xgb = XGBClassifier(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        reg_lambda=1.0,
        gamma=0.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    xgb.fit(X_train_imp, y_train)  # use same imputed matrices
    xgb_prob = xgb.predict_proba(X_val_imp)[:, 1]
    summarize_probs(xgb_prob, "XGB")

    results.append(evaluate_model("XGBoost balanced", y_val, xgb_prob, threshold=0.50))
    results.append(evaluate_model("XGBoost balanced", y_val, xgb_prob, threshold=0.20))

    best_xgb = pick_threshold_for_recall(y_val, xgb_prob, target_recall=0.70)
    print("Best XGB threshold (recall>=0.70):", best_xgb)
    if best_xgb:
        results.append(evaluate_model("XGBoost tuned", y_val, xgb_prob, threshold=float(best_xgb[2])))

except Exception as e:
    print("\n[XGBoost skipped] Install with: pip install xgboost")
    print("Reason:", e)

# %%
# Choose champion + save artifacts for backend
import joblib

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

# Build a comparison table (PR-AUC first; ROC-AUC second)
rows = []

# LR
rows.append({"model": "LogReg", "pr_auc": average_precision_score(y_val, lr_prob), "roc_auc": roc_auc_score(y_val, lr_prob)})
# RF
rows.append({"model": "RF", "pr_auc": average_precision_score(y_val, rf_prob), "roc_auc": roc_auc_score(y_val, rf_prob)})

# XGB if available
if "xgb_prob" in globals():
    rows.append({"model": "XGB", "pr_auc": average_precision_score(y_val, xgb_prob), "roc_auc": roc_auc_score(y_val, xgb_prob)})

score_df = pd.DataFrame(rows).sort_values(["pr_auc", "roc_auc"], ascending=False)
print("\nModel comparison (sorted by PR-AUC):")
display(score_df)

champion_name = score_df.iloc[0]["model"]

# Pick champion objects + tuned threshold (fallbacks)
if champion_name == "LogReg":
    champion_model = lr
    champion_threshold = float(best_lr[2]) if best_lr else 0.30
    feature_list = list(X_train.columns)

elif champion_name == "RF":
    champion_model = (rf, imputer)  # store rf + imputer together
    champion_threshold = float(best_rf[2]) if best_rf else 0.25
    feature_list = list(X_train.columns)

else:  # XGB
    champion_model = (xgb, imputer)
    champion_threshold = float(best_xgb[2]) if ("best_xgb" in globals() and best_xgb) else 0.20
    feature_list = list(X_train.columns)

joblib.dump(champion_model, ARTIFACT_DIR / "model.joblib")
joblib.dump(
    {"threshold": champion_threshold, "features": feature_list, "champion": champion_name},
    ARTIFACT_DIR / "meta.joblib"
)

print(f"\nSaved champion: {champion_name}")
print("Threshold:", champion_threshold)
print("Artifacts:", ARTIFACT_DIR / "model.joblib", "and", ARTIFACT_DIR / "meta.joblib")

# %%
