from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------
# IMS file sorting (timestamp)
# -----------------------------

TS_RE = re.compile(r"(\d{4})[.\-_](\d{2})[.\-_](\d{2})[.\-_](\d{2})[.\-_](\d{2})[.\-_](\d{2})")

def parse_timestamp_from_name(fname: str) -> datetime | None:
    m = TS_RE.search(fname)
    if not m:
        return None
    y, mo, d, h, mi, s = map(int, m.groups())
    return datetime(y, mo, d, h, mi, s)

def list_files_sorted(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir() if p.is_file()]
    files.sort(
        key=lambda p: (
            parse_timestamp_from_name(p.name) is None,
            parse_timestamp_from_name(p.name) or p.name
        )
    )
    return files


# -----------------------------
# IMS snapshot loading (ASCII)
# -----------------------------

def load_ims_file(path: Path) -> np.ndarray:
    """
    IMS snapshot files are ASCII numeric matrices:
      rows = samples (typically 20480)
      cols = channels (4 or 8 depending on test set)
    """
    x = np.loadtxt(path, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


# -----------------------------
# Feature extraction
# -----------------------------

def features_from_snapshot(x: np.ndarray, fs: float) -> tuple[np.ndarray, list[str]]:
    """
    Compute baseline features per channel:
      rms, std, p2p, kurtosis, skew, spectral_centroid, dominant_freq

    Returns:
      feats: shape (n_channels * 7,)
      names: list of feature names matching order
    """
    n = x.shape[0]
    n_channels = x.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    feats: list[float] = []
    names: list[str] = []

    for ch in range(n_channels):
        s = x[:, ch]

        # time domain
        rms = float(np.sqrt(np.mean(s ** 2)))
        std = float(np.std(s))
        p2p = float(np.ptp(s))
        krt = float(kurtosis(s, fisher=False))
        skw = float(skew(s))

        # frequency domain
        S = np.fft.rfft(s)
        mag = np.abs(S)
        mag_sum = float(np.sum(mag) + 1e-12)
        centroid = float(np.sum(freqs * mag) / mag_sum)
        dom_freq = float(freqs[int(np.argmax(mag))])

        feats.extend([rms, std, p2p, krt, skw, centroid, dom_freq])
        names.extend([
            f"rms_ch{ch+1}",
            f"std_ch{ch+1}",
            f"p2p_ch{ch+1}",
            f"kurt_ch{ch+1}",
            f"skew_ch{ch+1}",
            f"centroid_ch{ch+1}",
            f"domfreq_ch{ch+1}",
        ])

    return np.array(feats, dtype=np.float32), names


def build_feature_matrix(
    test_folder: Path,
    fs: float,
    max_files: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Loads all (or first max_files) snapshots in a test folder and returns X.
    """
    files = list_files_sorted(test_folder)
    if max_files is not None:
        files = files[:max_files]

    if not files:
        raise ValueError(f"No files found in: {test_folder}")

    X_rows: list[np.ndarray] = []
    feature_names: list[str] | None = None

    for f in files:
        snap = load_ims_file(f)
        feats, names = features_from_snapshot(snap, fs=fs)
        if feature_names is None:
            feature_names = names
        X_rows.append(feats)

    X = np.vstack(X_rows)
    return X, (feature_names or [])


def align_channel_features(X_list: list[np.ndarray]) -> list[np.ndarray]:
    """
    If different tests have different channel counts, feature vector lengths differ.
    We align by truncating to the minimum feature length across runs.
    """
    min_dim = min(X.shape[1] for X in X_list)
    return [X[:, :min_dim] for X in X_list]


def truncate_feature_names(feature_names: list[str], dim: int) -> list[str]:
    return feature_names[:dim]


# -----------------------------
# Training (unsupervised)
# -----------------------------

def train_isolation_forest(
    X_train: np.ndarray,
    random_state: int = 42,
    contamination: float = 0.05,
) -> Pipeline:
    """
    Pipeline: StandardScaler -> IsolationForest
    We keep the pipeline as the model, but also save the scaler separately
    for your current API design (model + preprocessor).
    """
    clf = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", clf),
    ])

    pipe.fit(X_train)
    return pipe


# -----------------------------
# Registry + artifact saving
# -----------------------------

@dataclass(frozen=True)
class Paths:
    backend_dir: Path
    raw_dir: Path
    models_dir: Path


def resolve_paths(repo_root: Path) -> Paths:
    backend_dir = repo_root / "backend"
    raw_dir = repo_root / "data" / "raw" if (repo_root / "data").exists() else backend_dir / "data" / "raw"
    # Prefer backend/data/raw if you keep data under backend/
    if (backend_dir / "data" / "raw").exists():
        raw_dir = backend_dir / "data" / "raw"

    models_dir = backend_dir / "app" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return Paths(backend_dir=backend_dir, raw_dir=raw_dir, models_dir=models_dir)


def save_registry(models_dir: Path, model_name: str, feature_names: list[str]) -> None:
    registry = {
        "active_model": {
            "name": model_name,
            "features": feature_names
        }
    }
    (models_dir / "model_registry.json").write_text(json.dumps(registry, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IMS anomaly model (IsolationForest) and export artifacts.")
    parser.add_argument("--repo_root", type=str, default=".", help="Path to repo root (where backend/ lives).")
    parser.add_argument("--fs", type=float, default=20000.0, help="Sampling frequency (Hz). IMS default is 20000.")
    parser.add_argument("--contamination", type=float, default=0.05, help="Expected anomaly ratio for IsolationForest.")
    parser.add_argument("--random_state", type=int, default=42)

    # Your three runs:
    parser.add_argument("--first", type=str, default="first_test", help="Folder name under data/raw for first run.")
    parser.add_argument("--second", type=str, default="second_test", help="Folder name under data/raw for second run.")
    parser.add_argument("--third", type=str, default="third_test", help="Folder name under data/raw for third run.")

    # Train only on early portion to represent "healthy" behavior (recommended)
    parser.add_argument("--train_frac", type=float, default=0.6, help="Fraction of each run used for training (early files).")

    # Speed control while developing
    parser.add_argument("--max_files", type=int, default=0, help="If >0, limit number of files per run (debug).")

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    paths = resolve_paths(repo_root)

    raw_dir = paths.raw_dir
    first_dir = raw_dir / args.first
    second_dir = raw_dir / args.second
    third_dir = raw_dir / args.third

    for d in [first_dir, second_dir, third_dir]:
        if not d.exists():
            raise FileNotFoundError(f"Missing dataset folder: {d}")

    max_files = args.max_files if args.max_files and args.max_files > 0 else None

    # Build X for each run
    X1, names1 = build_feature_matrix(first_dir, fs=args.fs, max_files=max_files)
    X2, names2 = build_feature_matrix(second_dir, fs=args.fs, max_files=max_files)
    X3, names3 = build_feature_matrix(third_dir, fs=args.fs, max_files=max_files)

    # Align dimensions across runs (handles 8-channel vs 4-channel differences)
    X1a, X2a, X3a = align_channel_features([X1, X2, X3])
    common_dim = X1a.shape[1]

    # Take feature names from the run with the longest list, then truncate to common_dim
    base_names = max([names1, names2, names3], key=len)
    feature_names = truncate_feature_names(base_names, common_dim)

    # Build training set from early fraction of each run
    def early_slice(X: np.ndarray, frac: float) -> np.ndarray:
        n = X.shape[0]
        k = max(1, int(n * frac))
        return X[:k, :]

    X_train = np.vstack([
        early_slice(X1a, args.train_frac),
        early_slice(X2a, args.train_frac),
        early_slice(X3a, args.train_frac),
    ])

    print(f"[train] Loaded features:")
    print(f"  first : {X1a.shape}")
    print(f"  second: {X2a.shape}")
    print(f"  third : {X3a.shape}")
    print(f"[train] Using common feature dim: {common_dim}")
    print(f"[train] Training on early fraction {args.train_frac:.2f} -> X_train: {X_train.shape}")

    # Train model pipeline
    pipe = train_isolation_forest(
        X_train,
        random_state=args.random_state,
        contamination=args.contamination,
    )

    # Export artifacts for API
    models_dir = paths.models_dir
    model_path = models_dir / "model.pkl"
    preproc_path = models_dir / "preprocessor.pkl"

    # Save entire pipeline as model.pkl OR save components separately
    # Your API currently loads model.pkl and (optionally) preprocessor.pkl.
    scaler = pipe.named_steps["scaler"]
    model = pipe.named_steps["model"]

    joblib.dump(model, model_path)
    joblib.dump(scaler, preproc_path)

    save_registry(models_dir, model_name="ims_isoforest_v1", feature_names=feature_names)

    print("[train] Saved artifacts:")
    print(f"  model       -> {model_path}")
    print(f"  preprocessor-> {preproc_path}")
    print(f"  registry    -> {models_dir / 'model_registry.json'}")


if __name__ == "__main__":
    main()