from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def read_ims_txt(file_path: Path) -> np.ndarray:
    """
    IMS bearing files are typically 2 columns (two accelerometers) with whitespace separation.
    Returns: array shape (n_samples, n_channels)
    """
    arr = np.loadtxt(file_path)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr

def window_signal(x: np.ndarray, win_size: int, step: int) -> list[np.ndarray]:
    windows = []
    n = x.shape[0]
    for start in range(0, n - win_size + 1, step):
        windows.append(x[start:start + win_size, :])
    return windows

def compute_features(w: np.ndarray) -> dict:
    """
    Window-level features per channel + a combined energy feature.
    No scipy dependency (keeps install light for tonight).
    """
    feats: dict[str, float] = {}
    for ch in range(w.shape[1]):
        s = w[:, ch]
        mu = float(np.mean(s))
        sd = float(np.std(s)) + 1e-9
        feats[f"ch{ch+1}_mean"] = mu
        feats[f"ch{ch+1}_std"] = float(np.std(s))
        feats[f"ch{ch+1}_rms"] = float(np.sqrt(np.mean(s**2)))
        feats[f"ch{ch+1}_p2p"] = float(np.max(s) - np.min(s))
        # kurtosis (no scipy): E[((x-mu)/sd)^4]
        feats[f"ch{ch+1}_kurtosis"] = float(np.mean(((s - mu) / sd) ** 4))

    feats["energy"] = float(np.mean(w**2))
    return feats

def label_by_last_fraction(file_index: int, total_files: int, last_frac: float = 0.15) -> int:
    """
    Weak-supervision labeling:
    early files = normal (0), last X% files = anomalous (1).
    This mirrors bearing degradation across time in IMS runs.
    """
    cutoff = int((1.0 - last_frac) * total_files)
    return 1 if file_index >= cutoff else 0

def find_test_dirs(raw_dir: Path) -> list[Path]:
    """
    Your folders are: data/raw/first test, second test, third test
    (with spaces). We handle them safely.
    """
    dirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    # sort stable
    return sorted(dirs, key=lambda p: p.name.lower())

def collect_txt_files(test_dir: Path) -> list[Path]:
    txt = sorted(test_dir.glob("*.txt"))
    if txt:
        return txt
    # sometimes nested
    return sorted(test_dir.rglob("*.txt"))

def build_dataset(
    raw_dir: Path,
    out_csv: Path,
    win_size: int = 1024,
    step: int = 512,
    last_frac: float = 0.15,
    max_files_per_test: int | None = 60,
) -> None:
    """
    Builds a feature dataset CSV from IMS raw txt files.

    max_files_per_test is a SPEED CONTROL for tonight.
    - Keep it at 60 for fast results.
    - Set to None later for full dataset.
    """
    rows = []
    test_dirs = find_test_dirs(raw_dir)

    for test_dir in test_dirs:
        txt_files = collect_txt_files(test_dir)
        if not txt_files:
            print(f"[WARN] No .txt files found under: {test_dir}")
            continue

        if max_files_per_test is not None:
            txt_files = txt_files[:max_files_per_test]

        total_files = len(txt_files)
        print(f"[INFO] {test_dir.name}: using {total_files} files")

        for i, fp in enumerate(txt_files):
            x = read_ims_txt(fp)
            windows = window_signal(x, win_size=win_size, step=step)
            y = label_by_last_fraction(i, total_files, last_frac=last_frac)

            for w_idx, w in enumerate(windows):
                feats = compute_features(w)
                feats.update({
                    "test_dir": test_dir.name,
                    "file_name": fp.name,
                    "file_index": i,
                    "window_index": w_idx,
                    "label": y
                })
                rows.append(feats)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[DONE] Saved dataset → {out_csv} | rows={len(df)} cols={df.shape[1]}")

if __name__ == "__main__":
    # IMPORTANT: this is correct for your repo (backend/src/ingestion/... -> parents[3] is repo root)
    project_root = Path(__file__).resolve().parents[3]
    raw_dir = project_root / "data" / "raw"
    out_csv = project_root / "data" / "processed" / "ims_features.csv"

    build_dataset(
        raw_dir=raw_dir,
        out_csv=out_csv,
        win_size=1024,
        step=512,
        last_frac=0.15,
        max_files_per_test=60,   # keep fast tonight; set None later
    )