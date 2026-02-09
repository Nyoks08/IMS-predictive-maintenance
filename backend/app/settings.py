from pathlib import Path
import os

# =========================
# Base paths
# =========================

# backend/
BASE_DIR = Path(__file__).resolve().parents[1]

APP_DIR = BASE_DIR / "app"
MODELS_DIR = APP_DIR / "models"

# =========================
# Model artifacts
# =========================

MODEL_PATH = MODELS_DIR / "model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
MODEL_REGISTRY_PATH = MODELS_DIR / "model_registry.json"

# =========================
# Redis configuration
# =========================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Redis key prefixes (keeps things organized)
JOB_STATUS_KEY = "job:{job_id}:status"
JOB_PAYLOAD_KEY = "job:{job_id}:payload"
JOB_RESULT_KEY = "job:{job_id}:result"
JOB_ERROR_KEY = "job:{job_id}:error"

# =========================
# Worker behavior
# =========================

# How often the worker polls for new jobs (seconds)
WORKER_POLL_INTERVAL = float(os.getenv("WORKER_POLL_INTERVAL", "1.0"))

# Job states
JOB_PENDING = "PENDING"
JOB_RUNNING = "RUNNING"
JOB_DONE = "DONE"
JOB_FAILED = "FAILED"

# =========================
# Safety checks
# =========================

def validate_paths() -> None:
    """
    Optional helper to fail fast if artifacts are missing.
    Call this once at worker startup.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    if not MODEL_REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Model registry not found at {MODEL_REGISTRY_PATH}"
        )