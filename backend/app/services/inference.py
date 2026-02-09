from pathlib import Path
import json
import joblib
import numpy as np

from app.services.preprocess import PreprocessService

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "app" / "models" / "model.pkl"
REGISTRY_PATH = BASE_DIR / "app" / "models" / "model_registry.json"
PREPROCESSOR_PATH = BASE_DIR / "app" / "models" / "preprocessor.pkl"  # optional


class InferenceService:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        if not REGISTRY_PATH.exists():
            raise FileNotFoundError("Model registry not found")

        self.model = joblib.load(MODEL_PATH)

        registry = json.loads(REGISTRY_PATH.read_text())
        self.feature_order = registry["active_model"]["features"]

        # Preprocessor is optional; only load if present
        self.preprocessor = joblib.load(PREPROCESSOR_PATH) if PREPROCESSOR_PATH.exists() else None

        self.preprocess = PreprocessService(
            feature_order=self.feature_order,
            preprocessor=self.preprocessor,
        )

    def predict(self, features: dict[str, float]) -> float:
        prepared = self.preprocess.prepare(features)  # prepared.X is (1, n_features)

        y = self.model.predict(prepared.X)
        return float(np.asarray(y).ravel()[0])