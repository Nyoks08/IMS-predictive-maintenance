from pathlib import Path
import numpy as np
import joblib

class Predictor:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.load()

    def load(self) -> None:
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)

    def predict_score(self, features: list[float]) -> float:
        x = np.array(features, dtype=float).reshape(1, -1)

        if self.model is None:
            # fallback (should not happen after training)
            return float(min(1.0, max(0.0, abs(x).mean() / 10.0)))

        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(x)[0, 1])

        return float(self.model.predict(x)[0])