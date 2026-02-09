import joblib
from pathlib import Path
from .artifacts import ModelArtifacts

MODEL_DIR = Path(__file__).parent

def load_artifacts() -> ModelArtifacts:
    model = joblib.load(MODEL_DIR / "model.pkl")
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")

    return ModelArtifacts(
        model=model,
        preprocessor=preprocessor
    )