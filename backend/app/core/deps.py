from functools import lru_cache
from app.models.model_loader import load_artifacts
from app.models.artifacts import ModelArtifacts

@lru_cache()
def get_artifacts() -> ModelArtifacts:
    return load_artifacts()