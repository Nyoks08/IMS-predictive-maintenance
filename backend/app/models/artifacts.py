from dataclasses import dataclass
from sklearn.base import BaseEstimator
from typing import Any

@dataclass
class ModelArtifacts:
    model: BaseEstimator
    preprocessor: Any