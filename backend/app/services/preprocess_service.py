from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class PreparedInput:
    """Output of preprocessing ready for model inference."""
    vector: list[float]
    X: np.ndarray  # shape (1, n_features)


class PreprocessService:
    """
    Turns an input features dict into an ordered numeric vector (and numpy array),
    using the feature order defined in model_registry.json.
    Optionally applies a fitted sklearn preprocessor if provided.
    """

    def __init__(self, feature_order: list[str], preprocessor: Any | None = None):
        if not feature_order:
            raise ValueError("feature_order is empty")
        self.feature_order = feature_order
        self.preprocessor = preprocessor

    def to_vector(self, features: dict[str, float]) -> list[float]:
        # Fill missing features with 0.0 to keep inference robust.
        # (You can change this to raise if you want strict behavior.)
        return [float(features.get(name, 0.0)) for name in self.feature_order]

    def prepare(self, features: dict[str, float]) -> PreparedInput:
        vector = self.to_vector(features)
        X = np.array(vector, dtype=np.float32).reshape(1, -1)

        # Optional sklearn preprocessing (StandardScaler, etc.)
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)

        return PreparedInput(vector=vector, X=X)