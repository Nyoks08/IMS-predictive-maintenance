from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    Request payload for /predict.
    `features` should contain the already-computed feature values (not raw IMS waveform).
    Keys should match model_registry.json -> active_model.features
    """
    features: dict[str, float] = Field(default_factory=dict)


class PredictResponse(BaseModel):
    """
    Response payload for /predict.
    """
    anomaly_score: float
    severity: str