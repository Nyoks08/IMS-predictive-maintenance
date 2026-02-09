from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.inference import InferenceService

router = APIRouter(prefix="/predict", tags=["Prediction"])

# Initialize service once (model loads once)
inference_service = InferenceService()

class PredictRequest(BaseModel):
    features: dict[str, float]

class PredictResponse(BaseModel):
    anomaly_score: float
    severity: str

@router.post("/", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        score = inference_service.predict(req.features)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid or missing feature keys")

    if score >= 0.8:
        severity = "HIGH"
    elif score >= 0.6:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return PredictResponse(
        anomaly_score=float(score),
        severity=severity
    )
