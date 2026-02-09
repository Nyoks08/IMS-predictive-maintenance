import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.serving.redis_queue import RedisClient
from src.serving.predictor import Predictor

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MODEL_PATH = os.getenv("MODEL_PATH", "models/artifacts/model.pkl")

app = FastAPI(title="Predictive Maintenance API", version="0.1.0")

redis_client = RedisClient(REDIS_URL)
predictor = Predictor(model_path=MODEL_PATH)

class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_items=5)

class PredictResponse(BaseModel):
    anomaly_score: float
    severity: str

class EventRequest(BaseModel):
    timestamp: str
    severity: str
    anomaly_score: float
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict-window", response_model=PredictResponse)
def predict_window(req: PredictRequest):
    score = predictor.predict_score(req.features)

    severity = "LOW"
    if score >= 0.8:
        severity = "HIGH"
    elif score >= 0.6:
        severity = "MEDIUM"

    return {"anomaly_score": float(score), "severity": severity}

@app.post("/event")
def log_event(req: EventRequest):
    redis_client.push_event(req.model_dump())
    return {"saved": True}