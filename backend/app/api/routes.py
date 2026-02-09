from fastapi import APIRouter

from backend.app.api.health import router as health_router
from backend.app.api.jobs import router as jobs_router
from backend.app.api.predict import router as predict_router

api_router = APIRouter()

api_router.include_router(health_router)
api_router.include_router(jobs_router)
api_router.include_router(predict_router)