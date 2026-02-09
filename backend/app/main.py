from fastapi import FastAPI
from app.api.routes import api_router

app = FastAPI(title="Realtime Predictive Maintenance")

app.include_router(api_router)