from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.stream_service import StreamService

router = APIRouter(prefix="/jobs", tags=["Jobs"])

stream_service = StreamService()


class SubmitJobRequest(BaseModel):
    features: dict[str, float]


class SubmitJobResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: dict | None = None
    error: str | None = None


@router.post("/submit", response_model=SubmitJobResponse)
def submit_job(req: SubmitJobRequest):
    try:
        job_id = stream_service.submit_job(req.features)
        return SubmitJobResponse(job_id=job_id, status="PENDING")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    info = stream_service.get_status(job_id)

    if info.status == "NOT_FOUND":
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=info.job_id,
        status=info.status,
        result=info.result,
        error=info.error,
    )


@router.post("/{job_id}/process", response_model=JobStatusResponse)
def process_job(job_id: str):
    """
    Temporary endpoint for development:
    Processes the job immediately using the same API server process.

    In production, you'd remove this and run a separate worker that calls
    stream_service.process_job(job_id) (or consumes a queue).
    """
    info = stream_service.process_job(job_id)

    if info.status == "NOT_FOUND":
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=info.job_id,
        status=info.status,
        result=info.result,
        error=info.error,
    )