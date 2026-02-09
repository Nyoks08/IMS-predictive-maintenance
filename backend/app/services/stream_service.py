from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4
import json
import time

from app.core.redis import redis_client
from app.services.inference import InferenceService
from app.services.redis_queue import RedisQueue


@dataclass
class JobInfo:
    job_id: str
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None


class StreamService:
    """
    Redis-backed async job orchestration using a real FIFO queue.

    API side:
      - submit_job(features) -> job_id (and enqueues job_id)

    Worker side:
      - process_next_job(block=True) -> processes the next queued job
    """

    def __init__(self):
        self.inference = InferenceService()
        self.queue = RedisQueue()

    # ---------- Key helpers ----------

    def _status_key(self, job_id: str) -> str:
        return f"job:{job_id}:status"

    def _payload_key(self, job_id: str) -> str:
        return f"job:{job_id}:payload"

    def _result_key(self, job_id: str) -> str:
        return f"job:{job_id}:result"

    def _error_key(self, job_id: str) -> str:
        return f"job:{job_id}:error"

    def _created_key(self, job_id: str) -> str:
        return f"job:{job_id}:created_at"

    # ---------- API methods ----------

    def submit_job(self, features: dict[str, float]) -> str:
        job_id = str(uuid4())

        redis_client.set(self._status_key(job_id), "PENDING")
        redis_client.set(self._payload_key(job_id), json.dumps(features))
        redis_client.set(self._created_key(job_id), str(time.time()))

        # enqueue job_id for workers (FIFO)
        self.queue.enqueue(job_id)

        return job_id

    def get_status(self, job_id: str) -> JobInfo:
        status = redis_client.get(self._status_key(job_id))
        if status is None:
            return JobInfo(job_id=job_id, status="NOT_FOUND")

        result_raw = redis_client.get(self._result_key(job_id))
        err = redis_client.get(self._error_key(job_id))

        result = json.loads(result_raw) if result_raw else None
        return JobInfo(job_id=job_id, status=status, result=result, error=err)

    # ---------- Worker methods ----------

    def process_job(self, job_id: str) -> JobInfo:
        """
        Process a specific job_id (used by dev endpoint /jobs/{id}/process).
        """
        status_key = self._status_key(job_id)

        status = redis_client.get(status_key)
        if status is None:
            return JobInfo(job_id=job_id, status="NOT_FOUND")

        if status not in ("PENDING", "RUNNING"):
            return self.get_status(job_id)

        # claim the job
        redis_client.set(status_key, "RUNNING")

        try:
            payload_raw = redis_client.get(self._payload_key(job_id))
            if not payload_raw:
                raise ValueError("Missing payload")

            features = json.loads(payload_raw)
            score = float(self.inference.predict(features))

            if score >= 0.8:
                severity = "HIGH"
            elif score >= 0.6:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            result = {"anomaly_score": score, "severity": severity}

            redis_client.set(self._result_key(job_id), json.dumps(result))
            redis_client.delete(self._error_key(job_id))
            redis_client.set(status_key, "DONE")

            return JobInfo(job_id=job_id, status="DONE", result=result)

        except Exception as e:
            redis_client.set(self._error_key(job_id), str(e))
            redis_client.set(status_key, "FAILED")
            return JobInfo(job_id=job_id, status="FAILED", error=str(e))

    def process_next_job(self, block: bool = True, timeout: int = 5) -> Optional[JobInfo]:
        """
        Worker-friendly: Pop the next job_id from the queue and process it.
        Returns JobInfo, or None if no job was available (timeout / empty queue).
        """
        job_id = self.queue.dequeue(block=block, timeout=timeout)
        if not job_id:
            return None
        return self.process_job(job_id)