from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.core.redis import redis_client


@dataclass(frozen=True)
class QueueNames:
    """
    Centralizes Redis queue/list names.
    """
    jobs: str = "jobs:queue"   # FIFO queue for job IDs


class RedisQueue:
    """
    Simple Redis LIST-based FIFO queue.

    Producer (API):
      - enqueue(job_id)

    Consumer (worker):
      - dequeue(block=True)  -> waits for a job
    """

    def __init__(self, names: QueueNames = QueueNames()):
        self.names = names

    def enqueue(self, job_id: str) -> None:
        # FIFO pattern: producer pushes to the right
        redis_client.rpush(self.names.jobs, job_id)

    def dequeue(self, block: bool = True, timeout: int = 5) -> Optional[str]:
        """
        Returns a job_id or None.

        If block=True, uses BLPOP with timeout seconds (waits up to timeout).
        If block=False, uses LPOP (immediate).
        """
        if block:
            item = redis_client.blpop(self.names.jobs, timeout=timeout)
            # blpop returns (queue_name, value) or None
            if not item:
                return None
            _, job_id = item
            return job_id
        else:
            return redis_client.lpop(self.names.jobs)

    def size(self) -> int:
        return int(redis_client.llen(self.names.jobs))

    def clear(self) -> None:
        redis_client.delete(self.names.jobs)