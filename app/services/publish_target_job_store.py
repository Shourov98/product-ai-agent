from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from app.schemas.response import PublishTargetAnalysisJobResponse, PublishTargetAnalysisResponse


@dataclass(slots=True)
class PublishTargetJobRecord:
    job_id: str
    user_id: str | None
    product_id: str
    marketplace: str
    status: str
    result: PublishTargetAnalysisResponse | None
    error: str | None
    created_at: str
    updated_at: str


class PublishTargetJobStore:
    _jobs: dict[str, PublishTargetJobRecord] = {}

    @classmethod
    def create(
        cls,
        product_id: str,
        marketplace: str,
        *,
        user_id: str | None = None,
        result: PublishTargetAnalysisResponse | None = None,
    ) -> PublishTargetJobRecord:
        now = cls._timestamp()
        record = PublishTargetJobRecord(
            job_id=uuid4().hex,
            user_id=user_id,
            product_id=product_id,
            marketplace=marketplace,
            status="pending",
            result=result,
            error=None,
            created_at=now,
            updated_at=now,
        )
        cls._jobs[record.job_id] = record
        return record

    @classmethod
    def get(cls, job_id: str) -> PublishTargetJobRecord | None:
        return cls._jobs.get(job_id)

    @classmethod
    def update_running(cls, job_id: str) -> PublishTargetJobRecord | None:
        record = cls._jobs.get(job_id)
        if record is None:
            return None
        record.status = "running"
        record.updated_at = cls._timestamp()
        return record

    @classmethod
    def update_completed(cls, job_id: str, result: PublishTargetAnalysisResponse) -> PublishTargetJobRecord | None:
        record = cls._jobs.get(job_id)
        if record is None:
            return None
        record.status = "completed"
        record.result = result
        record.error = None
        record.updated_at = cls._timestamp()
        return record

    @classmethod
    def update_failed(cls, job_id: str, error: str) -> PublishTargetJobRecord | None:
        record = cls._jobs.get(job_id)
        if record is None:
            return None
        record.status = "failed"
        record.error = error
        record.updated_at = cls._timestamp()
        return record

    @staticmethod
    def to_response(record: PublishTargetJobRecord) -> PublishTargetAnalysisJobResponse:
        return PublishTargetAnalysisJobResponse(
            job_id=record.job_id,
            product_id=record.product_id,
            marketplace=record.marketplace,
            status=record.status,  # type: ignore[arg-type]
            result=record.result,
            error=record.error,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(UTC).isoformat()
