import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import HTTPException
from pydantic import ValidationError

from api.schemas import JobRequest, JobStatus
from common.registries import JobRecord, jobs_registry
from common.services import SERVICES
from common.utils import check_mock_exists, mark_job_complete, persist_job_output
from constants import JOB_COMPLETE_MARKER, MOCK_OUTPUT_DIR, VOLUME_MOCKS_DIR, JobState
from core import volume


def reload_volume() -> None:
    """Refresh container's view of the volume.

    Writes committed by other containers stay invisible until a reload, so reads
    of job data can otherwise return a stale snapshot.
    """
    volume.reload()


def validate_job_request(job: JobRequest) -> None:
    """Validate a job request before any work is spawned.

    Args:
        job: The job to check.

    Raises:
        HTTPException: If the service is unknown, mocking is requested but
            unavailable, or the params fail the service's input model.
    """
    if job.service not in SERVICES:
        raise HTTPException(400, f"Unknown service '{job.service}'. Options: {list(SERVICES)}")
    if job.mock and not check_mock_exists(job.service):
        raise HTTPException(501, f"Mock not available for service '{job.service}'")
    try:
        SERVICES[job.service].params(**job.params)
    except ValidationError as e:
        raise HTTPException(422, f"Invalid params for '{job.name or job.service}': {e}")


def spawn_job(job: JobRequest, batch_id: str | None = None) -> JobStatus:
    """Run one validated job and record it in the registry.

    Real jobs spawn a Modal call. Mock jobs carry no call_id: the service's
    fixture is copied into the job dir and the job is marked complete, so a
    download returns a representative full result without any compute. The
    fixture's own run log is carried over because mark_job_complete rewrites it,
    and make_mocks guarantees that log is present.

    Args:
        job: The validated job to run.
        batch_id: The batch this job belongs to, or None for a standalone job.

    Returns:
        The job's initial status, carrying its server-generated id.
    """
    service_entry = SERVICES[job.service]
    job_id = uuid.uuid4().hex
    if job.mock:
        mock_dir = Path(VOLUME_MOCKS_DIR) / job.service / MOCK_OUTPUT_DIR
        persist_job_output(job_id, mock_dir)
        mock_log = (mock_dir / JOB_COMPLETE_MARKER).read_text()
        mark_job_complete(job_id, mock_log)
        call_id = None
    else:
        params = service_entry.params(**job.params)
        call_id = service_entry.run.spawn(job_id=job_id, job_name=job.name, params=params.model_dump()).object_id
    created_at = time.time()
    job_record: JobRecord = {
        "name": job.name,
        "service": job.service,
        "call_id": call_id,
        "created_at": created_at,
        "batch_id": batch_id,
    }
    jobs_registry[job_id] = job_record
    return JobStatus(
        id=job_id,
        name=job.name,
        service=job.service,
        status=JobState.PENDING,
        created_at=datetime.fromtimestamp(created_at, tz=UTC),
        batch_id=batch_id,
    )
