import logging
from datetime import UTC, datetime

import modal
from fastapi import APIRouter, Depends, HTTPException, Response

from api.schemas import JobRequest, JobStatus
from api.utils import reload_volume, spawn_job, validate_job_request
from common.registries import JobRecord, jobs_registry
from common.utils import check_job_complete, delete_job_data, get_job_status, zip_job_data

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(reload_volume)])


@router.post("/jobs", response_model=JobStatus)
def submit_job(job: JobRequest):
    """Submit a single job naming a service and its params.

    The job runs as an independent Modal call with no parent batch. Returns a
    server-generated id used for status, download, and deletion.
    """
    validate_job_request(job)
    status = spawn_job(job)
    logger.info(f"Submitted job={status.id}")
    return status


@router.get("/jobs", response_model=list[JobStatus])
def list_jobs():
    """List all jobs and their current status."""
    return [
        JobStatus(
            id=job_id,
            name=job_record["name"],
            service=job_record["service"],
            status=get_job_status(job_id, job_record),
            created_at=datetime.fromtimestamp(job_record["created_at"], tz=UTC),
            batch_id=job_record["batch_id"],
        )
        for job_id, job_record in jobs_registry.items()
    ]


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    """Get a single job's status."""
    if job_id not in jobs_registry:
        raise HTTPException(404, f"No such job: '{job_id}'")
    job_record: JobRecord = jobs_registry[job_id]
    return JobStatus(
        id=job_id,
        name=job_record["name"],
        service=job_record["service"],
        status=get_job_status(job_id, job_record),
        created_at=datetime.fromtimestamp(job_record["created_at"], tz=UTC),
        batch_id=job_record["batch_id"],
    )


@router.get("/jobs/{job_id}/download")
def download_job(job_id: str):
    """Download a completed job's output directory as a zip archive.

    Errors with 409 if the output isn't ready yet (job pending or failed).
    """
    if job_id not in jobs_registry:
        raise HTTPException(404, f"No such job: '{job_id}'")
    job_record: JobRecord = jobs_registry[job_id]
    if not check_job_complete(job_id):
        status = get_job_status(job_id, job_record)
        raise HTTPException(409, f"Output not available for '{job_id}' (status={status})")
    filename = f"{job_record['name'] or job_id}.zip"
    return Response(
        content=zip_job_data(job_id),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/jobs/{job_id}/stop")
def stop_job(job_id: str):
    """Stop a running job without deleting it or its data."""
    if job_id not in jobs_registry:
        raise HTTPException(404, f"No such job: '{job_id}'")
    call_id = jobs_registry[job_id]["call_id"]
    if call_id is not None:  # mock jobs have no Modal call to cancel
        modal.functions.FunctionCall.from_id(call_id).cancel()
    logger.info(f"Stopped job={job_id}")
    return {"stopped": job_id}


@router.delete("/jobs/{job_id}")
def stop_and_delete_job(job_id: str):
    """Stop a job if running and delete it along with its stored data."""
    if job_id not in jobs_registry:
        raise HTTPException(404, f"No such job: '{job_id}'")
    call_id = jobs_registry[job_id]["call_id"]
    if call_id is not None:  # mock jobs have no Modal call to cancel
        try:
            modal.functions.FunctionCall.from_id(call_id).cancel()
        except Exception as e:
            logger.warning(f"Cancel during delete failed for job={job_id}: {e}")

    delete_job_data(job_id)

    del jobs_registry[job_id]
    logger.info(f"Deleted job={job_id}")
    return {"deleted": job_id}
