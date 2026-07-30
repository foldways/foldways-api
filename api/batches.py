import logging
import time
import uuid
from collections import Counter
from datetime import UTC, datetime

import modal
from fastapi import APIRouter, Depends, HTTPException

from api.schemas import (
    BatchRequest,
    BatchStatus,
    BatchSubmitResponse,
    JobRequest,
    JobStatus,
)
from api.utils import reload_volume, spawn_job, validate_job_request
from common.registries import BatchRecord, JobRecord, batches_registry, jobs_registry
from common.utils import delete_job_data, get_batch_status, get_job_status
from constants import JobState

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(reload_volume)])


@router.post("/batches", response_model=BatchSubmitResponse)
def submit_batch(request: BatchRequest):
    """Submit a sweep: run one service over many param sets.

    Each param set becomes an independent Modal call under a shared batch id,
    named "{name}-{i}" when a batch name is given. Returns the batch id plus a
    server-generated id per job; use either for status and deletion, and each
    job id with GET /jobs/{job_id}/download to fetch that job's output.
    """
    jobs = [
        JobRequest(
            service=request.service,
            params=params,
            name=f"{request.name}-{i}" if request.name else None,
            mock=request.mock,
        )
        for i, params in enumerate(request.params)
    ]
    # Validate every job before spawning any, so a bad request spawns no work.
    for job in jobs:
        validate_job_request(job)
    batch_id = uuid.uuid4().hex
    statuses = [spawn_job(job, batch_id) for job in jobs]
    batch_record: BatchRecord = {
        "created_at": time.time(),
        "job_ids": [s.id for s in statuses],
    }
    batches_registry[batch_id] = batch_record
    logger.info(f"Submitted batch={batch_id} jobs={len(statuses)}")
    return BatchSubmitResponse(batch_id=batch_id, jobs=statuses)


@router.get("/batches/{batch_id}", response_model=BatchStatus)
def get_batch(batch_id: str):
    """Get a batch's aggregate status and the status of each job in it."""
    if batch_id not in batches_registry:
        raise HTTPException(404, f"No such batch: '{batch_id}'")
    job_state_counts: Counter[JobState] = Counter()
    jobs = []
    for job_id in batches_registry[batch_id]["job_ids"]:
        if job_id not in jobs_registry:  # a child may have been deleted on its own
            continue
        job_record: JobRecord = jobs_registry[job_id]
        status = get_job_status(job_id, job_record)
        job_state_counts[status] += 1
        jobs.append(
            JobStatus(
                id=job_id,
                name=job_record["name"],
                service=job_record["service"],
                status=status,
                created_at=datetime.fromtimestamp(job_record["created_at"], tz=UTC),
                batch_id=job_record["batch_id"],
            )
        )
    return BatchStatus(
        id=batch_id,
        created_at=datetime.fromtimestamp(batches_registry[batch_id]["created_at"], tz=UTC),
        status=get_batch_status(job_state_counts),
        counts={"total": sum(job_state_counts.values()), **job_state_counts},
        jobs=jobs,
    )


@router.post("/batches/{batch_id}/stop")
def stop_batch(batch_id: str):
    """Stop all running jobs in a batch without deleting them."""
    if batch_id not in batches_registry:
        raise HTTPException(404, f"No such batch: '{batch_id}'")
    stopped = []
    for job_id in batches_registry[batch_id]["job_ids"]:
        if job_id not in jobs_registry:  # a child may have been deleted on its own
            continue
        job_record: JobRecord = jobs_registry[job_id]
        if job_record["call_id"] is None:  # mock jobs have no Modal call to cancel
            continue
        try:
            modal.functions.FunctionCall.from_id(job_record["call_id"]).cancel()
            stopped.append(job_id)
        except Exception as e:
            logger.warning(f"Cancel failed for job={job_id}: {e}")
    logger.info(f"Stopped batch={batch_id} jobs={len(stopped)}")
    return {"stopped": batch_id, "jobs": stopped}


@router.delete("/batches/{batch_id}")
def delete_batch(batch_id: str):
    """Stop a batch and delete every job in it along with their stored data."""
    if batch_id not in batches_registry:
        raise HTTPException(404, f"No such batch: '{batch_id}'")
    deleted = []
    for job_id in batches_registry[batch_id]["job_ids"]:
        if job_id not in jobs_registry:  # a child may have been deleted on its own
            continue
        job_record: JobRecord = jobs_registry[job_id]
        if job_record["call_id"] is not None:  # mock jobs have no Modal call to cancel
            try:
                modal.functions.FunctionCall.from_id(job_record["call_id"]).cancel()
            except Exception as e:
                logger.warning(f"Cancel during delete failed for job={job_id}: {e}")
        delete_job_data(job_id)
        del jobs_registry[job_id]
        deleted.append(job_id)
    del batches_registry[batch_id]
    logger.info(f"Deleted batch={batch_id} jobs={len(deleted)}")
    return {"deleted": batch_id, "jobs": deleted}
