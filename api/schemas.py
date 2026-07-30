from datetime import datetime

from pydantic import BaseModel

from constants import BatchState, JobState


class JobRequest(BaseModel):
    service: str
    params: dict
    name: str | None = None
    mock: bool = False


class JobStatus(BaseModel):
    id: str
    name: str | None = None
    service: str
    status: JobState
    created_at: datetime
    batch_id: str | None = None


class JobStopResponse(BaseModel):
    stopped: str  # id of the stopped job


class JobDeleteResponse(BaseModel):
    deleted: str  # id of the deleted job


class BatchRequest(BaseModel):
    service: str
    params: list[dict]  # one param set per job, all run the same service
    name: str | None = None  # batch label, jobs are named "{name}-{i}"
    mock: bool = False


class BatchSubmitResponse(BaseModel):
    batch_id: str
    jobs: list[JobStatus]


class BatchStatus(BaseModel):
    id: str
    created_at: datetime
    status: BatchState
    counts: dict
    jobs: list[JobStatus]


class BatchStopResponse(BaseModel):
    stopped: str  # the batch id
    jobs: list[str]  # ids of the jobs that were stopped


class BatchDeleteResponse(BaseModel):
    deleted: str  # the batch id
    jobs: list[str]  # ids of the jobs that were deleted


class ServiceInfo(BaseModel):
    name: str
    description: str
    params_schema: dict
