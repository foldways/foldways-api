from enum import StrEnum
from typing import TypedDict

import modal


class JobState(StrEnum):
    """The status values a single job can report."""

    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"
    INIT_FAILED = "init_failed"
    STOPPED = "stopped"
    TIMED_OUT = "timed_out"
    UNKNOWN = "unknown"


class BatchState(StrEnum):
    """The status values a batch can report, aggregated from its jobs."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    COMPLETED_WITH_FAILURES = "completed_with_failures"


class JobRecord(TypedDict):
    """What a jobs_registry entry holds, keyed by job id."""

    name: str | None
    service: str
    call_id: str | None  # None for mock jobs, which have no Modal call
    created_at: float
    batch_id: str | None


class BatchRecord(TypedDict):
    """What a batches_registry entry holds, keyed by batch id."""

    created_at: float
    job_ids: list[str]


jobs_registry = modal.Dict.from_name("foldways-jobs", create_if_missing=True)
batches_registry = modal.Dict.from_name("foldways-batches", create_if_missing=True)
