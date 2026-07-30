from typing import TypedDict

import modal


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
