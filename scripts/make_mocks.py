"""Record a service's mock fixture from a real run against a deployed API.

Submits the service's committed request, waits for the job to finish, then
replaces the fixture with the job's downloaded output. This runs real compute on
a GPU, so it is meant to be run rarely and deliberately, not as part of a test.

Reads the deployment URL from FOLDWAYS_API_URL. The recorded fixture overwrites
whatever was there before, so review the diff before committing it.

Run it as a module from the repository root, so the project's imports resolve:

    uv run python -m scripts.make_mocks boltz2
"""

import io
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import httpx2

from constants import JOB_COMPLETE_MARKER, LOCAL_MOCKS_DIR, MINUTES_40, MOCK_OUTPUT_DIR, MOCK_REQUEST_FILE, JobState

POLL_INTERVAL_SECONDS = 10
JOB_TIMEOUT_SECONDS = MINUTES_40


def main(service: str) -> None:
    """Record the fixture for one service, named as its mocks/ subdirectory."""
    api_url = os.environ.get("FOLDWAYS_API_URL")
    if not api_url:
        sys.exit("FOLDWAYS_API_URL is not set. Point it at your deployment.")

    service_dir = Path(__file__).parent.parent / LOCAL_MOCKS_DIR / service
    request_path = service_dir / MOCK_REQUEST_FILE
    if not request_path.exists():
        sys.exit(f"No request at {request_path}. Write one before recording a fixture.")

    with httpx2.Client(base_url=api_url.rstrip("/"), timeout=60) as client:
        response = client.post("/jobs", json=json.loads(request_path.read_text()))
        response.raise_for_status()
        job_id = response.json()["id"]
        print(f"Submitted job {job_id}, polling every {POLL_INTERVAL_SECONDS}s")

        deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
        while True:
            if time.monotonic() > deadline:
                sys.exit(f"Job {job_id} did not finish within {JOB_TIMEOUT_SECONDS}s")
            status = client.get(f"/jobs/{job_id}").raise_for_status().json()["status"]
            if status == JobState.COMPLETE:
                break
            if status != JobState.PENDING:
                sys.exit(f"Job {job_id} ended as '{status}'. Check the Modal logs.")
            time.sleep(POLL_INTERVAL_SECONDS)

        archive = client.get(f"/jobs/{job_id}/download").raise_for_status().content

    output_dir = service_dir / MOCK_OUTPUT_DIR
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    zipfile.ZipFile(io.BytesIO(archive)).extractall(output_dir)

    # Without the marker a mock job never reads as complete, so the fixture would
    # be staged but unusable.
    if not (output_dir / JOB_COMPLETE_MARKER).exists():
        sys.exit(f"Recorded output has no {JOB_COMPLETE_MARKER}, so the fixture would not work.")

    files = sorted(p.relative_to(output_dir) for p in output_dir.rglob("*") if p.is_file())
    size_kb = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file()) / 1024
    print(f"Recorded {len(files)} files ({size_kb:.0f} KB) to {output_dir}")
    for path in files:
        print(f"  {path}")
    print("Run `uv run modal run setup_artifacts.py` to stage it on the volume.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: uv run python -m scripts.make_mocks <service>")
    main(sys.argv[1])
