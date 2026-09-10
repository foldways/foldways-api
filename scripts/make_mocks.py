"""Record a service's mock fixture from a real run against a deployed API.

Submits the service's committed request, waits for the job to finish, then
replaces the fixture with the job's downloaded output. This runs real compute on
a GPU, so it is meant to be run rarely and deliberately, not as part of a test.

Takes the deployment URL from --url. The recorded fixture overwrites whatever was
there before, so review the diff before committing it.

Run it as a module from the repository root, so the project's imports resolve:

    uv run python -m scripts.make_mocks boltz2 --url https://<workspace>--foldways-api.modal.run
"""

import argparse
import io
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import httpx2

from common.registries import JobState
from constants import JOB_COMPLETE_MARKER, LOCAL_MOCKS_DIR, MINUTES_40, MOCK_OUTPUT_DIR, MOCK_REQUEST_FILE

POLL_INTERVAL_SECONDS = 10
JOB_TIMEOUT_SECONDS = MINUTES_40


def main(service: str, url: str) -> None:
    """Record the fixture for one service, named as its mocks/ subdirectory."""
    service_dir = Path(__file__).parent.parent / LOCAL_MOCKS_DIR / service
    request_path = service_dir / MOCK_REQUEST_FILE
    if not request_path.exists():
        sys.exit(f"No request at {request_path}. Write one before recording a fixture.")

    with httpx2.Client(base_url=url.rstrip("/"), timeout=60) as client:
        response = client.post("/jobs", json=json.loads(request_path.read_text()))
        response.raise_for_status()
        job_id = response.json()["id"]
        print(f"Submitted job {job_id}, polling every {POLL_INTERVAL_SECONDS}s")

        deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
        while True:
            if time.monotonic() > deadline:
                sys.exit(f"Job {job_id} did not finish within {JOB_TIMEOUT_SECONDS}s")
            try:
                response = client.get(f"/jobs/{job_id}")
                response.raise_for_status()
            except httpx2.HTTPStatusError as e:
                if e.response.status_code < 500:
                    raise
                print(f"Transient {e.response.status_code} polling job {job_id}, retrying")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            except httpx2.RequestError as e:
                print(f"Transient network error polling job {job_id} ({e}), retrying")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            status = response.json()["status"]
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

    if not (output_dir / JOB_COMPLETE_MARKER).exists():
        sys.exit(f"Recorded output has no {JOB_COMPLETE_MARKER}, so the fixture would not work.")

    files = sorted(p.relative_to(output_dir) for p in output_dir.rglob("*") if p.is_file())
    size_kb = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file()) / 1024
    print(f"Recorded {len(files)} files ({size_kb:.0f} KB) to {output_dir}")
    for path in files:
        print(f"  {path}")
    print("Run `uv run modal run setup_artifacts.py` to stage it on the volume.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a service's mock fixture from a real run.")
    parser.add_argument("service", help="Service name, matching its mocks/ subdirectory.")
    parser.add_argument("--url", required=True, help="Deployment URL, e.g. https://<workspace>--foldways-api.modal.run")
    args = parser.parse_args()
    main(args.service, args.url)
