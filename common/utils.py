import io
import json
import shutil
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import modal
from modal.types import InputStatus

from common.registries import JobRecord
from constants import (
    JOB_COMPLETE_MARKER,
    JOB_ERROR_MARKER,
    MOCK_OUTPUT_DIR,
    VOLUME_JOBS_DIR,
    VOLUME_MOCKS_DIR,
    BatchState,
    JobState,
)
from core import volume


def format_run_log(
    job_id: str,
    job_name: str | None,
    service: str,
    service_spec: str,
    job_command: str,
    output: str,
    stderr: str,
    started_at: datetime,
) -> str:
    """Render a job's run log: a key:value header, then the tool's output.

    Every service formats its run log through here, so the header keeps the same
    fields and order as services are added. Each field is a named argument rather
    than a caller-built dict, so a service cannot omit or misname one.

    Args:
        job_id: Unique id identifying the job.
        job_name: The caller's label for the job, or None when none was supplied.
        service: The service that ran the job.
        service_spec: The pinned spec or version of the tool that ran (e.g. "boltz==2.1.1").
        job_command: The command the service invoked. Pass an empty string when
            the job failed before a command was built, and the line is left out.
        output: The tool's own output (stdout, or a failure detail).
        stderr: The tool's stderr. Pass an empty string when there is none, and
            the section is left out of the log.
        started_at: When the job began. completed_at is read now, and elapsed_s
            is derived from the two, so both the success and failure paths record
            timing without repeating the arithmetic.

    Returns:
        The header lines where each key:value pair is rendered on a separate
        line, followed by a separator, then the output, then stderr when present.
    """
    completed_at = datetime.now(UTC)
    header = {"job_id": job_id, "job_name": job_name, "service": service, "service_spec": service_spec}
    if job_command:
        header["job_command"] = job_command
    header["started_at"] = started_at.isoformat()
    header["completed_at"] = completed_at.isoformat()
    header["elapsed_s"] = f"{(completed_at - started_at).total_seconds():.1f}"
    lines = [f"{key}: {value}" for key, value in header.items()]
    separator = "=" * 40
    stderr_separator = "--- stderr ---"
    log = "\n".join(lines) + "\n" + separator + "\n" + output
    if stderr:
        log += "\n" + stderr_separator + "\n" + stderr
    return log


def mark_job_complete(job_id: str, log: str = "") -> None:
    """Mark a job complete by writing its run log.

    The marker's presence is the durable completion signal: get_job_status treats
    a job dir containing it as `complete`. Write it last, after the artifacts are
    persisted, so completion implies the outputs exist. Commits the volume.

    Args:
        job_id: Unique name identifying the job.
        log: The run log to persist, e.g. the tool's stdout.
    """
    job_dir = Path(VOLUME_JOBS_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / JOB_COMPLETE_MARKER).write_text(log)
    volume.commit()


def mark_job_failed(job_id: str, log: str) -> None:
    """Mark a job failed by writing its error log to the job dir on the persistent volume.

    Lets a failed job report `failed` with its reason even after Modal's
    function-call result has expired, which.

    Args:
        job_id: Unique name identifying the job.
        log: Failure detail to persist, e.g. captured stderr or the exception.
    """
    job_dir = Path(VOLUME_JOBS_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / JOB_ERROR_MARKER).write_text(log)
    volume.commit()


def persist_job_output(job_id: str, source_dir: Path) -> None:
    """Copy a job output directory to the persistent volume.

    Copies the service's full set of raw artifacts.

    Args:
        job_id: Unique name identifying the job.
        source_dir: Directory whose contents become the job's downloadable artifacts.
    """
    job_dir = Path(VOLUME_JOBS_DIR) / job_id
    shutil.copytree(source_dir, job_dir, dirs_exist_ok=True)


def zip_job_data(job_id: str) -> bytes:
    """Zip a job's entire output directory into an in-memory archive.

    Every file under the job dir is included, stored with paths relative to the
    job dir. Contents are written from bytes rather than from disk, so files
    with pre-1980 mtimes — e.g. mock fixtures staged from the image — don't trip
    zipfile's 1980 timestamp floor.

    Args:
        job_id: Unique name identifying the job.

    Returns:
        The zip archive as bytes.
    """
    job_dir = Path(VOLUME_JOBS_DIR) / job_id
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(job_dir.rglob("*")):
            if path.is_file():
                zf.writestr(str(path.relative_to(job_dir)), path.read_bytes())
    return buffer.getvalue()


def check_job_complete(job_id: str) -> bool:
    """Check whether a job's completion marker has been written."""
    return (Path(VOLUME_JOBS_DIR) / job_id / JOB_COMPLETE_MARKER).exists()


def check_job_failed(job_id: str) -> bool:
    """Check whether a job's error marker has been written."""
    return (Path(VOLUME_JOBS_DIR) / job_id / JOB_ERROR_MARKER).exists()


def delete_job_data(job_id: str) -> None:
    """Remove a job's stored data from the volume."""
    job_dir = Path(VOLUME_JOBS_DIR) / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
        volume.commit()


def check_mock_exists(service: str) -> bool:
    """Check whether a mock fixture exists for a service."""
    return (Path(VOLUME_MOCKS_DIR) / service / MOCK_OUTPUT_DIR).exists()


def get_job_status(job_id: str, job_record: JobRecord) -> JobState:
    """Resolve a job's status from durable markers, then Modal's live call state.

    The volume markers (job.log / error.log) are the durable source of truth for a
    finished job and outlive Modal's call-retention window. Completion comes only
    from the job.log marker, never the call graph (Modal can report a call SUCCESS
    before run() writes it). The call graph only enriches the unfinished case with
    Modal's live state (pending / init_failed / stopped / timed_out). SUCCESS falls
    through to "pending" until the marker lands. Once the window expires with no
    marker written, the outcome is no longer knowable ("unknown").
    """
    call_id = job_record["call_id"]
    if check_job_complete(job_id):
        status = JobState.COMPLETE
    elif check_job_failed(job_id):
        status = JobState.FAILED
    elif call_id is None:
        # A mock job, which never had a Modal call, and whose marker is now gone.
        status = JobState.UNKNOWN
    else:
        try:
            graph = modal.functions.FunctionCall.from_id(call_id).get_call_graph()
        except Exception:
            status = JobState.UNKNOWN
        else:
            if not graph:
                status = JobState.PENDING
            else:
                match graph[0].status:
                    case InputStatus.FAILURE:
                        status = JobState.FAILED
                    case InputStatus.INIT_FAILURE:
                        status = JobState.INIT_FAILED
                    case InputStatus.TERMINATED:
                        status = JobState.STOPPED
                    case InputStatus.TIMEOUT:
                        status = JobState.TIMED_OUT
                    case _:
                        status = JobState.PENDING
    return status


def get_batch_status(job_state_counts: Mapping[JobState, int]) -> BatchState:
    """Aggregate a batch's per-state job counts into a single batch status.

    Args:
        job_state_counts: How many jobs hold each JobState, e.g. {JobState.COMPLETE:
            4, JobState.PENDING: 1}. A state with no jobs is absent rather than zero.
            Keys must be JobState members only: any other key counts as an unfinished
            state and forces COMPLETED_WITH_FAILURES.

    Returns:
        IN_PROGRESS while any job is still pending, COMPLETED_WITH_FAILURES if every
        job finished but some ended in a state other than complete, otherwise
        COMPLETED.
    """
    if job_state_counts.get(JobState.PENDING):
        return BatchState.IN_PROGRESS
    if any(state != JobState.COMPLETE and n for state, n in job_state_counts.items()):
        return BatchState.COMPLETED_WITH_FAILURES
    return BatchState.COMPLETED


class LigandMpnnDesignParams(Protocol):
    """The shared design parameters for LigandMPNN, ProteinMPNN, and SolubleMPNN models."""

    batch_size: int
    fixed_residues: str | None
    redesigned_residues: str | None
    chains_to_design: str | None
    parse_these_chains_only: str | None
    bias_aa: str | None
    omit_aa: str | None
    bias_aa_per_residue: dict | None
    omit_aa_per_residue: dict | None
    symmetry_residues: str | None
    symmetry_weights: str | None
    homo_oligomer: bool
    save_stats: bool
    parse_atoms_with_zero_occupancy: bool
    zero_indexed: bool
    pack_side_chains: bool
    number_of_packs_per_design: int
    sc_num_denoising_steps: int
    sc_num_samples: int
    repack_everything: bool


def build_shared_args(params: LigandMpnnDesignParams, tmpdir: Path, checkpoint_path_sc: str) -> list[str]:
    """Translate the shared LigandMPNN, ProteinMPNN, and SolubleMPNN, params into run.py CLI flags.

    Shared by ligandmpnn, proteinmpnn, and solublempnn, which invoke the same run.py.
    Per-residue bias and omit maps are dicts in the API, written here to temp JSON
    files since run.py reads them from disk. batch_size is always passed. run.py takes
    booleans as 0/1 ints. The packing flags are emitted only when pack_side_chains is
    set, and checkpoint_path_sc is the staged side-chain packer the caller supplies.
    """
    args: list[str] = ["--batch_size", str(params.batch_size)]

    if params.fixed_residues:
        args += ["--fixed_residues", params.fixed_residues]
    if params.redesigned_residues:
        args += ["--redesigned_residues", params.redesigned_residues]
    if params.chains_to_design:
        args += ["--chains_to_design", params.chains_to_design]
    if params.parse_these_chains_only:
        args += ["--parse_these_chains_only", params.parse_these_chains_only]

    if params.bias_aa:
        args += ["--bias_AA", params.bias_aa]
    if params.omit_aa:
        args += ["--omit_AA", params.omit_aa]
    if params.bias_aa_per_residue:
        path = tmpdir / "bias_aa_per_residue.json"
        path.write_text(json.dumps(params.bias_aa_per_residue))
        args += ["--bias_AA_per_residue", str(path)]
    if params.omit_aa_per_residue:
        path = tmpdir / "omit_aa_per_residue.json"
        path.write_text(json.dumps(params.omit_aa_per_residue))
        args += ["--omit_AA_per_residue", str(path)]

    if params.symmetry_residues:
        args += ["--symmetry_residues", params.symmetry_residues]
    if params.symmetry_weights:
        args += ["--symmetry_weights", params.symmetry_weights]
    if params.homo_oligomer:
        args += ["--homo_oligomer", "1"]

    if params.save_stats:
        args += ["--save_stats", "1"]
    if params.parse_atoms_with_zero_occupancy:
        args += ["--parse_atoms_with_zero_occupancy", "1"]
    if params.zero_indexed:
        args += ["--zero_indexed", "1"]

    if params.pack_side_chains:
        args += [
            "--pack_side_chains",
            "1",
            "--checkpoint_path_sc",
            checkpoint_path_sc,
            "--number_of_packs_per_design",
            str(params.number_of_packs_per_design),
            "--sc_num_denoising_steps",
            str(params.sc_num_denoising_steps),
            "--sc_num_samples",
            str(params.sc_num_samples),
        ]
        if params.repack_everything:
            args += ["--repack_everything", "1"]

    return args
