import json
import logging
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import modal
from pydantic import BaseModel, Field, model_validator

from common.utils import format_run_log, mark_job_complete, mark_job_failed, persist_job_output
from config import BINDCRAFT_GPU, BINDCRAFT_MAX_CONTAINERS, BINDCRAFT_SCALEDOWN_WINDOW, BINDCRAFT_TIMEOUT
from constants import (
    BINDCRAFT_COMMIT,
    BINDCRAFT_DIR,
    BINDCRAFT_REPO,
    BINDCRAFT_SETTINGS_ADVANCED_DIR,
    BINDCRAFT_SETTINGS_FILTERS_DIR,
    BINDCRAFT_SPEC,
    COLABDESIGN_SPEC,
    PYDANTIC_SPEC,
    PYROSETTA_FIND_LINKS,
    PYROSETTA_SPEC,
    PYTHON_3_11,
    SERVICE_SOURCES,
    VOLUME_BINDCRAFT_CACHE,
    VOLUME_ROOT,
)
from core import app, volume

logger = logging.getLogger(__name__)


bindcraft_image = (
    modal.Image.debian_slim(python_version=PYTHON_3_11)
    .apt_install("git", "ffmpeg", "libgfortran5", "libgmp10")
    .pip_install(
        "pandas<3.0.0",
        "matplotlib<3.9.0",
        "numpy<2.0.0",
        "biopython",
        "scipy",
        "seaborn",
        "tqdm",
        "fsspec",
        "py3dmol",
        "chex",
        "dm-haiku",
        "flax<0.10.0",
        "dm-tree",
        "joblib",
        "ml-collections",
        "immutabledict",
        "optax",
        "jax[cuda12]>=0.4,<=0.6.0",
        PYDANTIC_SPEC,
    )
    # ColabDesign pins fight the resolved environment, so it installs without its deps.
    .pip_install(COLABDESIGN_SPEC, extra_options="--no-deps")
    # BindCraft calls InterfaceAnalyzerMover.set_interface with a string, which later
    # PyRosetta releases replaced with a DockingPartners argument.
    .pip_install(PYROSETTA_SPEC, find_links=PYROSETTA_FIND_LINKS)
    .run_commands(
        f"git clone {BINDCRAFT_REPO} {BINDCRAFT_DIR}",
        f"cd {BINDCRAFT_DIR} && git checkout {BINDCRAFT_COMMIT}",
        f"chmod +x {BINDCRAFT_DIR}/functions/dssp {BINDCRAFT_DIR}/functions/DAlphaBall.gcc",
    )
    .add_local_python_source(*SERVICE_SOURCES)
)


AdvancedProtocol = Literal[
    "default_4stage_multimer",
    "default_4stage_multimer_flexible",
    "default_4stage_multimer_flexible_hardtarget",
    "default_4stage_multimer_hardtarget",
    "default_4stage_multimer_mpnn",
    "default_4stage_multimer_mpnn_flexible",
    "default_4stage_multimer_mpnn_flexible_hardtarget",
    "default_4stage_multimer_mpnn_hardtarget",
    "betasheet_4stage_multimer",
    "betasheet_4stage_multimer_flexible",
    "betasheet_4stage_multimer_flexible_hardtarget",
    "betasheet_4stage_multimer_hardtarget",
    "betasheet_4stage_multimer_mpnn",
    "betasheet_4stage_multimer_mpnn_flexible",
    "betasheet_4stage_multimer_mpnn_flexible_hardtarget",
    "betasheet_4stage_multimer_mpnn_hardtarget",
    "peptide_3stage_multimer",
    "peptide_3stage_multimer_flexible",
    "peptide_3stage_multimer_mpnn",
    "peptide_3stage_multimer_mpnn_flexible",
]

FilterPreset = Literal[
    "default_filters",
    "relaxed_filters",
    "peptide_filters",
    "peptide_relaxed_filters",
    "no_filters",
]


class BindCraftParams(BaseModel):
    """Parameters for de novo binder design against a target with BindCraft.

    BindCraft hallucinates binders with AlphaFold2 backpropagation, redesigns them
    with MPNN, and scores the result with PyRosetta. It runs a loop rather than a
    single pass, stopping when it has `number_of_final_designs` designs that pass
    the filters or when it has tried `max_trajectories` trajectories. Accepted
    designs are rare, so the defaults here are sized for a bounded API call rather
    than for a production campaign, where hundreds of trajectories are typical.
    """

    pdb: str = Field(description="Inline PDB content of the target to design a binder against.")
    chains: str = Field(default="A", description="Target chains to design against; others are ignored.", examples=["A"])
    target_hotspot_residues: str | None = Field(
        default=None,
        description=(
            "Residues to target, e.g. `1,2-10`, chain-specific `A1-10,B1-20`, or whole chains `A`. "
            "Null lets AlphaFold2 pick the binding site, which widens the search."
        ),
        examples=["56"],
    )
    binder_name: str = Field(default="binder", description="Filename prefix for the designed binders.")
    min_length: int = Field(default=65, ge=1, description="Shortest binder length to sample.")
    max_length: int = Field(default=150, ge=1, description="Longest binder length to sample.")
    number_of_final_designs: int = Field(default=1, ge=1, description="Stop once this many designs pass all filters.")
    max_trajectories: int = Field(
        default=5, ge=1, description="Stop after this many trajectories even if no design has passed."
    )
    advanced: AdvancedProtocol = Field(
        default="default_4stage_multimer", description="Advanced settings preset controlling the design protocol."
    )
    filters: FilterPreset = Field(default="default_filters", description="Filter preset applied to designs.")

    @model_validator(mode="after")
    def check_lengths(self) -> "BindCraftParams":
        if self.max_length < self.min_length:
            raise ValueError("max_length must be greater than or equal to min_length")
        return self


def build_settings(params: BindCraftParams, tmpdir: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    """Write the three settings files BindCraft reads, and return their paths.

    BindCraft takes its target, filters, and advanced settings as JSON files on
    disk. The target file is written from the request. The filter preset is passed
    through from the clone untouched. The advanced preset is copied and patched:
    `af_params_dir` points at the staged AlphaFold2 weights on the volume, and
    `max_trajectories` is set so a run is bounded even when nothing passes filters.
    `dssp_path` and `dalphaball_path` are left empty, which makes BindCraft resolve
    them to the executables in its own clone.
    """
    target_settings = {
        "design_path": f"{output_dir}/",
        "binder_name": params.binder_name,
        "starting_pdb": str(tmpdir / "target.pdb"),
        "chains": params.chains,
        "target_hotspot_residues": params.target_hotspot_residues,
        "lengths": [params.min_length, params.max_length],
        "number_of_final_designs": params.number_of_final_designs,
    }
    settings_path = tmpdir / "target.json"
    settings_path.write_text(json.dumps(target_settings))

    advanced_settings = json.loads(Path(f"{BINDCRAFT_SETTINGS_ADVANCED_DIR}/{params.advanced}.json").read_text())
    advanced_settings["af_params_dir"] = VOLUME_BINDCRAFT_CACHE
    advanced_settings["max_trajectories"] = params.max_trajectories
    advanced_path = tmpdir / "advanced.json"
    advanced_path.write_text(json.dumps(advanced_settings))

    filters_path = Path(f"{BINDCRAFT_SETTINGS_FILTERS_DIR}/{params.filters}.json")
    return settings_path, filters_path, advanced_path


@app.function(
    name="bindcraft",
    image=bindcraft_image,
    gpu=BINDCRAFT_GPU,
    volumes={VOLUME_ROOT: volume},
    timeout=BINDCRAFT_TIMEOUT,
    max_containers=BINDCRAFT_MAX_CONTAINERS,
    scaledown_window=BINDCRAFT_SCALEDOWN_WINDOW,
)
def run(job_id: str, job_name: str | None, params: dict) -> None:
    """Design binders with BindCraft and persist the output.

    The output directory holds the accepted designs under Accepted/, every
    trajectory under Trajectory/, the MPNN redesigns under MPNN/, and the
    per-stage statistics as CSVs at the top level. A run that finishes without
    an accepted design is still a successful job: the trajectories and the
    failure CSV explain what was rejected and why.

    Output is streamed rather than captured, because a design loop runs for hours
    and its per-trajectory progress is the only sign it is alive. Capturing would
    withhold every line until exit, and a cancelled or timed out container never
    reaches the exception handler, so such a run would leave no diagnostic at all.
    stderr is merged into that stream so tracebacks stay in order with the progress
    lines before them, which leaves the run log with no separate stderr section.

    Args:
        job_id: Unique id identifying the job.
        job_name: The caller's label for the job, recorded in the run log. None
            when the caller did not supply one.
        params: A BindCraftParams dump, revalidated here so the container never
            trusts the payload it was handed.
    """
    logger.info(f"Starting bindcraft design: job={job_id}")
    started_at = datetime.now(UTC)
    job_command = ""

    try:
        job_params = BindCraftParams.model_validate(params)
        volume.reload()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "target.pdb").write_text(job_params.pdb)
            output_dir = tmpdir / "output"
            settings_path, filters_path, advanced_path = build_settings(job_params, tmpdir, output_dir)

            cmd = [
                "python",
                "-u",
                f"{BINDCRAFT_DIR}/bindcraft.py",
                "--settings",
                str(settings_path),
                "--filters",
                str(filters_path),
                "--advanced",
                str(advanced_path),
            ]
            job_command = " ".join(cmd)
            logger.info(f"Running: {job_command}")

            output_lines: list[str] = []
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=BINDCRAFT_DIR,
            )
            assert process.stdout is not None
            for line in process.stdout:
                line = line.rstrip()
                logger.info(f"bindcraft: {line}")
                output_lines.append(line)
            returncode = process.wait()
            output = "\n".join(output_lines)

            if returncode != 0:
                raise RuntimeError(f"bindcraft run exited with code {returncode}\n{output}")

            persist_job_output(job_id, output_dir)

        log = format_run_log(job_id, job_name, "bindcraft", BINDCRAFT_SPEC, job_command, output, "", started_at)
        mark_job_complete(job_id, log)
        logger.info(f"Done: job={job_id}")
    except Exception as e:
        logger.error(f"Failed: job={job_id}: {e}")
        log = format_run_log(job_id, job_name, "bindcraft", BINDCRAFT_SPEC, job_command, str(e), "", started_at)
        mark_job_failed(job_id, log)
        raise
