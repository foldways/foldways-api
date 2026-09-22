import json
import logging
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import modal
from pydantic import BaseModel, Field, model_validator

from common.utils import format_run_log, mark_job_complete, mark_job_failed, persist_job_output
from config import BINDCRAFT2_GPU, BINDCRAFT2_MAX_CONTAINERS, BINDCRAFT2_SCALEDOWN_WINDOW, BINDCRAFT2_TIMEOUT
from constants import (
    BINDCRAFT2_ACCELERATOR_EXTRA,
    BINDCRAFT2_AF2_PARAMS_ENV,
    BINDCRAFT2_COMMIT,
    BINDCRAFT2_COMPILE_CACHE_ENV,
    BINDCRAFT2_DIR,
    BINDCRAFT2_REPO,
    BINDCRAFT2_SPEC,
    PYDANTIC_SPEC,
    PYTHON_3_12,
    SERVICE_SOURCES,
    VOLUME_BINDCRAFT2_CACHE,
    VOLUME_ROOT,
)
from core import app, volume

logger = logging.getLogger(__name__)


bindcraft2_image = (
    modal.Image.debian_slim(python_version=PYTHON_3_12)
    .apt_install("git")
    .run_commands(
        f"git clone {BINDCRAFT2_REPO} {BINDCRAFT2_DIR}",
        f"cd {BINDCRAFT2_DIR} && git checkout {BINDCRAFT2_COMMIT}",
        "python -m pip install --upgrade pip",
        f"cd {BINDCRAFT2_DIR} && python -m pip install -e '.[{BINDCRAFT2_ACCELERATOR_EXTRA}]'",
    )
    .pip_install(PYDANTIC_SPEC)
    .add_local_python_source(*SERVICE_SOURCES)
)


Modality = Literal[
    "binder",
    "large_binder",
    "peptide",
    "cyclic_peptide",
    "homo_oligomer",
    "multidomain",
    "VHH",
    "ARP",
    "scFv",
    "Fab",
]

SCAFFOLDED_MODALITIES = frozenset({"VHH", "ARP", "scFv", "Fab"})

Conformation = Literal["induced_fit", "fold_switch"]

STRUCTURE_SUFFIXES = {"pdb": ".pdb", "cif": ".cif", "mmcif": ".mmcif", "fasta": ".fasta"}
StructureFormat = Literal["pdb", "cif", "mmcif", "fasta"]

ShippedTarget = Literal["hPDL1", "mPDL1", "hPD1", "hIL2R", "hIL7RA", "dynorphin_a"]

PROPERTY_FLAGS = (
    "humanize",
    "protease_stable",
    "disulfide_staple",
    "mixed_topology",
    "termini_together",
    "termini_accessible",
    "forced_targeting",
    "initial_guess",
    "bigbang",
)


class TargetInput(BaseModel):
    """One target to design against, supplied as an inline structure or sequence."""

    name: str = Field(description="Label for this target, used in the target chain names and output.")
    structure: str = Field(description="Inline structure or sequence content, in the format named by `format`.")
    format: StructureFormat = Field(
        default="pdb",
        description="Content format. Use `fasta` for a sequence target, such as a peptide or disordered region.",
    )
    chains: str = Field(
        default="A", description="Target chains to design against, e.g. `A` or `A,B`. Ignored for a fasta target."
    )
    hotspots: str | None = Field(
        default=None,
        description=(
            "Residues to bind, using this target's own residue numbers, e.g. `54,56,66-70` or "
            "chain-prefixed `A54,B12-16`. Null lets AlphaFold2 pick the binding site."
        ),
    )
    coldspots: str | None = Field(
        default=None, description="Residues to keep free of the binder, in the same numbering as hotspots."
    )
    objective: Literal["target", "detarget"] = Field(
        default="target", description="`target` to bind this target, or `detarget` to design against binding it."
    )


class BindCraft2Params(BaseModel):
    """Parameters for de novo binder design against a target with BindCraft2 (BC2).

    BC2 optimises a binder with AlphaFold2 and ProteinMPNN, then evaluates it with
    separate AlphaFold models and structural filters. It runs a loop rather than a
    single pass, stopping when it has `number_of_final_designs` accepted designs or
    when it has tried `max_trajectories` trajectories. Accepted designs are rare, so
    the defaults here are sized for a bounded API call rather than a production
    campaign, where hundreds of trajectories are typical.

    Supply the target either as one or more shipped targets by name, through `target`,
    or as inline structures through `targets`, but not both. Several targets bind at
    once, and a target with `objective: detarget` is one the binder should avoid.
    """

    target: ShippedTarget | list[ShippedTarget] | None = Field(
        default=None, description="Shipped target name, or a list of them to bind at once."
    )
    targets: list[TargetInput] | None = Field(
        default=None, description="Inline targets to design against, as an alternative to a shipped `target`."
    )
    modality: Modality = Field(default="binder", description="Binder format to design.")
    conformation: Conformation | None = Field(
        default=None,
        description="Optional conformational objective layered on the format. Each one requires a single target.",
    )
    binder_name: str = Field(default="binder", description="Filename prefix for the designed binders.")
    min_length: int = Field(
        default=80, ge=1, description="Shortest binder length to sample. Ignored for scaffolded formats."
    )
    max_length: int = Field(
        default=80, ge=1, description="Longest binder length to sample. Ignored for scaffolded formats."
    )
    copies: int | None = Field(
        default=None,
        ge=2,
        description="Number of identical chains for the homo_oligomer format. Null uses the BC2 default.",
    )
    number_of_final_designs: int = Field(default=1, ge=1, description="Stop once this many designs pass all filters.")
    max_trajectories: int = Field(
        default=5, ge=1, description="Stop after this many trajectories even if no design has passed."
    )
    humanize: bool = Field(default=False, description="Favour human-like sequence features.")
    protease_stable: bool = Field(default=False, description="Reduce predicted protease cleavage susceptibility.")
    disulfide_staple: bool = Field(default=False, description="Include a predicted disulfide bond.")
    mixed_topology: bool = Field(default=False, description="Select for beta-sheet content and limit helicity.")
    termini_together: bool = Field(default=False, description="Bring the N and C termini close together.")
    termini_accessible: bool = Field(default=False, description="Direct both chain ends away from the target.")
    forced_targeting: bool = Field(
        default=False,
        description="Concentrate binding on the named hotspots. Requires hotspots on a structured target.",
    )
    initial_guess: bool = Field(
        default=False, description="Re-predict each candidate from the pose the trajectory folded."
    )
    bigbang: bool = Field(
        default=False, description="Seed the gradient stages so a binder starts folded from the origin."
    )

    def target_count(self) -> int:
        """Number of targets the request describes, whether shipped or inline."""
        if self.target is not None:
            return 1 if isinstance(self.target, str) else len(self.target)
        return len(self.targets or [])

    @model_validator(mode="after")
    def check_params(self) -> "BindCraft2Params":
        if (self.target is None) == (self.targets is None):
            raise ValueError("provide exactly one of target or targets")
        if self.targets is not None and not self.targets:
            raise ValueError("targets must not be empty")
        if self.max_length < self.min_length:
            raise ValueError("max_length must be greater than or equal to min_length")
        if self.conformation is not None and self.target_count() != 1:
            raise ValueError("conformation requires exactly one target")
        if self.forced_targeting and self.targets is not None and not any(t.hotspots for t in self.targets):
            raise ValueError("forced_targeting requires hotspots on at least one target")
        return self


def build_target_entries(targets: list[TargetInput], tmpdir: Path) -> list[dict]:
    """Write each inline target to a file and return the BC2 target entries.

    BC2 detects a target's format from its file suffix, so the structure is written
    under the suffix that matches its declared format. A fasta target is a sequence,
    so its chains are left off.
    """
    entries = []
    for index, target in enumerate(targets):
        target_path = tmpdir / f"target_{index}{STRUCTURE_SUFFIXES[target.format]}"
        target_path.write_text(target.structure)
        entry: dict = {"name": target.name, "target_path": str(target_path)}
        if target.format != "fasta":
            entry["chains"] = target.chains
        if target.hotspots:
            entry["hotspots"] = target.hotspots
        if target.coldspots:
            entry["coldspots"] = target.coldspots
        if target.objective == "detarget":
            entry["objective"] = "detarget"
        entries.append(entry)
    return entries


def build_settings(params: BindCraft2Params, tmpdir: Path, output_dir: Path) -> Path:
    """Write the BC2 campaign settings file and return its path.

    BC2 takes one JSON campaign file that layers over its shipped presets. The target
    is either the shipped presets named in `target` or inline entries written from
    `targets`. `max_trajectories` is set so a run is bounded even when nothing passes
    the filters, a conformational objective rides on the format as a modality list,
    and `binder_lengths` is left out for scaffolded formats, whose length comes from
    the scaffold.
    """
    settings: dict = {
        "modality": [params.modality, params.conformation] if params.conformation else params.modality,
        "binder_name": params.binder_name,
        "number_of_final_designs": params.number_of_final_designs,
        "max_trajectories": params.max_trajectories,
        "project_folder": str(output_dir),
    }
    if params.target is not None:
        settings["target"] = params.target
    else:
        assert params.targets is not None
        settings["targets"] = build_target_entries(params.targets, tmpdir)
    if params.modality not in SCAFFOLDED_MODALITIES:
        settings["binder_lengths"] = [params.min_length, params.max_length]
    if params.copies is not None:
        settings["copies"] = params.copies
    for flag in PROPERTY_FLAGS:
        if getattr(params, flag):
            settings[flag] = True

    settings_path = tmpdir / "campaign.json"
    settings_path.write_text(json.dumps(settings))
    return settings_path


@app.function(
    name="bindcraft2",
    image=bindcraft2_image,
    gpu=BINDCRAFT2_GPU,
    volumes={VOLUME_ROOT: volume},
    timeout=BINDCRAFT2_TIMEOUT,
    max_containers=BINDCRAFT2_MAX_CONTAINERS,
    scaledown_window=BINDCRAFT2_SCALEDOWN_WINDOW,
)
def run(job_id: str, job_name: str | None, params: dict) -> None:
    """Design binders with BindCraft2 and persist the output.

    The output directory holds the design attempts under 1_Trajectories/, the
    redesigned sequences and their predicted complexes under 2_Refolded/, and the
    accepted, ranked designs under 3_Ranked/, with per-stage CSVs beside them. A run
    that finishes without an accepted design is still a successful job: the earlier
    stages and their filter outcomes explain what was rejected and why.

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
        params: A BindCraft2Params dump, revalidated here so the container never
            trusts the payload it was handed.
    """
    logger.info(f"Starting bindcraft2 design: job={job_id}")
    started_at = datetime.now(UTC)
    job_command = ""

    try:
        job_params = BindCraft2Params.model_validate(params)
        volume.reload()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_dir = tmpdir / "output"
            settings_path = build_settings(job_params, tmpdir, output_dir)

            cmd = ["python", "-u", "bindcraft.py", "design", str(settings_path)]
            job_command = " ".join(cmd)
            logger.info(f"Running: {job_command}")

            # BC2 writes its JAX compile cache into the project folder unless this points elsewhere.
            # It is a machine-specific cache, not a result, so it is kept out of the persisted output.
            output_lines: list[str] = []
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=BINDCRAFT2_DIR,
                env={
                    **os.environ,
                    BINDCRAFT2_AF2_PARAMS_ENV: VOLUME_BINDCRAFT2_CACHE,
                    BINDCRAFT2_COMPILE_CACHE_ENV: str(tmpdir / "compile_cache"),
                },
            )
            assert process.stdout is not None
            for line in process.stdout:
                line = line.rstrip()
                logger.info(f"bindcraft2: {line}")
                output_lines.append(line)
            returncode = process.wait()
            output = "\n".join(output_lines)

            if returncode != 0:
                raise RuntimeError(f"bindcraft2 run exited with code {returncode}\n{output}")

            persist_job_output(job_id, output_dir)

        log = format_run_log(job_id, job_name, "bindcraft2", BINDCRAFT2_SPEC, job_command, output, "", started_at)
        mark_job_complete(job_id, log)
        logger.info(f"Done: job={job_id}")
    except Exception as e:
        logger.error(f"Failed: job={job_id}: {e}")
        log = format_run_log(job_id, job_name, "bindcraft2", BINDCRAFT2_SPEC, job_command, str(e), "", started_at)
        mark_job_failed(job_id, log)
        raise
