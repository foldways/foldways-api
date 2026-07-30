import logging
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import modal
from pydantic import BaseModel, Field

from common.utils import format_run_log, mark_job_complete, mark_job_failed, persist_job_output
from config import BOLTZGEN_GPU, BOLTZGEN_MAX_CONTAINERS, BOLTZGEN_SCALEDOWN_WINDOW, BOLTZGEN_TIMEOUT
from constants import (
    BOLTZGEN_SPEC,
    PYDANTIC_SPEC,
    PYTHON_3_12,
    SERVICE_SOURCES,
    VOLUME_BOLTZGEN_CACHE,
    VOLUME_ROOT,
)
from core import app, volume

logger = logging.getLogger(__name__)


boltzgen_image = (
    modal.Image.debian_slim(python_version=PYTHON_3_12)
    .uv_pip_install(BOLTZGEN_SPEC, "pyyaml", PYDANTIC_SPEC)
    .add_local_python_source(*SERVICE_SOURCES)
)


class DesignChain(BaseModel):
    id: str = Field(description="Chain id of the designed protein.", examples=["A"])
    sequence: str = Field(
        description="Designed length as a range like `80..140`, a fixed count, or an amino-acid sequence to redesign.",
        examples=["80..140"],
    )


class Target(BaseModel):
    """A target structure the design is generated against, given as inline mmCIF."""

    cif: str = Field(description="Inline mmCIF content of the target (the file's text, not a path).")
    chain_ids: list[str] = Field(description="Chain ids in the mmCIF to include as the target.", examples=[["A"]])


class BoltzGenParams(BaseModel):
    """Parameters for de novo protein design with BoltzGen.

    Generates `num_designs` candidates, then filters to a diverse `budget`. A
    target may be supplied to design against a binding partner. In practice
    `num_designs` is large (thousands), so the default here is modest.
    """

    design: DesignChain = Field(description="The protein chain to design.")
    target: Target | None = Field(default=None, description="Optional target structure to design against.")
    protocol: Literal[
        "protein-anything",
        "peptide-anything",
        "nanobody-anything",
        "antibody-anything",
        "protein-redesign",
    ] = Field(default="protein-anything", description="Design protocol.")
    num_designs: int = Field(default=10, ge=1, description="Total designs to generate before filtering.")
    budget: int = Field(default=2, ge=1, description="Size of the final diversity-optimized set.")


def build_design_spec(params: BoltzGenParams, tmpdir: Path) -> dict:
    """Render the request into a BoltzGen design-spec dict.

    A target is written to a temp mmCIF file and referenced by path, since BoltzGen
    reads targets from disk.
    """
    entities: list[dict] = [{"protein": {"id": params.design.id, "sequence": params.design.sequence}}]
    if params.target is not None:
        target_path = tmpdir / "target.cif"
        target_path.write_text(params.target.cif)
        include = [{"chain": {"id": chain_id}} for chain_id in params.target.chain_ids]
        entities.append({"file": {"path": str(target_path), "include": include}})
    return {"entities": entities}


@app.function(
    name="boltzgen",
    image=boltzgen_image,
    gpu=BOLTZGEN_GPU,
    volumes={VOLUME_ROOT: volume},
    timeout=BOLTZGEN_TIMEOUT,
    max_containers=BOLTZGEN_MAX_CONTAINERS,
    scaledown_window=BOLTZGEN_SCALEDOWN_WINDOW,
)
def run(job_id: str, job_name: str | None, params: dict) -> None:
    """Design proteins with BoltzGen and persist the ranked output.

    Args:
        job_id: Unique id identifying the job.
        job_name: The caller's label for the job, recorded in the run log. None
            when the caller did not supply one.
        params: A BoltzGenParams dump, revalidated here so the container never
            trusts the payload it was handed.
    """
    import yaml

    logger.info(f"Starting boltzgen design: job={job_id}")
    started_at = datetime.now(UTC)
    job_command = ""
    stderr = ""

    try:
        job_params = BoltzGenParams.model_validate(params)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            spec_path = tmpdir / "spec.yaml"
            spec_path.write_text(yaml.dump(build_design_spec(job_params, tmpdir), sort_keys=False))
            output_dir = tmpdir / "output"

            cmd = [
                "boltzgen",
                "run",
                str(spec_path),
                "--output",
                str(output_dir),
                "--protocol",
                job_params.protocol,
                "--num_designs",
                str(job_params.num_designs),
                "--budget",
                str(job_params.budget),
                "--cache",
                VOLUME_BOLTZGEN_CACHE,
                "--devices",
                "1",
            ]
            job_command = " ".join(cmd)
            logger.info(f"Running: {job_command}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            stderr = result.stderr

            if result.returncode != 0:
                logger.error(f"boltzgen stderr: {stderr}")
                raise RuntimeError(f"boltzgen run exited with code {result.returncode}")

            persist_job_output(job_id, output_dir)

        log = format_run_log(
            job_id, job_name, "boltzgen", BOLTZGEN_SPEC, job_command, result.stdout, stderr, started_at
        )
        mark_job_complete(job_id, log)
        logger.info(f"Done: job={job_id}")
    except Exception as e:
        logger.error(f"Failed: job={job_id}: {e}")
        log = format_run_log(job_id, job_name, "boltzgen", BOLTZGEN_SPEC, job_command, str(e), stderr, started_at)
        mark_job_failed(job_id, log)
        raise
