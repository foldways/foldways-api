import logging
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import modal
from pydantic import BaseModel, Field

from common.utils import format_run_log, mark_job_complete, mark_job_failed, persist_job_output
from config import THERMOMPNN_GPU, THERMOMPNN_MAX_CONTAINERS, THERMOMPNN_SCALEDOWN_WINDOW, THERMOMPNN_TIMEOUT
from constants import (
    PYDANTIC_SPEC,
    PYTHON_3_12,
    SERVICE_SOURCES,
    THERMOMPNN_COMMIT,
    THERMOMPNN_DIR,
    THERMOMPNN_MODEL_PATH,
    THERMOMPNN_REPO,
    THERMOMPNN_SPEC,
    VOLUME_ROOT,
)
from core import app, volume

logger = logging.getLogger(__name__)


# ThermoMPNN is a research repo with its checkpoints committed in git, so the clone carries
# the weights and no volume staging is needed. Its local.yaml hardcodes the author's NAS
# paths, and the inference path reads platform.thermompnn_dir to locate the vanilla
# ProteinMPNN backbone, so it is rewritten to the clone. Python is 3.12 rather than the 3.10
# the repo pins, because the shared modules use StrEnum and datetime.UTC.
thermompnn_image = (
    modal.Image.debian_slim(python_version=PYTHON_3_12)
    .apt_install("git")
    .pip_install(
        "torch",
        "pytorch-lightning<3.0.0",
        "torchmetrics<2.0.0",
        "biopython",
        "omegaconf",
        "pandas",
        "numpy",
        "tqdm",
        "joblib",
        "wandb",
        PYDANTIC_SPEC,
    )
    .env({"WANDB_MODE": "disabled"})
    .run_commands(
        f"git clone {THERMOMPNN_REPO} {THERMOMPNN_DIR}",
        f"cd {THERMOMPNN_DIR} && git checkout {THERMOMPNN_COMMIT}",
        f'printf \'platform:\\n  accel: "gpu"\\n  thermompnn_dir: "{THERMOMPNN_DIR}"\\n\' > {THERMOMPNN_DIR}/local.yaml',
    )
    .add_local_python_source(*SERVICE_SOURCES)
)


class ThermoMPNNParams(BaseModel):
    """Parameters for predicting point-mutation stability changes with ThermoMPNN.

    ThermoMPNN runs site-saturation mutagenesis over a chain: for every position it
    predicts the ddG of every single amino-acid substitution. A positive ddG is
    destabilizing and a negative one is stabilizing. The output is one row per
    substitution, so a chain of length L yields about L by 19 predictions.
    """

    pdb: str = Field(description="Inline PDB content of the structure to run mutagenesis on.")
    chain: str = Field(
        default="A", description="Chain in the PDB to run site-saturation mutagenesis on.", examples=["A"]
    )


@app.function(
    name="thermompnn",
    image=thermompnn_image,
    gpu=THERMOMPNN_GPU,
    volumes={VOLUME_ROOT: volume},
    timeout=THERMOMPNN_TIMEOUT,
    max_containers=THERMOMPNN_MAX_CONTAINERS,
    scaledown_window=THERMOMPNN_SCALEDOWN_WINDOW,
)
def run(job_id: str, job_name: str | None, params: dict) -> None:
    """Predict point-mutation stability changes with ThermoMPNN and persist the output.

    The output directory holds ThermoMPNN_inference_input.csv, one row per single
    substitution with its predicted ddG. custom_inference.py is invoked from the
    repo's analysis directory, which is how it resolves its sibling imports.

    Args:
        job_id: Unique id identifying the job.
        job_name: The caller's label for the job, recorded in the run log. None
            when the caller did not supply one.
        params: A ThermoMPNNParams dump, revalidated here so the container never
            trusts the payload it was handed.
    """
    logger.info(f"Starting thermompnn job={job_id}")
    started_at = datetime.now(UTC)
    job_command = ""
    stderr = ""

    try:
        job_params = ThermoMPNNParams.model_validate(params)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdb_path = tmpdir / "input.pdb"
            pdb_path.write_text(job_params.pdb)
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            cmd = [
                "python",
                "custom_inference.py",
                "--pdb",
                str(pdb_path),
                "--chain",
                job_params.chain,
                "--model_path",
                THERMOMPNN_MODEL_PATH,
                "--out_dir",
                str(output_dir),
            ]
            job_command = " ".join(cmd)
            logger.info(f"Running: {job_command}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=f"{THERMOMPNN_DIR}/analysis")
            stderr = result.stderr

            if result.returncode != 0:
                logger.error(f"thermompnn stderr: {stderr}")
                raise RuntimeError(f"thermompnn run exited with code {result.returncode}")

            persist_job_output(job_id, output_dir)

        log = format_run_log(
            job_id, job_name, "thermompnn", THERMOMPNN_SPEC, job_command, result.stdout, stderr, started_at
        )
        mark_job_complete(job_id, log)
        logger.info(f"Done: job={job_id}")
    except Exception as e:
        logger.error(f"Failed: job={job_id}: {e}")
        log = format_run_log(job_id, job_name, "thermompnn", THERMOMPNN_SPEC, job_command, str(e), stderr, started_at)
        mark_job_failed(job_id, log)
        raise
