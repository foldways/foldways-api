import logging
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import modal
from pydantic import BaseModel, Field

from common.utils import format_run_log, mark_job_complete, mark_job_failed, persist_job_output
from config import LIGANDMPNN_GPU, LIGANDMPNN_MAX_CONTAINERS, LIGANDMPNN_SCALEDOWN_WINDOW, LIGANDMPNN_TIMEOUT
from constants import (
    LIGANDMPNN_COMMIT,
    LIGANDMPNN_DIR,
    LIGANDMPNN_REPO,
    LIGANDMPNN_SPEC,
    PYDANTIC_SPEC,
    PYTHON_3_11,
    SERVICE_SOURCES,
    VOLUME_LIGANDMPNN_CACHE,
    VOLUME_ROOT,
)
from core import app, volume

logger = logging.getLogger(__name__)


ligandmpnn_image = (
    modal.Image.debian_slim(python_version=PYTHON_3_11)
    .apt_install("git")
    .run_commands(
        f"git clone {LIGANDMPNN_REPO} {LIGANDMPNN_DIR}",
        f"cd {LIGANDMPNN_DIR} && git checkout {LIGANDMPNN_COMMIT}",
        f"pip install -r {LIGANDMPNN_DIR}/requirements.txt",
    )
    .uv_pip_install(PYDANTIC_SPEC)
    .add_local_python_source(*SERVICE_SOURCES)
)


class LigandMPNNParams(BaseModel):
    """Parameters for designing sequences for a backbone with LigandMPNN.

    LigandMPNN is an inverse-folding model that conditions on the non-protein
    atoms in the structure. Given a backbone plus its ligands, nucleotides, or
    metals it designs amino-acid sequences predicted to fold into it while
    respecting that atomic context.
    """

    pdb: str = Field(
        description="Inline PDB content of the backbone to design sequences for, including the ligand, "
        "nucleotide, or metal atoms LigandMPNN should condition on."
    )
    num_sequences: int = Field(default=1, ge=1, le=100, description="Number of sequences to design.")
    temperature: float = Field(
        default=0.1, gt=0, le=1.0, description="Sampling temperature. Higher gives more diversity."
    )
    noise: Literal["005", "010", "020", "030"] = Field(
        default="010",
        description="Backbone noise the checkpoint was trained with, in hundredths of an Angstrom.",
    )
    use_atom_context: bool = Field(
        default=True,
        description="Whether to condition on the ligand atoms. Turn off to see the design without atomic context.",
    )
    use_side_chain_context: bool = Field(
        default=False,
        description="Whether to use side-chain atoms of fixed residues as additional context.",
    )
    seed: int = Field(default=0, description="Random seed for reproducible sampling.")


@app.function(
    name="ligandmpnn",
    image=ligandmpnn_image,
    gpu=LIGANDMPNN_GPU,
    volumes={VOLUME_ROOT: volume},
    timeout=LIGANDMPNN_TIMEOUT,
    max_containers=LIGANDMPNN_MAX_CONTAINERS,
    scaledown_window=LIGANDMPNN_SCALEDOWN_WINDOW,
)
def run(job_id: str, job_name: str | None, params: dict) -> None:
    """Design sequences for a backbone with LigandMPNN and persist the output.

    The output directory holds the designed sequences as fasta under seqs/ and the
    designed backbones under backbones/. run.py is invoked from the LigandMPNN
    directory because it imports its sibling modules by bare name.

    Args:
        job_id: Unique id identifying the job.
        job_name: The caller's label for the job, recorded in the run log. None
            when the caller did not supply one.
        params: A LigandMPNNParams dump, revalidated here so the container never
            trusts the payload it was handed.
    """
    logger.info(f"Starting ligandmpnn design: job={job_id}")
    started_at = datetime.now(UTC)
    job_command = ""
    stderr = ""

    try:
        job_params = LigandMPNNParams.model_validate(params)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdb_path = tmpdir / "input.pdb"
            pdb_path.write_text(job_params.pdb)
            output_dir = tmpdir / "output"
            checkpoint = f"{VOLUME_LIGANDMPNN_CACHE}/ligandmpnn_v_32_{job_params.noise}_25.pt"

            cmd = [
                "python",
                "run.py",
                "--model_type",
                "ligand_mpnn",
                "--checkpoint_ligand_mpnn",
                checkpoint,
                "--pdb_path",
                str(pdb_path),
                "--out_folder",
                str(output_dir),
                "--number_of_batches",
                str(job_params.num_sequences),
                "--temperature",
                str(job_params.temperature),
                "--ligand_mpnn_use_atom_context",
                str(int(job_params.use_atom_context)),
                "--ligand_mpnn_use_side_chain_context",
                str(int(job_params.use_side_chain_context)),
                "--seed",
                str(job_params.seed),
            ]
            job_command = " ".join(cmd)
            logger.info(f"Running: {job_command}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=LIGANDMPNN_DIR)
            stderr = result.stderr

            if result.returncode != 0:
                logger.error(f"ligandmpnn stderr: {stderr}")
                raise RuntimeError(f"ligandmpnn run exited with code {result.returncode}")

            persist_job_output(job_id, output_dir)

        log = format_run_log(
            job_id, job_name, "ligandmpnn", LIGANDMPNN_SPEC, job_command, result.stdout, stderr, started_at
        )
        mark_job_complete(job_id, log)
        logger.info(f"Done: job={job_id}")
    except Exception as e:
        logger.error(f"Failed: job={job_id}: {e}")
        log = format_run_log(job_id, job_name, "ligandmpnn", LIGANDMPNN_SPEC, job_command, str(e), stderr, started_at)
        mark_job_failed(job_id, log)
        raise
