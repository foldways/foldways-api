import logging
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import modal
from pydantic import BaseModel, Field, model_validator

from common.utils import (
    build_shared_args,
    format_run_log,
    mark_job_complete,
    mark_job_failed,
    persist_job_output,
)
from config import SOLUBLEMPNN_GPU, SOLUBLEMPNN_MAX_CONTAINERS, SOLUBLEMPNN_SCALEDOWN_WINDOW, SOLUBLEMPNN_TIMEOUT
from constants import (
    LIGANDMPNN_COMMIT,
    LIGANDMPNN_DIR,
    LIGANDMPNN_REPO,
    LIGANDMPNN_SC_CHECKPOINT,
    PYDANTIC_SPEC,
    PYTHON_3_11,
    SERVICE_SOURCES,
    SOLUBLEMPNN_SPEC,
    VOLUME_ROOT,
    VOLUME_SOLUBLEMPNN_CACHE,
)
from core import app, volume

logger = logging.getLogger(__name__)


solublempnn_image = (
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


class SolubleMPNNParams(BaseModel):
    """Parameters for designing sequences for a backbone with SolubleMPNN.

    SolubleMPNN is an inverse-folding model, the ProteinMPNN architecture trained
    only on soluble proteins. Given a backbone structure it designs amino-acid
    sequences predicted to fold into it while biasing away from residues typical
    of membrane or buried environments.
    """

    pdb: str = Field(description="Inline PDB content of the backbone to design sequences for.")
    num_sequences: int = Field(
        default=1, ge=1, le=100, description="Number of design passes. Total sequences is this times batch_size."
    )
    batch_size: int = Field(
        default=1, ge=1, description="Sequences decoded in parallel per pass. Higher uses more GPU memory."
    )
    temperature: float = Field(
        default=0.1, gt=0, le=1.0, description="Sampling temperature. Higher gives more diversity."
    )
    noise: Literal["002", "010", "020", "030"] = Field(
        default="020",
        description="Backbone noise the checkpoint was trained with, in hundredths of an Angstrom.",
    )
    seed: int = Field(
        default=0,
        description="Random seed for sampling. Passing 0 draws a fresh random seed each time. Pass a nonzero "
        "value for reproducible runs.",
    )

    fixed_residues: str | None = Field(
        default=None,
        description="Residues to keep fixed, space-separated as chain plus number, e.g. `A12 A13 B2`. "
        "Everything else is redesigned. Mutually exclusive with redesigned_residues.",
        examples=["A12 A13 A14"],
    )
    redesigned_residues: str | None = Field(
        default=None,
        description="Residues to redesign, space-separated like fixed_residues. Everything else is kept fixed.",
    )
    chains_to_design: str | None = Field(
        default=None, description="Chains to redesign, comma-separated, e.g. `A,B`. Others are kept fixed."
    )
    parse_these_chains_only: str | None = Field(
        default=None, description="Only parse these chains from the PDB, comma-separated, e.g. `A,B`."
    )
    bias_aa: str | None = Field(
        default=None,
        description="Global per-amino-acid bias, e.g. `A:-1.02,P:2.34`. Positive favors, negative disfavors.",
    )
    bias_aa_per_residue: dict | None = Field(
        default=None,
        description="Per-residue amino-acid bias, e.g. `{'A12': {'G': -0.3, 'C': -2.0}}`.",
    )
    omit_aa: str | None = Field(
        default=None, description="Amino acids to forbid globally, as one string, e.g. `CM` to omit Cys and Met."
    )
    omit_aa_per_residue: dict | None = Field(
        default=None, description="Per-residue amino acids to forbid, e.g. `{'A12': 'APQ', 'A13': 'QST'}`."
    )
    symmetry_residues: str | None = Field(
        default=None,
        description="Groups of positions tied to be identical, e.g. `A12,A13|C2,C3`. Pipe separates groups.",
    )
    symmetry_weights: str | None = Field(
        default=None, description="Weights matching symmetry_residues, e.g. `1.0,1.0|2.0,2.0`."
    )
    homo_oligomer: bool = Field(
        default=False, description="Tie all chains as a homo-oligomer, setting symmetry across chains automatically."
    )
    save_stats: bool = Field(
        default=False, description="Also save per-design statistics such as scores and probabilities."
    )
    parse_atoms_with_zero_occupancy: bool = Field(
        default=False, description="Parse atoms with zero occupancy in the input PDB rather than dropping them."
    )
    zero_indexed: bool = Field(default=False, description="Number the output PDB starting from 0 rather than 1.")
    pack_side_chains: bool = Field(
        default=False, description="Run the side-chain packer to output full-atom packed structures."
    )
    number_of_packs_per_design: int = Field(
        default=4, ge=1, description="Independent side-chain packing samples per design."
    )
    sc_num_denoising_steps: int = Field(default=3, ge=1, description="Denoising steps for side-chain packing.")
    sc_num_samples: int = Field(default=16, ge=1, description="Samples drawn per side-chain packing step.")
    repack_everything: bool = Field(
        default=False, description="Repack all side chains, including fixed residues, rather than only designed ones."
    )

    @model_validator(mode="after")
    def check_design_control(self) -> "SolubleMPNNParams":
        """Reject fixed_residues and redesigned_residues together, since run.py takes only one."""
        if self.fixed_residues and self.redesigned_residues:
            raise ValueError("Provide fixed_residues or redesigned_residues, not both.")
        return self


@app.function(
    name="solublempnn",
    image=solublempnn_image,
    gpu=SOLUBLEMPNN_GPU,
    volumes={VOLUME_ROOT: volume},
    timeout=SOLUBLEMPNN_TIMEOUT,
    max_containers=SOLUBLEMPNN_MAX_CONTAINERS,
    scaledown_window=SOLUBLEMPNN_SCALEDOWN_WINDOW,
)
def run(job_id: str, job_name: str | None, params: dict) -> None:
    """Design sequences for a backbone with SolubleMPNN and persist the output.

    The output directory holds the designed sequences as fasta under seqs/ and the
    designed backbones under backbones/. run.py is invoked from the LigandMPNN
    directory because it imports its sibling modules by bare name.

    Args:
        job_id: Unique id identifying the job.
        job_name: The caller's label for the job, recorded in the run log. None
            when the caller did not supply one.
        params: A SolubleMPNNParams dump, revalidated here so the container never
            trusts the payload it was handed.
    """
    logger.info(f"Starting solublempnn design: job={job_id}")
    started_at = datetime.now(UTC)
    job_command = ""
    stderr = ""

    try:
        job_params = SolubleMPNNParams.model_validate(params)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            pdb_path = tmpdir / "input.pdb"
            pdb_path.write_text(job_params.pdb)
            output_dir = tmpdir / "output"
            checkpoint = f"{VOLUME_SOLUBLEMPNN_CACHE}/solublempnn_v_48_{job_params.noise}.pt"

            cmd = [
                "python",
                "run.py",
                "--model_type",
                "soluble_mpnn",
                "--checkpoint_soluble_mpnn",
                checkpoint,
                "--pdb_path",
                str(pdb_path),
                "--out_folder",
                str(output_dir),
                "--number_of_batches",
                str(job_params.num_sequences),
                "--temperature",
                str(job_params.temperature),
                "--seed",
                str(job_params.seed),
            ]
            cmd += build_shared_args(job_params, tmpdir, f"{VOLUME_SOLUBLEMPNN_CACHE}/{LIGANDMPNN_SC_CHECKPOINT}")
            job_command = " ".join(cmd)
            logger.info(f"Running: {job_command}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=LIGANDMPNN_DIR)
            stderr = result.stderr

            if result.returncode != 0:
                logger.error(f"solublempnn stderr: {stderr}")
                raise RuntimeError(f"solublempnn run exited with code {result.returncode}")

            persist_job_output(job_id, output_dir)

        log = format_run_log(
            job_id, job_name, "solublempnn", SOLUBLEMPNN_SPEC, job_command, result.stdout, stderr, started_at
        )
        mark_job_complete(job_id, log)
        logger.info(f"Done: job={job_id}")
    except Exception as e:
        logger.error(f"Failed: job={job_id}: {e}")
        log = format_run_log(job_id, job_name, "solublempnn", SOLUBLEMPNN_SPEC, job_command, str(e), stderr, started_at)
        mark_job_failed(job_id, log)
        raise
