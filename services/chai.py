import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import modal
from pydantic import BaseModel, Field

from common.utils import format_run_log, mark_job_complete, mark_job_failed, persist_job_output
from config import CHAI_GPU, CHAI_MAX_CONTAINERS, CHAI_SCALEDOWN_WINDOW, CHAI_TIMEOUT
from constants import (
    CHAI_SPEC,
    PYDANTIC_SPEC,
    PYTHON_3_12,
    SERVICE_SOURCES,
    VOLUME_CHAI_CACHE,
    VOLUME_ROOT,
)
from core import app, volume

logger = logging.getLogger(__name__)


# CHAI_DOWNLOADS_DIR points Chai at the weights staged on the volume, so it downloads nothing
# at inference as long as MSAs and templates are not requested from a server.
chai_image = (
    modal.Image.debian_slim(python_version=PYTHON_3_12)
    .apt_install("git")
    .uv_pip_install(CHAI_SPEC, PYDANTIC_SPEC)
    .env({"CHAI_DOWNLOADS_DIR": VOLUME_CHAI_CACHE})
    .add_local_python_source(*SERVICE_SOURCES)
)


class ChaiParams(BaseModel):
    """Parameters for all-atom structure prediction with Chai-1.

    The complex is given as an inline FASTA in Chai's format, where each entity has a
    typed header. Proteins are `>protein|name=...`, ligands are `>ligand|name=...` with
    a SMILES body, and nucleic acids and glycans use `>dna`, `>rna`, and `>glycan`.
    Chai folds every entity together, so one job predicts a whole multi-modal complex.

    By default it runs single-sequence, without MSAs. Setting `use_msa_server` builds
    MSAs at runtime from the public server, which improves accuracy and adds a network
    call. `num_diffn_samples` structures are generated and ranked best-first.
    """

    fasta: str = Field(
        description="Inline FASTA in Chai format. Headers are >protein|name=..., >ligand|name=... "
        "(SMILES body), >dna, >rna, or >glycan.",
    )
    num_diffn_samples: int = Field(default=5, ge=1, le=5, description="Number of structures to sample and rank.")
    num_trunk_recycles: int = Field(
        default=3, ge=1, description="Trunk recycling iterations. More gives higher accuracy and runs slower."
    )
    use_msa_server: bool = Field(
        default=False,
        description="Build MSAs at runtime from the public server. Improves accuracy, adds a network call.",
    )
    seed: int = Field(default=0, description="Random seed for reproducible sampling.")


@app.function(
    name="chai",
    image=chai_image,
    gpu=CHAI_GPU,
    volumes={VOLUME_ROOT: volume},
    timeout=CHAI_TIMEOUT,
    max_containers=CHAI_MAX_CONTAINERS,
    scaledown_window=CHAI_SCALEDOWN_WINDOW,
)
def run(job_id: str, job_name: str | None, params: dict) -> None:
    """Predict a complex structure with Chai-1 and persist the output.

    The output directory holds one mmCIF per ranked sample and a scores npz per sample
    with pTM, ipTM, pLDDT, and clash metrics. Structures are ranked best-first.

    Args:
        job_id: Unique id identifying the job.
        job_name: The caller's label for the job, recorded in the run log. None
            when the caller did not supply one.
        params: A ChaiParams dump, revalidated here so the container never trusts the
            payload it was handed.
    """
    from chai_lab.chai1 import run_inference  # pyright: ignore[reportMissingImports]

    logger.info(f"Starting chai job={job_id}")
    started_at = datetime.now(UTC)
    stderr = ""

    try:
        job_params = ChaiParams.model_validate(params)
        volume.reload()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            fasta_path = tmpdir / "input.fasta"
            fasta_path.write_text(job_params.fasta)
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            candidates = run_inference(
                fasta_file=fasta_path,
                output_dir=output_dir,
                num_trunk_recycles=job_params.num_trunk_recycles,
                num_diffn_samples=job_params.num_diffn_samples,
                use_msa_server=job_params.use_msa_server,
                use_esm_embeddings=True,
                seed=job_params.seed,
                device="cuda:0",
            )
            persist_job_output(job_id, output_dir)

        summary = f"Predicted {len(candidates.cif_paths)} structure sample(s)"
        log = format_run_log(job_id, job_name, "chai", CHAI_SPEC, "", summary, stderr, started_at)
        mark_job_complete(job_id, log)
        logger.info(f"Done: job={job_id}")
    except Exception as e:
        logger.error(f"Failed: job={job_id}: {e}")
        log = format_run_log(job_id, job_name, "chai", CHAI_SPEC, "", str(e), stderr, started_at)
        mark_job_failed(job_id, log)
        raise
