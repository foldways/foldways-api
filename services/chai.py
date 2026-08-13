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

    Restraints and templates steer the prediction. `restraints` supplies pocket, contact,
    covalent-bond, and glycosylation constraints as an inline CSV. Templates come from the
    public server with `use_templates_server`, or inline via `template_hits`. MSAs come
    from the public server with `use_msa_server`, on by default as Chai recommends. Turn
    it off to run single-sequence and offline.

    Precomputed MSA directories, the one remaining Chai input, are not exposed: they are
    directories of binary parquet files with no clean inline-JSON form. Use the MSA server
    or run single-sequence instead.
    """

    fasta: str = Field(
        description="Inline FASTA in Chai format. Headers are >protein|name=..., >ligand|name=... "
        "(SMILES body), >dna, >rna, or >glycan.",
    )

    # Restraints and templates, given as inline file content and written to a temp file.
    restraints: str | None = Field(
        default=None,
        description="Inline restraints as CSV, for pocket, contact, covalent-bond, and glycosylation "
        "constraints. Columns: chainA,res_idxA,chainB,res_idxB,connection_type,confidence,"
        "min_distance_angstrom,max_distance_angstrom,comment,restraint_id.",
    )
    use_templates_server: bool = Field(
        default=False, description="Fetch structural templates from the public server. Adds a network call."
    )
    template_hits: str | None = Field(
        default=None, description="Inline template hits in foldseek m8 format, as an alternative to the server."
    )

    # MSA.
    use_msa_server: bool = Field(
        default=True,
        description="Build MSAs from the public server, on by default as Chai recommends for accuracy. "
        "Set false to run single-sequence and offline, which adds no network call.",
    )
    msa_server_url: str = Field(
        default="https://api.colabfold.com",
        description="MSA server used when use_msa_server is set. Point at a self-hosted server to avoid "
        "sending sequences to the public one.",
    )

    # Sampling and quality.
    num_diffn_samples: int = Field(default=5, ge=1, description="Number of structures to sample and rank.")
    num_trunk_samples: int = Field(
        default=1, ge=1, description="Number of trunk samples. Total structures is this times num_diffn_samples."
    )
    num_diffn_timesteps: int = Field(
        default=200, ge=1, description="Diffusion denoising steps. More gives higher quality and runs slower."
    )
    num_trunk_recycles: int = Field(
        default=3, ge=1, description="Trunk recycling iterations. More gives higher accuracy and runs slower."
    )
    recycle_msa_subsample: int = Field(
        default=0, ge=0, description="MSA rows to subsample each recycle. 0 uses the full MSA."
    )
    use_esm_embeddings: bool = Field(default=True, description="Use ESM single-sequence embeddings.")
    low_memory: bool = Field(default=True, description="Trade speed for lower GPU memory, which large complexes need.")
    fasta_names_as_cif_chains: bool = Field(
        default=False,
        description="Use FASTA entity names as chain ids. Restraints must then reference those names.",
    )
    seed: int | None = Field(default=None, description="Random seed. Null draws a fresh seed each run.")


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
    with pTM, ipTM, pLDDT, and clash metrics. Structures are ranked best-first. Restraints
    and template hits are written to temp files, since Chai reads them from disk.

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

            constraint_path = None
            if job_params.restraints is not None:
                constraint_path = tmpdir / "restraints.csv"
                constraint_path.write_text(job_params.restraints)

            template_hits_path = None
            if job_params.template_hits is not None:
                template_hits_path = tmpdir / "template_hits.m8"
                template_hits_path.write_text(job_params.template_hits)

            candidates = run_inference(
                fasta_file=fasta_path,
                output_dir=output_dir,
                use_esm_embeddings=job_params.use_esm_embeddings,
                use_msa_server=job_params.use_msa_server,
                msa_server_url=job_params.msa_server_url,
                constraint_path=constraint_path,
                use_templates_server=job_params.use_templates_server,
                template_hits_path=template_hits_path,
                recycle_msa_subsample=job_params.recycle_msa_subsample,
                num_trunk_recycles=job_params.num_trunk_recycles,
                num_diffn_timesteps=job_params.num_diffn_timesteps,
                num_diffn_samples=job_params.num_diffn_samples,
                num_trunk_samples=job_params.num_trunk_samples,
                seed=job_params.seed,
                device="cuda:0",
                low_memory=job_params.low_memory,
                fasta_names_as_cif_chains=job_params.fasta_names_as_cif_chains,
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
