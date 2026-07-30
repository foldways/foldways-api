import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import modal
from pydantic import BaseModel, Field, model_validator

from common.utils import format_run_log, mark_job_complete, mark_job_failed, persist_job_output
from config import ESMFOLD2_GPU, ESMFOLD2_MAX_CONTAINERS, ESMFOLD2_SCALEDOWN_WINDOW, ESMFOLD2_TIMEOUT
from constants import (
    ESMC_SPEC,
    ESMFOLD2_WEIGHTS_REPO,
    PYDANTIC_SPEC,
    PYTHON_3_12,
    SERVICE_SOURCES,
    VOLUME_ESMFOLD2_CACHE,
    VOLUME_ROOT,
)
from core import app, volume

logger = logging.getLogger(__name__)


esmfold2_image = (
    modal.Image.debian_slim(python_version=PYTHON_3_12)
    .apt_install("git")
    .uv_pip_install(ESMC_SPEC, PYDANTIC_SPEC)
    .env({"HF_HUB_CACHE": VOLUME_ESMFOLD2_CACHE})
    .add_local_python_source(*SERVICE_SOURCES)
)


class Modification(BaseModel):
    position: int = Field(ge=0, description="Zero-indexed residue position of the modification.", examples=[0])
    ccd: str = Field(description="CCD code of the modified residue.", examples=["SEP"])


class ProteinChain(BaseModel):
    id: str | list[str] = Field(description="Chain id, or a list of ids for identical copies.", examples=["A"])
    sequence: str = Field(description="Amino-acid sequence in one-letter codes.", examples=["MKTAYIAKQR"])
    modifications: list[Modification] | None = Field(default=None, description="Modified residues in the chain.")


class NucleicChain(BaseModel):
    id: str | list[str] = Field(description="Chain id, or a list of ids for identical copies.", examples=["B"])
    sequence: str = Field(description="Nucleotide sequence.", examples=["GATC"])
    modifications: list[Modification] | None = Field(default=None, description="Modified residues in the chain.")


class Ligand(BaseModel):
    """One non-polymer molecule. Give exactly one of `smiles` or `ccd`."""

    id: str | list[str] = Field(description="Chain id, or a list of ids for identical copies.", examples=["L"])
    smiles: str | None = Field(default=None, description="Ligand as a SMILES string (exclusive with ccd).")
    ccd: list[str] | None = Field(default=None, description="Ligand as CCD codes (exclusive with smiles).")

    @model_validator(mode="after")
    def require_smiles_or_ccd(self):
        if bool(self.smiles) == bool(self.ccd):
            raise ValueError("ligand requires exactly one of 'smiles' or 'ccd'")
        return self


class Entity(BaseModel):
    """One chain or molecule in the complex. Set exactly one entity type."""

    protein: ProteinChain | None = Field(default=None, description="A protein chain.")
    dna: NucleicChain | None = Field(default=None, description="A DNA chain.")
    rna: NucleicChain | None = Field(default=None, description="An RNA chain.")
    ligand: Ligand | None = Field(default=None, description="A non-polymer molecule.")

    @model_validator(mode="after")
    def validate_one_entity_type(self):
        if sum(getattr(self, k) is not None for k in ("protein", "dna", "rna", "ligand")) != 1:
            raise ValueError("each entity must set exactly one of protein/dna/rna/ligand")
        return self


class ESMFold2Params(BaseModel):
    """Parameters ESMFold2 needs to predict one biomolecular complex structure.

    `sequences` declares the entities in the complex. The remaining fields tune the
    diffusion sampling. Set diffusion_samples above one to fold several structures
    from the same input, and one result file is written per sample.
    """

    sequences: list[Entity] = Field(
        min_length=1,
        description="One entry per entity in the complex, each setting exactly one of protein, dna, rna, or ligand.",
    )
    num_loops: int = Field(
        default=20, ge=1, le=100, description="Recycling loops. More gives higher accuracy and runs slower."
    )
    num_sampling_steps: int = Field(default=200, ge=1, le=500, description="Diffusion denoising steps.")
    diffusion_samples: int = Field(default=1, ge=1, le=25, description="Number of structures to generate.")
    seed: int | None = Field(default=None, description="Random seed for reproducible sampling.")


def build_structure_input(sequences: list[Entity]):
    """Convert the request entities into an esm StructurePredictionInput."""
    from esm.models.esmfold2 import (  # pyright: ignore[reportMissingImports]
        DNAInput,
        LigandInput,
        Modification,
        ProteinInput,
        RNAInput,
        StructurePredictionInput,
    )

    def mods(entity_mods):
        return [Modification(position=m.position, ccd=m.ccd) for m in entity_mods] if entity_mods else None

    entities = []
    for entity in sequences:
        if entity.protein is not None:
            p = entity.protein
            entities.append(ProteinInput(id=p.id, sequence=p.sequence, modifications=mods(p.modifications)))
        elif entity.dna is not None:
            d = entity.dna
            entities.append(DNAInput(id=d.id, sequence=d.sequence, modifications=mods(d.modifications)))
        elif entity.rna is not None:
            r = entity.rna
            entities.append(RNAInput(id=r.id, sequence=r.sequence, modifications=mods(r.modifications)))
        elif entity.ligand is not None:
            ligand = entity.ligand
            entities.append(LigandInput(id=ligand.id, smiles=ligand.smiles, ccd=ligand.ccd))
    return StructurePredictionInput(sequences=entities)


@app.function(
    name="esmfold2",
    image=esmfold2_image,
    gpu=ESMFOLD2_GPU,
    volumes={VOLUME_ROOT: volume},
    timeout=ESMFOLD2_TIMEOUT,
    max_containers=ESMFOLD2_MAX_CONTAINERS,
    scaledown_window=ESMFOLD2_SCALEDOWN_WINDOW,
)
def run(job_id: str, job_name: str | None, params: dict) -> None:
    """Predict one complex structure with ESMFold2 and persist the result.

    Writes a prediction_{i}.cif and a confidence_{i}.json (pLDDT, pTM, ipTM) per
    diffusion sample.

    Args:
        job_id: Unique id identifying the job.
        job_name: The caller's label for the job, recorded in the run log. None
            when the caller did not supply one.
        params: An ESMFold2Params dump, revalidated here so the container never
            trusts the payload it was handed.
    """
    from esm.models.esmfold2 import ESMFold2InputBuilder  # pyright: ignore[reportMissingImports]
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # pyright: ignore[reportMissingImports]

    logger.info(f"Starting esmfold2 prediction: job={job_id}")
    started_at = datetime.now(UTC)
    stderr = ""

    try:
        job_params = ESMFold2Params.model_validate(params)
        model = ESMFold2Model.from_pretrained(ESMFOLD2_WEIGHTS_REPO).cuda().eval()
        structure_input = build_structure_input(job_params.sequences)

        result = ESMFold2InputBuilder().fold(
            model,
            structure_input,
            num_loops=job_params.num_loops,
            num_sampling_steps=job_params.num_sampling_steps,
            num_diffusion_samples=job_params.diffusion_samples,
            seed=job_params.seed,
        )
        results = result if isinstance(result, list) else [result]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            for i, sample in enumerate(results):
                (output_dir / f"prediction_{i}.cif").write_text(sample.complex.to_mmcif())
                confidence = {
                    "plddt": sample.plddt.float().cpu().numpy().tolist() if sample.plddt is not None else None,
                    "ptm": float(sample.ptm) if sample.ptm is not None else None,
                    "iptm": float(sample.iptm) if sample.iptm is not None else None,
                }
                (output_dir / f"confidence_{i}.json").write_text(json.dumps(confidence, indent=2))
            persist_job_output(job_id, output_dir)

        summary = f"Predicted {len(results)} structure(s) for {len(job_params.sequences)} entities"
        log = format_run_log(job_id, job_name, "esmfold2", ESMC_SPEC, "", summary, stderr, started_at)
        mark_job_complete(job_id, log)
        logger.info(f"Done: job={job_id}")
    except Exception as e:
        logger.error(f"Failed: job={job_id}: {e}")
        log = format_run_log(job_id, job_name, "esmfold2", ESMC_SPEC, "", str(e), stderr, started_at)
        mark_job_failed(job_id, log)
        raise
