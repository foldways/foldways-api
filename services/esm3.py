import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import modal
from pydantic import BaseModel, Field, model_validator

from common.utils import format_run_log, mark_job_complete, mark_job_failed, persist_job_output
from config import ESM3_GPU, ESM3_MAX_CONTAINERS, ESM3_SCALEDOWN_WINDOW, ESM3_TIMEOUT
from constants import (
    ESM3_MODEL_NAME,
    ESM3_SPEC,
    ESM3_WEIGHTS_REPO,
    PYDANTIC_SPEC,
    PYTHON_3_12,
    SERVICE_SOURCES,
    VOLUME_ESM3_CACHE,
    VOLUME_ROOT,
)
from core import app, volume

logger = logging.getLogger(__name__)


esm3_image = (
    modal.Image.debian_slim(python_version=PYTHON_3_12)
    .apt_install("git")
    .uv_pip_install(ESM3_SPEC, PYDANTIC_SPEC)
    .env({"HF_HUB_CACHE": VOLUME_ESM3_CACHE})
    .add_local_python_source(*SERVICE_SOURCES)
)


class ESM3Params(BaseModel):
    """Parameters for generating protein designs with ESM3.

    ESM3 is a generative masked language model. It is prompted with a sequence in
    which `_` marks the positions to generate, then iteratively unmasks them. An
    all-`_` prompt is de novo generation. A prompt that fixes some residues and
    masks the rest is a scaffolding or infilling design. One design is written per
    prompt, so a job can cover a batch of them.

    When `predict_structure` is set, each generated sequence is folded and its
    structure written as a PDB, following ESM3's own sequence-then-structure flow.
    """

    prompts: list[str] = Field(
        min_length=1,
        description="Sequence prompts in one-letter codes, with `_` for positions to generate.",
        examples=[["___________________________"]],
    )
    num_steps: int = Field(
        default=8,
        ge=1,
        description="Iterative unmasking steps, capped at the prompt length. More gives higher quality, "
        "with diminishing returns past about 20.",
    )
    temperature: float = Field(
        default=0.7, gt=0, le=2.0, description="Sampling temperature. Higher gives more diversity."
    )
    predict_structure: bool = Field(
        default=False, description="After generating each sequence, fold it and write a PDB."
    )

    @model_validator(mode="after")
    def reject_empty_prompts(self):
        if any(not prompt for prompt in self.prompts):
            raise ValueError("prompts must not contain empty strings")
        return self


def load_esm3_model():
    """Load ESM3-open from the weights staged on the volume.

    ESM3.from_pretrained resolves the open model and its structure and function
    submodels through snapshot_download, which reads the HF cache on the volume set
    by HF_HUB_CACHE. It moves the model to CUDA and casts to bfloat16 on its own.
    """
    from esm.models.esm3 import ESM3  # pyright: ignore[reportMissingImports]

    return ESM3.from_pretrained(ESM3_MODEL_NAME)


def generate_designs(
    model, prompts: list[str], num_steps: int, temperature: float, predict_structure: bool, output_dir: Path
) -> list[dict]:
    """Generate a design per prompt, write designs.fasta, and return per-design metadata.

    Each prompt's `_` positions are filled by iteratively unmasking the sequence
    track. num_steps is capped at the prompt length, which ESM3 requires. When
    predict_structure is set, the completed sequence is folded and written as
    structure_{i}.pdb, and its pTM is recorded when the model returns one.
    """
    from esm.sdk.api import ESMProtein, ESMProteinError, GenerationConfig  # pyright: ignore[reportMissingImports]

    def raise_on_error(result, track: str):
        # generate returns an ESMProteinError rather than raising it, so surface its message.
        if isinstance(result, ESMProteinError):
            raise RuntimeError(f"ESM3 {track} generation failed ({result.error_code}): {result.error_msg}")
        return result

    results = []
    fasta_lines = []
    for i, prompt in enumerate(prompts):
        steps = min(num_steps, len(prompt))
        protein = ESMProtein(sequence=prompt)
        config = GenerationConfig(track="sequence", num_steps=steps, temperature=temperature)
        protein = raise_on_error(model.generate(protein, config), "sequence")
        fasta_lines.append(f">design_{i}\n{protein.sequence}")
        entry = {"index": i, "sequence": protein.sequence, "length": len(protein.sequence)}

        if predict_structure:
            protein = raise_on_error(
                model.generate(protein, GenerationConfig(track="structure", num_steps=steps)), "structure"
            )
            protein.to_pdb(str(output_dir / f"structure_{i}.pdb"))
            if protein.ptm is not None:
                entry["ptm"] = float(protein.ptm)
        results.append(entry)

    (output_dir / "designs.fasta").write_text("\n".join(fasta_lines) + "\n")
    return results


@app.function(
    name="esm3",
    image=esm3_image,
    gpu=ESM3_GPU,
    volumes={VOLUME_ROOT: volume},
    timeout=ESM3_TIMEOUT,
    max_containers=ESM3_MAX_CONTAINERS,
    scaledown_window=ESM3_SCALEDOWN_WINDOW,
)
def run(job_id: str, job_name: str | None, params: dict) -> None:
    """Generate protein designs with ESM3 and persist the output.

    The generated sequences are written to designs.fasta, one record per prompt.
    When predict_structure is set, each design's folded structure is written as
    structure_{i}.pdb. metadata.json records the per-design sequences and pTM.

    Args:
        job_id: Unique id identifying the job.
        job_name: The caller's label for the job, recorded in the run log. None
            when the caller did not supply one.
        params: An ESM3Params dump, revalidated here so the container never trusts
            the payload it was handed.
    """
    logger.info(f"Starting esm3 job={job_id}")
    started_at = datetime.now(UTC)
    stderr = ""

    try:
        job_params = ESM3Params.model_validate(params)
        model = load_esm3_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            results = generate_designs(
                model,
                job_params.prompts,
                job_params.num_steps,
                job_params.temperature,
                job_params.predict_structure,
                output_dir,
            )
            metadata = {
                "model": ESM3_WEIGHTS_REPO,
                "num_prompts": len(job_params.prompts),
                "num_steps": job_params.num_steps,
                "temperature": job_params.temperature,
                "predict_structure": job_params.predict_structure,
                "designs": results,
            }
            (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
            persist_job_output(job_id, output_dir)

        summary = f"Generated {len(results)} design(s)"
        log = format_run_log(job_id, job_name, "esm3", ESM3_SPEC, "", summary, stderr, started_at)
        mark_job_complete(job_id, log)
        logger.info(f"Done: job={job_id}")
    except Exception as e:
        logger.error(f"Failed: job={job_id}: {e}")
        log = format_run_log(job_id, job_name, "esm3", ESM3_SPEC, "", str(e), stderr, started_at)
        mark_job_failed(job_id, log)
        raise
