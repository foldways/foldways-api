import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import modal
from pydantic import BaseModel, Field, model_validator

from common.utils import format_run_log, mark_job_complete, mark_job_failed, persist_job_output
from config import VESM_GPU, VESM_MAX_CONTAINERS, VESM_SCALEDOWN_WINDOW, VESM_TIMEOUT
from constants import (
    ONE_LETTER_AMINO_ACIDS,
    PYDANTIC_SPEC,
    PYTHON_3_12,
    SERVICE_SOURCES,
    VESM_MODEL_WINDOW,
    VESM_MODELS,
    VESM_NUMPY_SPEC,
    VESM_SPEC,
    VESM_TORCH_SPEC,
    VESM_TRANSFORMERS_SPEC,
    VESM_WEIGHTS_REPO,
    VOLUME_ROOT,
    VOLUME_VESM_CACHE,
)
from core import app, volume

logger = logging.getLogger(__name__)


vesm_image = (
    modal.Image.debian_slim(python_version=PYTHON_3_12)
    .uv_pip_install(VESM_TORCH_SPEC, VESM_TRANSFORMERS_SPEC, VESM_NUMPY_SPEC, PYDANTIC_SPEC)
    .env({"HF_HUB_CACHE": VOLUME_VESM_CACHE})
    .add_local_python_source(*SERVICE_SOURCES)
)


class VESMParams(BaseModel):
    """Parameters for scoring sequence variants with VESM.

    VESM is a variant effect predictor distilled from the ESM2 family. Given a
    wildtype protein sequence and a set of mutations it returns a score per
    mutation, the summed log-likelihood ratio of the mutant residues against the
    wildtype. More negative means more deleterious.

    Each mutation is a wildtype residue, a 1-based position, and a mutant residue,
    e.g. `M1Y`. A double mutant joins single mutations with a colon, e.g. `M1Y:V2T`.

    Sequences longer than the model's window are scored in overlapping windows, so
    there is no length limit. Each position is scored in the window that centers it
    best, following VESM's own tiling.
    """

    sequence: str = Field(
        description="Wildtype protein sequence in one-letter codes.",
        examples=["MVNSTHRGMHTSLHLWNRSSYRLHSNASESLGKGYSDGGCYEQLFVSPEVFVTLGVISLLENILV"],
    )
    mutations: list[str] = Field(
        min_length=1,
        description="Mutations to score, each as wildtype residue, 1-based position, and mutant residue. "
        "Join a multiple mutant with colons, e.g. `M1Y:V2T`.",
        examples=[["M1Y", "V2T", "M1Y:V2T"]],
    )
    model_name: Literal["VESM_35M", "VESM_150M", "VESM_650M", "VESM_3B"] = Field(
        default="VESM_650M",
        description="Which distilled VESM model to score with. Larger models are more accurate and slower.",
    )

    @model_validator(mode="after")
    def check_sequence_and_mutations(self) -> "VESMParams":
        """Reject an empty sequence and any mutation that does not match it.

        Each mutation's wildtype residue and position are checked against the
        sequence here, so a malformed or mismatched mutation fails before the GPU
        run rather than producing a silently wrong score.
        """
        if not self.sequence:
            raise ValueError("sequence must not be empty.")
        for variant in self.mutations:
            for mutation in variant.split(":"):
                wildtype, position, mutant = mutation[:1], mutation[1:-1], mutation[-1:]
                if wildtype not in ONE_LETTER_AMINO_ACIDS or mutant not in ONE_LETTER_AMINO_ACIDS:
                    raise ValueError(f"Mutation '{mutation}' must be a residue, a position, and a residue, e.g. M1Y.")
                if not position.isdigit() or not 1 <= int(position) <= len(self.sequence):
                    raise ValueError(f"Mutation '{mutation}' has a position outside the sequence.")
                if self.sequence[int(position) - 1] != wildtype:
                    raise ValueError(f"Mutation '{mutation}' wildtype residue does not match the sequence.")
        return self


def load_vesm_model(model_name: str):
    """Load a VESM model from the base ESM2 weights and distilled head on the volume.

    The base ESM2 model and tokenizer resolve through the HF cache set by
    HF_HUB_CACHE, then the distilled VESM checkpoint is loaded over it. strict is
    False because the checkpoint only carries the weights VESM changed.
    """
    import torch  # pyright: ignore[reportMissingImports]
    from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports]
    from transformers import AutoTokenizer, EsmForMaskedLM  # pyright: ignore[reportMissingImports]

    base_repo = VESM_MODELS[model_name]
    model = EsmForMaskedLM.from_pretrained(base_repo)
    tokenizer = AutoTokenizer.from_pretrained(base_repo)
    weights = hf_hub_download(repo_id=VESM_WEIGHTS_REPO, filename=f"{model_name}.pth")
    model.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
    return model.to("cuda").eval(), tokenizer


def get_interval(one_based_position: int, seq_length: int, model_window: int = VESM_MODEL_WINDOW) -> tuple[int, int]:
    """Return the [start, end) window a position is scored in, ported from VESM.

    A sequence within the window is scored whole. A longer one places each position
    in a model_window-sized window offset so the position sits away from the window
    edges, where the model has least context. This mirrors VESM's utils/seq_ops.py
    so scores match the upstream tool. The end is always min(start + model_window,
    seq_length), so the window start alone identifies the window.
    """
    half_window = model_window // 2
    if seq_length <= model_window:
        return 0, seq_length
    position = one_based_position - 1
    block = (position // half_window) * half_window
    if block < half_window:
        return 0, model_window
    if block + half_window > seq_length:
        return max(0, block - half_window), seq_length
    if position - block < block + half_window - position:
        return max(0, block - half_window), min(seq_length, block + half_window)
    return block, min(seq_length, block + model_window)


def score_variants(model, tokenizer, sequence: str, mutations: list[str]) -> list[dict]:
    """Score each mutation as the summed log-likelihood ratio of its mutant residues.

    The masked-LM logits give a per-position distribution over residues. The
    log-likelihood ratio at a position is the mutant's log-probability minus the
    wildtype's. A 1-based position indexes the token row directly, since the
    tokenizer's start token sits at index 0.

    Sequences within the model window are scored in one pass. Longer ones are scored
    in overlapping windows, each mutated position taking its row from the window that
    centers it best, so the per-position score matches a whole-sequence pass would give.
    """
    import torch  # pyright: ignore[reportMissingImports]

    def window_llrs(window_sequence: str):
        tokens = tokenizer([window_sequence], return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(**tokens).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        input_ids = tokens["input_ids"][0]
        wildtype_log_probs = log_probs[range(len(input_ids)), input_ids].reshape(-1, 1)
        return log_probs - wildtype_log_probs

    positions = sorted({int(mutation[1:-1]) for variant in mutations for mutation in variant.split(":")})
    llr_row = {}
    if len(sequence) <= VESM_MODEL_WINDOW:
        llrs = window_llrs(sequence)
        for position in positions:
            llr_row[position] = llrs[position]
    else:
        windows: dict[int, list[int]] = {}
        for position in positions:
            start, _ = get_interval(position, len(sequence))
            windows.setdefault(start, []).append(position)
        for start, window_positions in windows.items():
            end = min(start + VESM_MODEL_WINDOW, len(sequence))
            llrs = window_llrs(sequence[start:end])
            for position in window_positions:
                llr_row[position] = llrs[position - start]

    vocab = tokenizer.get_vocab()
    results = []
    for variant in mutations:
        score = 0.0
        for mutation in variant.split(":"):
            position, mutant = int(mutation[1:-1]), mutation[-1]
            score += llr_row[position][vocab[mutant]].item()
        results.append({"mutation": variant, "score": score})
    return results


@app.function(
    name="vesm",
    image=vesm_image,
    gpu=VESM_GPU,
    volumes={VOLUME_ROOT: volume},
    timeout=VESM_TIMEOUT,
    max_containers=VESM_MAX_CONTAINERS,
    scaledown_window=VESM_SCALEDOWN_WINDOW,
)
def run(job_id: str, job_name: str | None, params: dict) -> None:
    """Score sequence variants with VESM and persist the output.

    predictions.json holds the per-mutation scores, and metadata.json records the
    model and sequence the scores were computed for.

    Args:
        job_id: Unique id identifying the job.
        job_name: The caller's label for the job, recorded in the run log. None
            when the caller did not supply one.
        params: A VESMParams dump, revalidated here so the container never trusts
            the payload it was handed.
    """
    logger.info(f"Starting vesm job={job_id}")
    started_at = datetime.now(UTC)
    stderr = ""

    try:
        job_params = VESMParams.model_validate(params)
        model, tokenizer = load_vesm_model(job_params.model_name)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            predictions = score_variants(model, tokenizer, job_params.sequence, job_params.mutations)
            (output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2))
            metadata = {
                "model": job_params.model_name,
                "base_model": VESM_MODELS[job_params.model_name],
                "sequence": job_params.sequence,
                "num_mutations": len(job_params.mutations),
            }
            (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
            persist_job_output(job_id, output_dir)

        summary = f"Scored {len(predictions)} mutation(s) with {job_params.model_name}"
        log = format_run_log(job_id, job_name, "vesm", VESM_SPEC, "", summary, stderr, started_at)
        mark_job_complete(job_id, log)
        logger.info(f"Done: job={job_id}")
    except Exception as e:
        logger.error(f"Failed: job={job_id}: {e}")
        log = format_run_log(job_id, job_name, "vesm", VESM_SPEC, "", str(e), stderr, started_at)
        mark_job_failed(job_id, log)
        raise
