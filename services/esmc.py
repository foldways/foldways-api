import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import modal
from pydantic import BaseModel, Field, model_validator

from common.utils import format_run_log, mark_job_complete, mark_job_failed, persist_job_output
from config import ESMC_GPU, ESMC_MAX_CONTAINERS, ESMC_SCALEDOWN_WINDOW, ESMC_TIMEOUT
from constants import (
    ESMC_600M_WEIGHTS_REPO,
    ESMC_SPEC,
    ONE_LETTER_AMINO_ACIDS,
    PYDANTIC_SPEC,
    PYTHON_3_12,
    SERVICE_SOURCES,
    VOLUME_ESMC_CACHE,
    VOLUME_ROOT,
)
from core import app, volume

logger = logging.getLogger(__name__)


esmc_image = (
    modal.Image.debian_slim(python_version=PYTHON_3_12)
    .apt_install("git")
    .uv_pip_install(ESMC_SPEC, PYDANTIC_SPEC)
    .env({"HF_HUB_CACHE": VOLUME_ESMC_CACHE})
    .add_local_python_source(*SERVICE_SOURCES)
)


class ESMCParams(BaseModel):
    """Parameters for running ESMC over protein sequences.

    Two modes, selected by `mode`. `embed` runs one forward pass per sequence and
    returns representations. `variant_scores` runs masked leave-one-out inference,
    one forward pass per residue, and returns a log-likelihood-ratio matrix for
    every single substitution. Variant scoring is therefore far heavier, roughly
    the sequence length times the cost of embedding.

    Each sequence is processed independently, with one result file written per
    sequence, so a job can cover a batch of them.
    """

    sequences: list[str] = Field(
        min_length=1,
        description="Protein sequences in one-letter codes, one result file written per sequence.",
        examples=[["MKTAYIAKQR"]],
    )
    mode: Literal["embed", "variant_scores"] = Field(
        default="embed",
        description="`embed` returns representations. `variant_scores` returns per-substitution scores.",
    )
    return_logits: bool = Field(
        default=False,
        description="In embed mode, also save the per-residue sequence logits next to the embeddings.",
    )

    @model_validator(mode="after")
    def reject_empty_sequences(self):
        if any(not sequence for sequence in self.sequences):
            raise ValueError("sequences must not contain empty strings")
        return self


def load_esmc_model():
    """Load ESMC 600M from the weights staged on the volume.

    esm's ESMC_600M_202412 loads with load_torch_model, which looks for a
    safetensors checkpoint at the snapshot root. The weights repo instead ships a
    single .pth under data/weights, so this constructs the model the way that
    loader does and reads the .pth directly, as esm's own ESM3 loaders do. The
    architecture arguments and the bfloat16 cast mirror ESMC_600M_202412.
    """
    import torch  # pyright: ignore[reportMissingImports]
    from accelerate import init_empty_weights  # pyright: ignore[reportMissingImports]
    from esm.models.esmc import ESMC  # pyright: ignore[reportMissingImports]
    from esm.tokenization import get_esmc_model_tokenizers  # pyright: ignore[reportMissingImports]
    from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

    snapshot = Path(snapshot_download(repo_id=ESMC_600M_WEIGHTS_REPO))
    weights = snapshot / "data" / "weights" / "esmc_600m_2024_12_v0.pth"
    with init_empty_weights():
        model = ESMC(
            d_model=1152,
            n_heads=18,
            n_layers=36,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=False,
        ).eval()
    state_dict = torch.load(weights, map_location="cpu", weights_only=False)
    result = model.load_state_dict(state_dict, strict=False, assign=True)
    if result.missing_keys or result.unexpected_keys:
        logger.warning(f"ESMC load: {len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected keys")
    return model.to("cuda").to(torch.bfloat16)


def write_embeddings(model, sequences: list[str], return_logits: bool, output_dir: Path) -> int:
    """Embed each sequence and write embeddings_{i}.npz. Returns the embedding dim.

    Each file holds the per-residue embeddings and their mean over the sequence
    length. The mean is pooled over every token, including the special tokens the
    tokenizer adds, so a consumer wanting a trimmed mean can recompute it from the
    per-residue array.

    The model output carries a batch dimension and is bfloat16, so each array is
    indexed to drop the batch dimension and cast to float before numpy holds it.
    """
    import numpy as np  # pyright: ignore[reportMissingImports]
    from esm.sdk.api import ESMProtein, LogitsConfig  # pyright: ignore[reportMissingImports]

    config = LogitsConfig(sequence=return_logits, return_embeddings=True)
    embedding_dim = 0
    for i, sequence in enumerate(sequences):
        output = model.logits(model.encode(ESMProtein(sequence=sequence)), config)
        embeddings = output.embeddings[0].float().cpu().numpy()
        embedding_dim = embeddings.shape[-1]
        arrays = {"embeddings": embeddings, "mean_embedding": embeddings.mean(axis=0)}
        if return_logits and output.logits is not None and output.logits.sequence is not None:
            arrays["logits"] = output.logits.sequence[0].float().cpu().numpy()
        np.savez(output_dir / f"embeddings_{i}.npz", **arrays)
    return embedding_dim


def write_variant_scores(model, sequences: list[str], output_dir: Path) -> None:
    """Score every single substitution per sequence and write variant_scores_{i}.npz.

    Follows the ESMC mutation-scoring method: mask each position in turn, run the
    model, and read the predicted distribution at that position. Each file holds an
    llr matrix of shape [length, 20] over ONE_LETTER_AMINO_ACIDS, where each entry is the
    log-likelihood ratio of a substitution against the wild-type residue, so the
    wild type is 0 and negative is deleterious. It also holds a per-position entropy
    in bits, a measure of how constrained the position is.

    The tokenizer prepends a BOS token, so sequence position i reads its
    distribution from logits index i + 1.
    """
    import numpy as np  # pyright: ignore[reportMissingImports]
    import torch  # pyright: ignore[reportMissingImports]
    from esm.sdk.api import ESMProtein, LogitsConfig  # pyright: ignore[reportMissingImports]
    from esm.tokenization import get_esmc_model_tokenizers  # pyright: ignore[reportMissingImports]

    vocab = get_esmc_model_tokenizers().get_vocab()
    aa_indices = [vocab[aa] for aa in ONE_LETTER_AMINO_ACIDS]
    config = LogitsConfig(sequence=True)

    for i, sequence in enumerate(sequences):
        length = len(sequence)
        llr = np.zeros((length, len(ONE_LETTER_AMINO_ACIDS)), dtype=np.float32)
        entropy = np.zeros(length, dtype=np.float32)
        for position in range(length):
            masked = sequence[:position] + "_" + sequence[position + 1 :]
            output = model.logits(model.encode(ESMProtein(sequence=masked)), config)
            position_logits = output.logits.sequence[0, position + 1].float()
            log_probs = torch.log_softmax(position_logits, dim=-1)
            wild_type_log_prob = log_probs[vocab[sequence[position]]]
            llr[position] = (log_probs[aa_indices] - wild_type_log_prob).cpu().numpy()
            probs = torch.softmax(position_logits, dim=-1)
            entropy[position] = float(-(probs * torch.log2(probs + 1e-9)).sum())
        np.savez(output_dir / f"variant_scores_{i}.npz", llr=llr, entropy=entropy)


@app.function(
    name="esmc",
    image=esmc_image,
    gpu=ESMC_GPU,
    volumes={VOLUME_ROOT: volume},
    timeout=ESMC_TIMEOUT,
    max_containers=ESMC_MAX_CONTAINERS,
    scaledown_window=ESMC_SCALEDOWN_WINDOW,
)
def run(job_id: str, job_name: str | None, params: dict) -> None:
    """Run ESMC over one or more protein sequences and persist the result.

    The mode in the params selects embeddings or variant scores. See ESMCParams
    and the two writer helpers for what each mode produces.

    Args:
        job_id: Unique id identifying the job.
        job_name: The caller's label for the job, recorded in the run log. None
            when the caller did not supply one.
        params: An ESMCParams dump, revalidated here so the container never trusts
            the payload it was handed.
    """
    logger.info(f"Starting esmc job={job_id}")
    started_at = datetime.now(UTC)
    stderr = ""

    try:
        job_params = ESMCParams.model_validate(params)
        model = load_esmc_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            metadata = {
                "model": ESMC_600M_WEIGHTS_REPO,
                "mode": job_params.mode,
                "num_sequences": len(job_params.sequences),
                "sequence_lengths": [len(sequence) for sequence in job_params.sequences],
            }
            if job_params.mode == "embed":
                embedding_dim = write_embeddings(model, job_params.sequences, job_params.return_logits, output_dir)
                metadata["embedding_dim"] = embedding_dim
                summary = f"Embedded {len(job_params.sequences)} sequence(s), embedding dim {embedding_dim}"
            else:
                write_variant_scores(model, job_params.sequences, output_dir)
                metadata["amino_acids"] = ONE_LETTER_AMINO_ACIDS
                metadata["method"] = "masked leave-one-out log-likelihood ratios"
                summary = f"Scored variants for {len(job_params.sequences)} sequence(s)"

            (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
            persist_job_output(job_id, output_dir)

        log = format_run_log(job_id, job_name, "esmc", ESMC_SPEC, "", summary, stderr, started_at)
        mark_job_complete(job_id, log)
        logger.info(f"Done: job={job_id}")
    except Exception as e:
        logger.error(f"Failed: job={job_id}: {e}")
        log = format_run_log(job_id, job_name, "esmc", ESMC_SPEC, "", str(e), stderr, started_at)
        mark_job_failed(job_id, log)
        raise
