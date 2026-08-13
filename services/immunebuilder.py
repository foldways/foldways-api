import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import modal
from pydantic import BaseModel, Field, model_validator

from common.utils import format_run_log, mark_job_complete, mark_job_failed, persist_job_output
from config import (
    IMMUNEBUILDER_GPU,
    IMMUNEBUILDER_MAX_CONTAINERS,
    IMMUNEBUILDER_SCALEDOWN_WINDOW,
    IMMUNEBUILDER_TIMEOUT,
)
from constants import (
    IMMUNEBUILDER_SPEC,
    PYDANTIC_SPEC,
    PYTHON_3_11,
    SERVICE_SOURCES,
    VOLUME_IMMUNEBUILDER_CACHE,
    VOLUME_ROOT,
)
from core import app, volume

logger = logging.getLogger(__name__)


immunebuilder_image = (
    modal.Image.micromamba(python_version=PYTHON_3_11)
    .micromamba_install("openmm", "pdbfixer", "anarci", channels=["conda-forge", "bioconda"])
    .pip_install(IMMUNEBUILDER_SPEC, "torch", PYDANTIC_SPEC)
    .add_local_python_source(*SERVICE_SOURCES)
)


class ImmuneBuilderParams(BaseModel):
    """Parameters for predicting an immune-receptor structure with ImmuneBuilder.

    ImmuneBuilder predicts the structure of an antibody, a nanobody, or a T-cell receptor
    from its chain sequences. It is a single-pass ensemble, so a job produces one
    refined structure with no sampling to tune. The refinement uses OpenMM.

    receptor selects the model and which chains are read. An antibody needs heavy and
    light, a nanobody needs heavy alone, and a TCR needs alpha and beta.
    """

    receptor: Literal["antibody", "nanobody", "tcr"] = Field(
        default="antibody", description="Which receptor to predict. Selects the model and required chains."
    )
    heavy: str | None = Field(
        default=None,
        description="Heavy chain sequence in one-letter codes. Required for antibody and nanobody.",
        examples=[
            "EVQLVESGGGVVQPGGSLRLSCAASGFTFNSYGMHWVRQAPGKGLEWVAFIRYDGGNKYYADSVKGRFTISRDNSKNTLYLQMKSLRAEDTAVYYCANLKDSRYSGSYYDYWGQGTLVTVS"
        ],
    )
    light: str | None = Field(
        default=None,
        description="Light chain sequence in one-letter codes. Required for antibody.",
        examples=[
            "VIWMTQSPSSLSASVGDRVTITCQASQDIRFYLNWYQQKPGKAPKLLISDASNMETGVPSRFSGSGSGTDFTFTISSLQPEDIATYYCQQYDNLPFTFGPGTKVDFK"
        ],
    )
    alpha: str | None = Field(
        default=None, description="TCR alpha chain sequence in one-letter codes. Required for tcr."
    )
    beta: str | None = Field(default=None, description="TCR beta chain sequence in one-letter codes. Required for tcr.")
    numbering_scheme: Literal["imgt", "chothia", "kabat", "aho", "wolfguy", "martin", "raw"] = Field(
        default="imgt", description="Residue numbering scheme applied to the output structure."
    )

    @model_validator(mode="after")
    def check_chains(self) -> "ImmuneBuilderParams":
        """Require the chains the chosen receptor needs and reject chains from other receptors."""
        required = {
            "antibody": ("heavy", "light"),
            "nanobody": ("heavy",),
            "tcr": ("alpha", "beta"),
        }[self.receptor]
        for field in ("heavy", "light", "alpha", "beta"):
            value = getattr(self, field)
            if field in required and not value:
                raise ValueError(f"receptor '{self.receptor}' requires {field}")
            if field not in required and value:
                raise ValueError(f"receptor '{self.receptor}' does not take {field}")
        return self


def predict_structure(params: ImmuneBuilderParams, output_path: Path) -> None:
    """Run the receptor's predictor on the staged weights and write the refined PDB.

    Each predictor reads its chains as a labelled dict, runs its ensemble, and refines
    the best-ranked structure with OpenMM on save.
    """
    if params.receptor == "antibody":
        from ImmuneBuilder import ABodyBuilder2  # pyright: ignore[reportMissingImports]

        predictor = ABodyBuilder2(weights_dir=VOLUME_IMMUNEBUILDER_CACHE, numbering_scheme=params.numbering_scheme)
        result = predictor.predict({"H": params.heavy, "L": params.light})
    elif params.receptor == "nanobody":
        from ImmuneBuilder import NanoBodyBuilder2  # pyright: ignore[reportMissingImports]

        predictor = NanoBodyBuilder2(weights_dir=VOLUME_IMMUNEBUILDER_CACHE, numbering_scheme=params.numbering_scheme)
        result = predictor.predict({"H": params.heavy})
    else:
        from ImmuneBuilder import TCRBuilder2  # pyright: ignore[reportMissingImports]

        predictor = TCRBuilder2(weights_dir=VOLUME_IMMUNEBUILDER_CACHE, numbering_scheme=params.numbering_scheme)
        result = predictor.predict({"A": params.alpha, "B": params.beta})

    result.save(str(output_path))


@app.function(
    name="immunebuilder",
    image=immunebuilder_image,
    gpu=IMMUNEBUILDER_GPU,
    volumes={VOLUME_ROOT: volume},
    timeout=IMMUNEBUILDER_TIMEOUT,
    max_containers=IMMUNEBUILDER_MAX_CONTAINERS,
    scaledown_window=IMMUNEBUILDER_SCALEDOWN_WINDOW,
)
def run(job_id: str, job_name: str | None, params: dict) -> None:
    """Predict an immune-receptor structure with ImmuneBuilder and persist the output.

    The output directory holds structure.pdb, the refined receptor structure.

    Args:
        job_id: Unique id identifying the job.
        job_name: The caller's label for the job, recorded in the run log. None
            when the caller did not supply one.
        params: An ImmuneBuilderParams dump, revalidated here so the container never
            trusts the payload it was handed.
    """
    logger.info(f"Starting immunebuilder prediction: job={job_id}")
    started_at = datetime.now(UTC)
    stderr = ""

    try:
        job_params = ImmuneBuilderParams.model_validate(params)
        volume.reload()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            predict_structure(job_params, output_dir / "structure.pdb")
            persist_job_output(job_id, output_dir)

        summary = f"Predicted {job_params.receptor} structure"
        log = format_run_log(job_id, job_name, "immunebuilder", IMMUNEBUILDER_SPEC, "", summary, stderr, started_at)
        mark_job_complete(job_id, log)
        logger.info(f"Done: job={job_id}")
    except Exception as e:
        logger.error(f"Failed: job={job_id}: {e}")
        log = format_run_log(job_id, job_name, "immunebuilder", IMMUNEBUILDER_SPEC, "", str(e), stderr, started_at)
        mark_job_failed(job_id, log)
        raise
