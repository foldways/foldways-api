import logging
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import modal
from pydantic import BaseModel, Field, model_validator

from common.utils import format_run_log, mark_job_complete, mark_job_failed, persist_job_output
from config import (
    INTELLIFOLD_GPU,
    INTELLIFOLD_MAX_CONTAINERS,
    INTELLIFOLD_SCALEDOWN_WINDOW,
    INTELLIFOLD_TIMEOUT,
)
from constants import (
    INTELLIFOLD_SPEC,
    PYDANTIC_SPEC,
    PYTHON_3_11,
    SERVICE_SOURCES,
    VOLUME_INTELLIFOLD_CACHE,
    VOLUME_ROOT,
)
from core import app, volume

logger = logging.getLogger(__name__)


intellifold_image = (
    modal.Image.debian_slim(python_version=PYTHON_3_11)
    .apt_install("build-essential")
    .pip_install(INTELLIFOLD_SPEC, PYDANTIC_SPEC)
    .run_commands("pip uninstall -y deepspeed")
    .env({"INTELLIFOLD_CACHE": VOLUME_INTELLIFOLD_CACHE})
    .add_local_python_source(*SERVICE_SOURCES)
)


CHAIN_ID_EXAMPLE = "A"
LIGAND_ID_EXAMPLE = "B"
ID_DESC = "Chain id, or a list of ids for identical copies of this entity."
CHAIN_ID_DESC = "Chain id. Must match an `id` declared in `sequences`."


class Modification(BaseModel):
    position: int = Field(ge=1, description="Residue index (1-based) of the modified residue.", examples=[1])
    ccd: str = Field(description="CCD code of the modified residue.", examples=["SEP"])


class Protein(BaseModel):
    id: str | list[str] = Field(description=ID_DESC, examples=[CHAIN_ID_EXAMPLE])
    sequence: str = Field(description="Amino-acid sequence in one-letter codes.", examples=["MKTAYIAKQR"])
    modifications: list[Modification] | None = Field(default=None, description="Modified residues in the chain.")
    cyclic: bool = Field(default=False, description="Whether the chain is a cyclic peptide.")


class Nucleic(BaseModel):  # dna or rna
    id: str | list[str] = Field(description=ID_DESC, examples=[CHAIN_ID_EXAMPLE])
    sequence: str = Field(description="Nucleotide sequence.", examples=["ATCG"])
    modifications: list[Modification] | None = Field(default=None, description="Modified residues in the chain.")
    cyclic: bool = Field(default=False, description="Whether the chain is cyclic.")


class Ligand(BaseModel):
    """One non-polymer molecule. Give exactly one of `smiles` or `ccd`."""

    id: str | list[str] = Field(description=ID_DESC, examples=[LIGAND_ID_EXAMPLE])
    smiles: str | None = Field(
        default=None,
        description="Ligand as a SMILES string (exclusive with ccd).",
        examples=["N[C@@H](Cc1ccc(O)cc1)C(=O)O"],
    )
    ccd: str | None = Field(default=None, description="Ligand as a CCD code (exclusive with smiles).", examples=["SAH"])

    @model_validator(mode="after")
    def require_smiles_or_ccd(self):
        if bool(self.smiles) == bool(self.ccd):
            raise ValueError("ligand requires exactly one of 'smiles' or 'ccd'")
        return self


class Sequence(BaseModel):
    """One chain or molecule in the complex. Set exactly one entity type."""

    protein: Protein | None = Field(default=None, description="A protein chain, as an amino-acid sequence.")
    dna: Nucleic | None = Field(default=None, description="A DNA chain, as a nucleotide sequence.")
    rna: Nucleic | None = Field(default=None, description="An RNA chain, as a nucleotide sequence.")
    ligand: Ligand | None = Field(default=None, description="A non-polymer molecule, as SMILES or a CCD code.")

    @model_validator(mode="after")
    def validate_one_entity_type(self):
        if sum(getattr(self, k) is not None for k in ("protein", "dna", "rna", "ligand")) != 1:
            raise ValueError("each sequence must set exactly one of protein/dna/rna/ligand")
        return self


class Atom(BaseModel):
    chain_id: str = Field(description=CHAIN_ID_DESC, examples=[CHAIN_ID_EXAMPLE])
    residue: int = Field(ge=1, description="Residue index, 1-based (1 for ligands).", examples=[1])
    atom_name: str = Field(description="Atom name, as in the component's CIF.", examples=["CA"])

    def as_list(self) -> list:
        """Render as the positional [chain, residue, atom] triple IntelliFold's spec wants."""
        return [self.chain_id, self.residue, self.atom_name]


class Token(BaseModel):
    chain_id: str = Field(description=CHAIN_ID_DESC, examples=[CHAIN_ID_EXAMPLE])
    residue: int = Field(ge=1, description="Residue index (1-based).", examples=[1])

    def as_list(self) -> list:
        """Render as the positional [chain, residue] pair IntelliFold's spec wants."""
        return [self.chain_id, self.residue]


class Bond(BaseModel):
    atom1: Atom = Field(description="First atom in the bond.")
    atom2: Atom = Field(description="Second atom in the bond.")


class Pocket(BaseModel):
    binder: str = Field(description="Chain id of the binder.", examples=[LIGAND_ID_EXAMPLE])
    contacts: list[Token] = Field(description="Residues forming the binding site the binder should contact.")


class Constraint(BaseModel):
    """One structural constraint. Set exactly one type."""

    bond: Bond | None = Field(default=None, description="A covalent bond between two atoms.")
    pocket: Pocket | None = Field(default=None, description="A binding site, as the residues a binder should contact.")

    @model_validator(mode="after")
    def validate_one_constraint_type(self):
        if sum(getattr(self, k) is not None for k in ("bond", "pocket")) != 1:
            raise ValueError("each constraint must set exactly one of bond/pocket")
        return self


class IntelliFoldParams(BaseModel):
    """Parameters IntelliFold needs to predict one biomolecular complex.

    `sequences` declares the entities in the complex and is the only required field.
    Each entity carries its own `id`, which names that chain, and constraints refer
    back to entities by those ids. Add `constraints` to pin covalent bonds or a
    binding pocket. The remaining fields tune the model choice and sampling.

    Multiple sequence alignments are generated automatically by the public MSA server,
    so no MSA needs to be supplied. Turn use_msa_server off to run single-sequence,
    which IntelliFold advises against since it hurts accuracy.

    Precomputed MSAs, which IntelliFold takes as on-disk paths, are not exposed. Use the
    MSA server or run single-sequence instead. Templates are not exposed either, since
    IntelliFold's template search needs a full PDB mmCIF mirror staged on the volume.
    """

    sequences: list[Sequence] = Field(
        description="One entry per entity in the complex. Each entry sets exactly one of protein, dna, rna, or ligand."
    )
    constraints: list[Constraint] | None = Field(default=None, description="Optional structural constraints.")

    model: Literal["v2-flash", "v2", "v1"] = Field(
        default="v2-flash",
        description="Model checkpoint. v2-flash is the fastest and most accurate. v2 is slower. v1 is the original.",
    )
    use_msa_server: bool = Field(
        default=True,
        description="Build MSAs from the public server, on by default for accuracy. Set false to run "
        "single-sequence and offline, which adds no network call and lowers accuracy.",
    )
    msa_server_url: str = Field(
        default="https://api.colabfold.com",
        description="MSA server used when use_msa_server is set. Point at a self-hosted server to avoid sending "
        "sequences to the public one.",
    )
    msa_pairing_strategy: Literal["greedy", "complete"] = Field(
        default="greedy", description="How to pair MSAs across chains of a multimer. Used only with the MSA server."
    )
    no_pairing: bool = Field(
        default=False, description="Skip MSA pairing for multimers. Used only with the MSA server."
    )
    seeds: list[int] = Field(default=[42], min_length=1, description="Random seeds. One prediction is run per seed.")
    recycling_iters: int = Field(
        default=10, ge=1, description="Recycling iterations. More gives higher accuracy and runs slower."
    )
    num_diffusion_samples: int = Field(default=5, ge=1, description="Structures sampled per seed.")
    sampling_steps: int = Field(
        default=200, ge=1, description="Diffusion denoising steps. More gives higher quality and runs slower."
    )
    output_format: Literal["mmcif", "pdb"] = Field(default="mmcif", description="Structure file format to write.")

    @model_validator(mode="after")
    def validate_chain_references(self):
        """Cross-check every constraint chain id against `sequences` before any GPU work starts.

        IntelliFold enforces these too, but only after a container has spun up, loaded
        the model, and built MSAs. Rejecting here turns that into an immediate 422.
        """
        entity_ids: set[str] = set()
        for sequence in self.sequences:
            for entity_type in ("protein", "dna", "rna", "ligand"):
                entity = getattr(sequence, entity_type)
                if entity is None:
                    continue
                ids = entity.id if isinstance(entity.id, list) else [entity.id]
                for chain_id in ids:
                    if chain_id in entity_ids:
                        raise ValueError(f"duplicate chain id '{chain_id}' in sequences")
                    entity_ids.add(chain_id)

        for constraint in self.constraints or []:
            if constraint.bond:
                validate_chain_reference(entity_ids, constraint.bond.atom1.chain_id, "bond atom1")
                validate_chain_reference(entity_ids, constraint.bond.atom2.chain_id, "bond atom2")
            if constraint.pocket:
                validate_chain_reference(entity_ids, constraint.pocket.binder, "pocket binder")
                for contact in constraint.pocket.contacts:
                    validate_chain_reference(entity_ids, contact.chain_id, "pocket contact")
        return self


def validate_chain_reference(entity_ids: set[str], chain_id: str, location: str) -> None:
    """Validate that a referenced chain id was declared in `sequences`.

    Args:
        entity_ids: Chain ids declared in the sequences.
        chain_id: The chain id to validate.
        location: Where the reference appears, e.g. "pocket binder", named in the error.

    Raises:
        ValueError: If the chain id was never declared.
    """
    if chain_id not in entity_ids:
        raise ValueError(f"{location} references chain '{chain_id}', which is not declared in sequences")


def render_constraint_spec(constraint: Constraint) -> dict:
    """Render one constraint into IntelliFold's spec form, where atoms and tokens are lists."""
    if constraint.bond:
        return {"bond": {"atom1": constraint.bond.atom1.as_list(), "atom2": constraint.bond.atom2.as_list()}}
    assert constraint.pocket is not None
    return {
        "pocket": {
            "binder": constraint.pocket.binder,
            "contacts": [contact.as_list() for contact in constraint.pocket.contacts],
        }
    }


def render_intellifold_spec(params: IntelliFoldParams) -> dict:
    """Render validated params into IntelliFold's YAML spec as a dict.

    When use_msa_server is off, each protein is marked `msa: empty` so IntelliFold runs
    it single-sequence rather than erroring on the missing MSA it otherwise requires.
    """
    sequences = []
    for sequence in params.sequences:
        entry = sequence.model_dump(exclude_none=True)
        if not params.use_msa_server and "protein" in entry:
            entry["protein"]["msa"] = "empty"
        sequences.append(entry)

    spec: dict = {"version": 1, "sequences": sequences}
    if params.constraints:
        spec["constraints"] = [render_constraint_spec(constraint) for constraint in params.constraints]
    return spec


@app.function(
    name="intellifold",
    image=intellifold_image,
    gpu=INTELLIFOLD_GPU,
    volumes={VOLUME_ROOT: volume},
    timeout=INTELLIFOLD_TIMEOUT,
    max_containers=INTELLIFOLD_MAX_CONTAINERS,
    scaledown_window=INTELLIFOLD_SCALEDOWN_WINDOW,
)
def run(job_id: str, job_name: str | None, params: dict) -> None:
    """Predict a complex structure with IntelliFold and persist the output.

    The output directory holds one structure per seed and diffusion sample, each with a
    summary and full confidence JSON. Structures are written as mmCIF or PDB.

    Args:
        job_id: Unique id identifying the job.
        job_name: The caller's label for the job, recorded in the run log. None
            when the caller did not supply one.
        params: An IntelliFoldParams dump, revalidated here so the container never
            trusts the payload it was handed.
    """
    import yaml

    logger.info(f"Starting intellifold prediction: job={job_id}")
    started_at = datetime.now(UTC)
    job_command = ""
    stderr = ""

    try:
        job_params = IntelliFoldParams.model_validate(params)
        intellifold_spec = render_intellifold_spec(job_params)
        volume.reload()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.yaml"
            input_path.write_text(yaml.dump(intellifold_spec, sort_keys=False))
            output_dir = tmpdir / "output"

            cmd = [
                "intellifold",
                "predict",
                str(input_path),
                "--out_dir",
                str(output_dir),
                "--cache",
                VOLUME_INTELLIFOLD_CACHE,
                "--model",
                job_params.model,
                "--seed",
                ",".join(str(seed) for seed in job_params.seeds),
                "--recycling_iters",
                str(job_params.recycling_iters),
                "--num_diffusion_samples",
                str(job_params.num_diffusion_samples),
                "--sampling_steps",
                str(job_params.sampling_steps),
                "--output_format",
                job_params.output_format,
                "--msa_server_url",
                job_params.msa_server_url,
                "--msa_pairing_strategy",
                job_params.msa_pairing_strategy,
            ]
            if job_params.use_msa_server:
                cmd.append("--use_msa_server")
            if job_params.no_pairing:
                cmd.append("--no_pairing")
            job_command = " ".join(cmd)

            logger.info(f"Running: {job_command}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            stderr = result.stderr

            if result.returncode != 0:
                logger.error(f"intellifold stderr: {stderr}")
                raise RuntimeError(f"intellifold predict exited with code {result.returncode}")

            persist_job_output(job_id, output_dir)

        log = format_run_log(
            job_id, job_name, "intellifold", INTELLIFOLD_SPEC, job_command, result.stdout, stderr, started_at
        )
        mark_job_complete(job_id, log)
        logger.info(f"Done: job={job_id}")
    except Exception as e:
        logger.error(f"Failed: job={job_id}: {e}")
        log = format_run_log(job_id, job_name, "intellifold", INTELLIFOLD_SPEC, job_command, str(e), stderr, started_at)
        mark_job_failed(job_id, log)
        raise
