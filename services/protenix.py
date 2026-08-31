import json
import logging
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import modal
from pydantic import BaseModel, Field, model_validator

from common.utils import format_run_log, mark_job_complete, mark_job_failed, persist_job_output
from config import (
    PROTENIX_GPU,
    PROTENIX_MAX_CONTAINERS,
    PROTENIX_SCALEDOWN_WINDOW,
    PROTENIX_TIMEOUT,
)
from constants import (
    PROTENIX_COLABFOLD_MSA_URL,
    PROTENIX_CUDA_ARCH,
    PROTENIX_CUDA_IMAGE,
    PROTENIX_MSA_SERVER_ENV,
    PROTENIX_ROOT_ENV,
    PROTENIX_SPEC,
    PYDANTIC_SPEC,
    PYTHON_3_11,
    SERVICE_SOURCES,
    VOLUME_PROTENIX_CACHE,
    VOLUME_ROOT,
)
from core import app, volume

logger = logging.getLogger(__name__)


protenix_image = (
    # Protenix JIT-compiles a fused LayerNorm CUDA kernel on import, so the image needs nvcc from a
    # CUDA devel base. CUDA_HOME and the arch list let the compile run, and the kernel is built once
    # at image build so containers start warm and never fail the compile at run time.
    modal.Image.from_registry(PROTENIX_CUDA_IMAGE, add_python=PYTHON_3_11)
    .apt_install("build-essential")
    .pip_install(PROTENIX_SPEC, PYDANTIC_SPEC)
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TORCH_CUDA_ARCH_LIST": PROTENIX_CUDA_ARCH,
            PROTENIX_ROOT_ENV: VOLUME_PROTENIX_CACHE,
        }
    )
    .run_commands("python -c 'from protenix.model.layer_norm.layer_norm import FusedLayerNorm'")
    .add_local_python_source(*SERVICE_SOURCES)
)


CHAIN_ID_EXAMPLE = "A"
LIGAND_ID_EXAMPLE = "B"
ID_DESC = "Chain id, or a list of ids for identical copies of this entity."
CHAIN_ID_DESC = "Chain id. Must match an `id` declared in `sequences`."
DEFAULT_MODEL = "protenix_base_default_v1.0.0"

# The job name Protenix stamps onto the output directory and file names.
JOB_NAME = "prediction"


class Modification(BaseModel):
    position: int = Field(ge=1, description="Residue index (1-based) of the modified residue.", examples=[1])
    ccd: str = Field(description="CCD code of the modified residue.", examples=["SEP"])


class Protein(BaseModel):
    id: str | list[str] = Field(description=ID_DESC, examples=[CHAIN_ID_EXAMPLE])
    sequence: str = Field(description="Amino-acid sequence in one-letter codes.", examples=["MKTAYIAKQR"])
    modifications: list[Modification] | None = Field(default=None, description="Modified residues in the chain.")


class Nucleic(BaseModel):  # dna or rna
    id: str | list[str] = Field(description=ID_DESC, examples=[CHAIN_ID_EXAMPLE])
    sequence: str = Field(description="Nucleotide sequence.", examples=["ATCG"])
    modifications: list[Modification] | None = Field(default=None, description="Modified residues in the chain.")


class Ligand(BaseModel):
    """One non-polymer molecule. Give exactly one of `smiles` or `ccd`."""

    id: str | list[str] = Field(description=ID_DESC, examples=[LIGAND_ID_EXAMPLE])
    smiles: str | None = Field(
        default=None,
        description="Ligand as a SMILES string (exclusive with ccd).",
        examples=["N[C@@H](Cc1ccc(O)cc1)C(=O)O"],
    )
    ccd: str | None = Field(
        default=None,
        description="Ligand as a CCD code, or underscore-joined codes for a glycan, e.g. NAG_BMA (exclusive with smiles).",
        examples=["ATP"],
    )

    @model_validator(mode="after")
    def require_smiles_or_ccd(self):
        if bool(self.smiles) == bool(self.ccd):
            raise ValueError("ligand requires exactly one of 'smiles' or 'ccd'")
        return self


class Ion(BaseModel):
    id: str | list[str] = Field(description=ID_DESC, examples=[LIGAND_ID_EXAMPLE])
    ccd: str = Field(description="Ion as a CCD code.", examples=["MG"])


class Sequence(BaseModel):
    """One chain or molecule in the complex. Set exactly one entity type."""

    protein: Protein | None = Field(default=None, description="A protein chain, as an amino-acid sequence.")
    dna: Nucleic | None = Field(default=None, description="A DNA chain, as a nucleotide sequence.")
    rna: Nucleic | None = Field(default=None, description="An RNA chain, as a nucleotide sequence.")
    ligand: Ligand | None = Field(default=None, description="A non-polymer molecule, as SMILES or a CCD code.")
    ion: Ion | None = Field(default=None, description="An ion, as a CCD code.")

    @model_validator(mode="after")
    def validate_one_entity_type(self):
        if sum(getattr(self, k) is not None for k in ("protein", "dna", "rna", "ligand", "ion")) != 1:
            raise ValueError("each sequence must set exactly one of protein/dna/rna/ligand/ion")
        return self

    def entity(self) -> tuple[str, Protein | Nucleic | Ligand | Ion]:
        """Return the entity's type name and the model set on this sequence."""
        for entity_type in ("protein", "dna", "rna", "ligand", "ion"):
            entity = getattr(self, entity_type)
            if entity is not None:
                return entity_type, entity
        raise ValueError("sequence has no entity set")  # unreachable, guarded by validate_one_entity_type


class Atom(BaseModel):
    chain_id: str = Field(description=CHAIN_ID_DESC, examples=[CHAIN_ID_EXAMPLE])
    residue: int = Field(ge=1, description="Residue index, 1-based (1 for ligands).", examples=[1])
    atom_name: str = Field(description="Atom name, as in the component's CIF.", examples=["CA"])


class Token(BaseModel):
    chain_id: str = Field(description=CHAIN_ID_DESC, examples=[CHAIN_ID_EXAMPLE])
    residue: int = Field(ge=1, description="Residue index (1-based).", examples=[1])
    atom_name: str | None = Field(
        default=None,
        description="Atom name for an atom-level reference. Omit to reference the residue's central atom.",
        examples=["CA"],
    )


class Bond(BaseModel):
    """A covalent bond between a polymer and a ligand, or between two ligands."""

    atom1: Atom = Field(description="First atom in the bond.")
    atom2: Atom = Field(description="Second atom in the bond.")


class Contact(BaseModel):
    """A soft distance constraint between two residues or atoms."""

    token1: Token = Field(description="First residue or atom in the contact.")
    token2: Token = Field(description="Second residue or atom in the contact.")
    max_distance: float = Field(default=6.0, gt=0, description="Expected maximum distance in Angstroms.")
    min_distance: float = Field(default=0.0, ge=0, description="Expected minimum distance in Angstroms.")


class Pocket(BaseModel):
    """A soft binding-site constraint between a binder chain and contact residues."""

    binder: str = Field(description="Chain id of the binder.", examples=[LIGAND_ID_EXAMPLE])
    contacts: list[Token] = Field(description="Residues forming the binding site the binder should contact.")
    max_distance: float = Field(default=6.0, gt=0, description="Maximum distance in Angstroms.")


class Constraint(BaseModel):
    """One structural constraint. Set exactly one type."""

    contact: Contact | None = Field(default=None, description="A distance constraint between two residues or atoms.")
    pocket: Pocket | None = Field(default=None, description="A binding site, as the residues a binder should contact.")

    @model_validator(mode="after")
    def validate_one_constraint_type(self):
        if sum(getattr(self, k) is not None for k in ("contact", "pocket")) != 1:
            raise ValueError("each constraint must set exactly one of contact/pocket")
        return self


class ProtenixParams(BaseModel):
    """Parameters Protenix needs to predict one biomolecular complex.

    `sequences` declares the entities in the complex and is the only required field.
    Each entity carries its own `id`, which names that chain, and constraints and bonds
    refer back to entities by those ids. Add `covalent_bonds` to pin covalent links and
    `constraints` to add a soft contact or binding pocket. The remaining fields tune the
    model choice and sampling.

    Multiple sequence alignments are generated automatically by Protenix's public MSA
    server, so no MSA needs to be supplied. Turn use_msa off to run single-sequence, which
    lowers accuracy. Precomputed MSAs, templates, and RNA MSAs, which Protenix takes as
    on-disk paths, are not exposed.
    """

    sequences: list[Sequence] = Field(
        description="One entry per entity in the complex. Each entry sets exactly one of protein, dna, rna, ligand, or ion."
    )
    covalent_bonds: list[Bond] | None = Field(default=None, description="Optional covalent bonds.")
    constraints: list[Constraint] | None = Field(default=None, description="Optional soft structural constraints.")

    model: Literal["protenix_base_default_v1.0.0", "protenix_mini_default_v0.5.0"] = Field(
        default=DEFAULT_MODEL,
        description="Model checkpoint. The base model is the most accurate, mini is faster and lighter.",
    )
    use_default_params: bool = Field(
        default=False,
        description="Override recycling_cycles and sampling_steps with the chosen model's recommended defaults. "
        "Recommended for the mini model, which is tuned for fewer cycles and steps.",
    )
    seeds: list[int] = Field(default=[101], min_length=1, description="Random seeds. One prediction is run per seed.")
    recycling_cycles: int = Field(
        default=10, ge=1, description="Pairformer recycling cycles. More gives higher accuracy and runs slower."
    )
    sampling_steps: int = Field(
        default=200, ge=1, description="Diffusion denoising steps. More gives higher quality and runs slower."
    )
    num_samples: int = Field(default=5, ge=1, description="Structures sampled per seed.")
    dtype: Literal["bf16", "fp32", "fp16"] = Field(default="bf16", description="Inference precision.")
    use_msa: bool = Field(
        default=True,
        description="Build MSAs from the public server, on by default for accuracy. Set false to run single-sequence, "
        "which adds no network call and lowers accuracy.",
    )
    msa_server_mode: Literal["protenix", "colabfold"] = Field(
        default="protenix",
        description="MSA search backend, used only when use_msa is set. protenix uses Protenix's own "
        "server, colabfold uses the public ColabFold MMseqs2 server.",
    )
    need_atom_confidence: bool = Field(
        default=False, description="Also compute and write atom-level confidence scores."
    )

    @model_validator(mode="after")
    def validate_chain_references(self):
        """Cross-check every bond and constraint chain id against `sequences` before any GPU work starts.

        Protenix enforces these too, but only after a container has spun up, loaded the
        model, and built MSAs. Rejecting here turns that into an immediate 422.
        """
        chain_map = resolve_chain_map(self.sequences)

        for bond in self.covalent_bonds or []:
            validate_chain_reference(chain_map, bond.atom1.chain_id, "bond atom1")
            validate_chain_reference(chain_map, bond.atom2.chain_id, "bond atom2")

        for constraint in self.constraints or []:
            if constraint.contact:
                validate_chain_reference(chain_map, constraint.contact.token1.chain_id, "contact token1")
                validate_chain_reference(chain_map, constraint.contact.token2.chain_id, "contact token2")
            if constraint.pocket:
                validate_chain_reference(chain_map, constraint.pocket.binder, "pocket binder")
                for contact in constraint.pocket.contacts:
                    validate_chain_reference(chain_map, contact.chain_id, "pocket contact")
        return self


def resolve_chain_map(sequences: list[Sequence]) -> dict[str, tuple[int, int]]:
    """Map each chain id to Protenix's (entity number, copy index) addressing.

    Protenix references chains by their 1-based position in `sequences` (the entity number)
    and a 1-based copy index within that entity. This API references chains by id instead,
    matching the other structure-prediction services, so this builds the translation and
    rejects duplicate ids along the way.
    """
    chain_map: dict[str, tuple[int, int]] = {}
    for entity_number, sequence in enumerate(sequences, start=1):
        _, entity = sequence.entity()
        ids = entity.id if isinstance(entity.id, list) else [entity.id]
        for copy_index, chain_id in enumerate(ids, start=1):
            if chain_id in chain_map:
                raise ValueError(f"duplicate chain id '{chain_id}' in sequences")
            chain_map[chain_id] = (entity_number, copy_index)
    return chain_map


def validate_chain_reference(chain_map: dict[str, tuple[int, int]], chain_id: str, location: str) -> None:
    """Validate that a referenced chain id was declared in `sequences`.

    Args:
        chain_map: Chain ids declared in the sequences, mapped to their Protenix addressing.
        chain_id: The chain id to validate.
        location: Where the reference appears, e.g. "pocket binder", named in the error.

    Raises:
        ValueError: If the chain id was never declared.
    """
    if chain_id not in chain_map:
        raise ValueError(f"{location} references chain '{chain_id}', which is not declared in sequences")


def render_entity(entity_type: str, entity: Protein | Nucleic | Ligand | Ion) -> dict:
    """Render one entity into the object Protenix's `sequences` list expects.

    Protenix keys each entity by a type name and takes a per-entity `count` with an
    explicit `id` list, so identical copies declared as a list of ids become one entity
    of that count. Modifications carry different field names for polymers versus nucleic
    acids, and a CCD ligand is prefixed with `CCD_` while an ion is a bare CCD code.
    """
    ids = entity.id if isinstance(entity.id, list) else [entity.id]
    spec: dict = {"count": len(ids), "id": ids}

    if entity_type == "protein":
        assert isinstance(entity, Protein)
        spec["sequence"] = entity.sequence
        spec["modifications"] = [{"ptmType": m.ccd, "ptmPosition": m.position} for m in entity.modifications or []]
        return {"proteinChain": spec}
    if entity_type in ("dna", "rna"):
        assert isinstance(entity, Nucleic)
        spec["sequence"] = entity.sequence
        spec["modifications"] = [
            {"modificationType": m.ccd, "basePosition": m.position} for m in entity.modifications or []
        ]
        return {"dnaSequence" if entity_type == "dna" else "rnaSequence": spec}
    if entity_type == "ligand":
        assert isinstance(entity, Ligand)
        spec["ligand"] = entity.smiles if entity.smiles else f"CCD_{entity.ccd}"
        return {"ligand": spec}
    assert isinstance(entity, Ion)
    spec["ion"] = entity.ccd
    return {"ion": spec}


def render_bond(bond: Bond, chain_map: dict[str, tuple[int, int]]) -> dict:
    """Render one covalent bond into Protenix's entity-and-copy addressed form."""
    entity1, copy1 = chain_map[bond.atom1.chain_id]
    entity2, copy2 = chain_map[bond.atom2.chain_id]
    return {
        "entity1": str(entity1),
        "copy1": copy1,
        "position1": str(bond.atom1.residue),
        "atom1": bond.atom1.atom_name,
        "entity2": str(entity2),
        "copy2": copy2,
        "position2": str(bond.atom2.residue),
        "atom2": bond.atom2.atom_name,
    }


def render_contact(contact: Contact, chain_map: dict[str, tuple[int, int]]) -> dict:
    """Render one contact constraint, addressing each end by entity, copy, and position."""
    spec: dict = {}
    for suffix, token in (("1", contact.token1), ("2", contact.token2)):
        entity, copy = chain_map[token.chain_id]
        spec[f"entity{suffix}"] = entity
        spec[f"copy{suffix}"] = copy
        spec[f"position{suffix}"] = token.residue
        if token.atom_name is not None:
            spec[f"atom{suffix}"] = token.atom_name
    spec["max_distance"] = contact.max_distance
    spec["min_distance"] = contact.min_distance
    return spec


def render_pocket(pocket: Pocket, chain_map: dict[str, tuple[int, int]]) -> dict:
    """Render one pocket constraint into Protenix's binder-and-contacts form."""
    binder_entity, binder_copy = chain_map[pocket.binder]
    contact_residues = []
    for contact in pocket.contacts:
        entity, copy = chain_map[contact.chain_id]
        contact_residues.append({"entity": entity, "copy": copy, "position": contact.residue})
    return {
        "binder_chain": {"entity": binder_entity, "copy": binder_copy},
        "contact_residues": contact_residues,
        "max_distance": pocket.max_distance,
    }


def render_protenix_spec(params: ProtenixParams) -> list[dict]:
    """Render validated params into Protenix's input JSON, a list holding one job dict.

    Contact constraints collect into a `contact` list and a single pocket into `pocket`,
    matching Protenix's one-constraint-object-per-job layout.
    """
    chain_map = resolve_chain_map(params.sequences)
    job: dict = {
        "name": JOB_NAME,
        "sequences": [render_entity(*sequence.entity()) for sequence in params.sequences],
    }

    if params.covalent_bonds:
        job["covalent_bonds"] = [render_bond(bond, chain_map) for bond in params.covalent_bonds]

    constraint: dict = {}
    contacts = [render_contact(c.contact, chain_map) for c in params.constraints or [] if c.contact]
    if contacts:
        constraint["contact"] = contacts
    pockets = [render_pocket(c.pocket, chain_map) for c in params.constraints or [] if c.pocket]
    if pockets:
        constraint["pocket"] = pockets[0]
    if constraint:
        job["constraint"] = constraint

    return [job]


@app.function(
    name="protenix",
    image=protenix_image,
    gpu=PROTENIX_GPU,
    volumes={VOLUME_ROOT: volume},
    timeout=PROTENIX_TIMEOUT,
    max_containers=PROTENIX_MAX_CONTAINERS,
    scaledown_window=PROTENIX_SCALEDOWN_WINDOW,
)
def run(job_id: str, job_name: str | None, params: dict) -> None:
    """Predict a complex structure with Protenix and persist the output.

    The output directory holds one structure per seed and diffusion sample, each with a
    summary confidence JSON, plus a full per-atom confidence JSON when
    need_atom_confidence is set. Structures are written as mmCIF.

    Args:
        job_id: Unique id identifying the job.
        job_name: The caller's label for the job, recorded in the run log. None
            when the caller did not supply one.
        params: A ProtenixParams dump, revalidated here so the container never
            trusts the payload it was handed.
    """
    logger.info(f"Starting protenix prediction: job={job_id}")
    started_at = datetime.now(UTC)
    job_command = ""
    stderr = ""

    try:
        job_params = ProtenixParams.model_validate(params)
        protenix_spec = render_protenix_spec(job_params)
        volume.reload()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.json"
            input_path.write_text(json.dumps(protenix_spec, indent=2))
            output_dir = tmpdir / "output"

            cmd = [
                "protenix",
                "pred",
                "-i",
                str(input_path),
                "-o",
                str(output_dir),
                "-n",
                job_params.model,
                "-s",
                ",".join(str(seed) for seed in job_params.seeds),
                "-c",
                str(job_params.recycling_cycles),
                "-p",
                str(job_params.sampling_steps),
                "-e",
                str(job_params.num_samples),
                "-d",
                job_params.dtype,
                "--use_msa",
                str(job_params.use_msa),
                "--use_default_params",
                str(job_params.use_default_params),
                "--msa_server_mode",
                job_params.msa_server_mode,
                "--need_atom_confidence",
                str(job_params.need_atom_confidence),
            ]
            job_command = " ".join(cmd)

            # colabfold mode is routed to the public ColabFold server. Without this it would
            # hit Protenix's own server, which protenix mode leaves as the default.
            run_env = os.environ.copy()
            if job_params.use_msa and job_params.msa_server_mode == "colabfold":
                run_env[PROTENIX_MSA_SERVER_ENV] = PROTENIX_COLABFOLD_MSA_URL

            logger.info(f"Running: {job_command}")
            result = subprocess.run(cmd, capture_output=True, text=True, env=run_env)
            stderr = result.stderr

            if result.returncode != 0:
                logger.error(f"protenix stderr: {stderr}")
                raise RuntimeError(f"protenix pred exited with code {result.returncode}")

            persist_job_output(job_id, output_dir)

        log = format_run_log(
            job_id, job_name, "protenix", PROTENIX_SPEC, job_command, result.stdout, stderr, started_at
        )
        mark_job_complete(job_id, log)
        logger.info(f"Done: job={job_id}")
    except Exception as e:
        logger.error(f"Failed: job={job_id}: {e}")
        log = format_run_log(job_id, job_name, "protenix", PROTENIX_SPEC, job_command, str(e), stderr, started_at)
        mark_job_failed(job_id, log)
        raise
