import logging
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import modal
from pydantic import BaseModel, Field, model_validator

from common.utils import format_run_log, mark_job_complete, mark_job_failed, persist_job_output
from config import (
    BOLTZ2_GPU,
    BOLTZ2_MAX_CONTAINERS,
    BOLTZ2_SCALEDOWN_WINDOW,
    BOLTZ2_TIMEOUT,
)
from constants import (
    BOLTZ2_SPEC,
    PYDANTIC_SPEC,
    PYTHON_3_12,
    SERVICE_SOURCES,
    VOLUME_BOLTZ2_CACHE,
    VOLUME_ROOT,
)
from core import app, volume

logger = logging.getLogger(__name__)


boltz2_image = (
    modal.Image.debian_slim(python_version=PYTHON_3_12)
    .uv_pip_install(BOLTZ2_SPEC, "pyyaml", PYDANTIC_SPEC)
    .env({"BOLTZ_CACHE": VOLUME_BOLTZ2_CACHE})  # the name boltz reads, and --cache is passed too
    .add_local_python_source(*SERVICE_SOURCES)
)


CHAIN_ID_EXAMPLE = "A"
LIGAND_ID_EXAMPLE = "C"
ID_DESC = "Chain id, or a list of ids for identical copies of this entity."
CHAIN_ID_DESC = "Chain id. Must match an `id` declared in `sequences`."
FORCE_DESC = "If true, enforce with an inference-time potential."
MAX_DISTANCE_DESC = "Maximum distance in Angstroms (supported range 4-20 Å)."


# sequences parameter types
class Modification(BaseModel):
    position: int = Field(ge=1, description="Residue index (1-based) of the modified residue.", examples=[1])
    ccd: str = Field(description="CCD code of the modified residue.", examples=["SEP"])


class Protein(BaseModel):
    id: str | list[str] = Field(description=ID_DESC, examples=[CHAIN_ID_EXAMPLE])
    sequence: str = Field(description="Amino-acid sequence in one-letter codes.", examples=["MKTAYIAKQR"])
    modifications: list[Modification] | None = Field(default=None, description="Modified residues in the chain.")
    cyclic: bool = Field(default=False, description="Whether the chain is cyclic.")


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
        """Render as the positional [chain, residue, atom] triple boltz's spec wants."""
        return [self.chain_id, self.residue, self.atom_name]


class Token(BaseModel):
    chain_id: str = Field(description=CHAIN_ID_DESC, examples=[CHAIN_ID_EXAMPLE])
    residue: int | str = Field(description="Residue index (1-based), or atom name for ligand chains.", examples=[10])

    def as_list(self) -> list:
        """Render as the positional [chain, residue] pair boltz's spec wants."""
        return [self.chain_id, self.residue]


# constraints parameter types
class Bond(BaseModel):
    atom1: Atom = Field(description="First atom in the bond.")
    atom2: Atom = Field(description="Second atom in the bond.")


class Pocket(BaseModel):
    binder: str = Field(
        description="Chain id of the binder (the chain binding the pocket).",
        examples=[CHAIN_ID_EXAMPLE],
    )
    contacts: list[Token] = Field(description="Residues or atoms forming the binding site.")
    max_distance: float = Field(default=6.0, ge=4.0, le=20.0, description=MAX_DISTANCE_DESC)
    force: bool = Field(default=False, description=FORCE_DESC)


class Contact(BaseModel):
    token1: Token = Field(description="First residue or atom in the contact.")
    token2: Token = Field(description="Second residue or atom in the contact.")
    max_distance: float = Field(default=6.0, ge=4.0, le=20.0, description=MAX_DISTANCE_DESC)
    force: bool = Field(default=False, description=FORCE_DESC)


class Constraint(BaseModel):
    """One structural constraint. Set exactly one type."""

    bond: Bond | None = Field(
        default=None,
        description="A covalent bond between two atoms. Supported only for CCD ligands and canonical residues.",
    )
    pocket: Pocket | None = Field(
        default=None, description="A binding site, as the residues or atoms a binder should contact."
    )
    contact: Contact | None = Field(default=None, description="A contact between two residues or atoms.")

    @model_validator(mode="after")
    def validate_one_constraint_type(self):
        if sum(getattr(self, k) is not None for k in ("bond", "pocket", "contact")) != 1:
            raise ValueError("each constraint must set exactly one of bond/pocket/contact")
        return self


# templates parameter types
class Template(BaseModel):
    """A structural template to guide the prediction of one or more chains.

    boltz2 reads templates from disk, but this API takes them over HTTP, so the
    file's text is submitted inline in `cif` or `pdb` rather than as a path.

    Give exactly one of `cif` or `pdb`. Without `chain_id`, boltz2 picks the best
    matching chains itself. Naming `chain_id` restricts it to those chains, and
    adding `template_id` maps them onto specific chains within the template file.
    """

    cif: str | None = Field(default=None, description="Inline mmCIF content (the file's text, not a path).")
    pdb: str | None = Field(default=None, description="Inline PDB content (the file's text, not a path).")
    chain_id: str | list[str] | None = Field(
        default=None, description="Chain(s) in `sequences` to apply the template to."
    )
    template_id: str | list[str] | None = Field(default=None, description="Template chain id(s) to map onto chain_id.")
    force: bool = Field(
        default=False,
        description="If true, keep the backbone near the template via a potential (needs threshold).",
    )
    threshold: float | None = Field(
        default=None,
        gt=0,
        description="Max allowed deviation from the template, in Angstroms.",
        examples=[5.0],
    )

    @model_validator(mode="after")
    def validate_format_and_force(self):
        if bool(self.cif) == bool(self.pdb):
            raise ValueError("template requires exactly one of 'cif' or 'pdb'")
        if self.force and self.threshold is None:
            raise ValueError("template 'force' requires a 'threshold'")
        return self


# properties parameter types
class Affinity(BaseModel):
    binder: str = Field(
        description="Chain id of the ligand to score for binding affinity. Must match a ligand `id` in `sequences`.",
        examples=[LIGAND_ID_EXAMPLE],
    )


class Property(BaseModel):
    """A property to compute. Only affinity is currently supported."""

    affinity: Affinity | None = Field(
        default=None, description="Predict binding affinity for one ligand chain in the complex."
    )

    @model_validator(mode="after")
    def validate_affinity(self):
        if self.affinity is None:
            raise ValueError("each property must set 'affinity'")
        return self


class Boltz2Params(BaseModel):
    """Parameters Boltz-2 needs to predict one biomolecular complex.

    `sequences` declares the entities in the complex and is the only required
    field. Each entity carries its own `id`, which names that chain, and every
    other section refers back to entities by those ids. Add
    `constraints` to pin bonds, pockets, or contacts, `templates` to steer chains
    toward a known structure, and `properties` to score a ligand's binding affinity.
    The remaining fields tune sampling and which extra outputs are written.

    Multiple sequence alignments are generated automatically by the public MSA
    server, so no MSA needs to be supplied.
    """

    sequences: list[Sequence] = Field(
        description=(
            "One entry per entity in the complex. Each entry sets exactly one of "
            "protein, dna, rna, or ligand, so a polymer chain and a small molecule "
            "are both declared here."
        )
    )
    constraints: list[Constraint] | None = Field(default=None, description="Optional structural constraints.")
    templates: list[Template] | None = Field(default=None, description="Optional structural templates.")
    properties: list[Property] | None = Field(
        default=None, description="Optional properties to compute (e.g. affinity)."
    )
    diffusion_samples: int = Field(default=1, ge=1, le=25, description="Number of structures to generate.")
    recycling_steps: int = Field(
        default=3, ge=0, le=10, description="Recycling iterations. More gives higher accuracy and runs slower."
    )
    sampling_steps: int = Field(default=200, ge=1, le=500, description="Diffusion denoising steps.")
    step_scale: float = Field(
        default=1.5, ge=1.0, le=2.0, description="Diffusion temperature. Lower gives more diversity among samples."
    )
    use_potentials: bool = Field(
        default=False, description="Apply inference-time potentials for physical plausibility."
    )
    write_full_pae: bool = Field(default=False, description="Save the full PAE matrix alongside the structures.")
    write_full_pde: bool = Field(default=False, description="Save the full PDE matrix alongside the structures.")
    no_kernels: bool = Field(
        default=False,
        description="Disable trifast kernels for triangular updates, which older GPUs need.",
    )
    affinity_mw_correction: bool = Field(
        default=False, description="Apply the molecular-weight correction to the affinity value."
    )
    sampling_steps_affinity: int = Field(default=200, ge=1, le=500, description="Denoising steps for affinity.")
    diffusion_samples_affinity: int = Field(default=5, ge=1, le=25, description="Diffusion samples for affinity.")

    @model_validator(mode="after")
    def validate_chain_references(self):
        """Cross-check every chain id against `sequences` before any GPU work starts.

        boltz2 enforces these too, but only after a container has spun up, loaded the
        model, and built MSAs. Rejecting here turns that into an immediate 422.
        """
        entity_types: dict[str, str] = {}
        multi_copy: set[str] = set()
        for sequence in self.sequences:
            for entity_type in ("protein", "dna", "rna", "ligand"):
                entity = getattr(sequence, entity_type)
                if entity is None:
                    continue
                ids = entity.id if isinstance(entity.id, list) else [entity.id]
                for chain_id in ids:
                    if chain_id in entity_types:
                        raise ValueError(f"duplicate chain id '{chain_id}' in sequences")
                    entity_types[chain_id] = entity_type
                if len(ids) > 1:
                    multi_copy.update(ids)

        for constraint in self.constraints or []:
            if constraint.bond:
                validate_chain_reference(entity_types, constraint.bond.atom1.chain_id, "bond atom1")
                validate_chain_reference(entity_types, constraint.bond.atom2.chain_id, "bond atom2")
            if constraint.pocket:
                validate_chain_reference(entity_types, constraint.pocket.binder, "pocket binder")
                for contact in constraint.pocket.contacts:
                    validate_chain_reference(entity_types, contact.chain_id, "pocket contact")
            if constraint.contact:
                validate_chain_reference(entity_types, constraint.contact.token1.chain_id, "contact token1")
                validate_chain_reference(entity_types, constraint.contact.token2.chain_id, "contact token2")

        for template in self.templates or []:
            chain_ids = template.chain_id if isinstance(template.chain_id, list) else [template.chain_id]
            for chain_id in chain_ids:
                if chain_id is not None:
                    validate_chain_reference(entity_types, chain_id, "template chain_id")

        binders = [p.affinity.binder for p in self.properties or [] if p.affinity]
        if len(binders) > 1:
            raise ValueError("only one affinity property is supported")
        for binder in binders:
            validate_chain_reference(entity_types, binder, "affinity binder")
            if entity_types[binder] != "ligand":
                raise ValueError(f"affinity binder '{binder}' must be a ligand chain")
            if binder in multi_copy:
                raise ValueError(f"cannot compute affinity for ligand '{binder}', which has multiple copies")
        return self


def validate_chain_reference(entity_types: dict[str, str], chain_id: str, location: str) -> None:
    """Validate that a referenced chain id was declared in `sequences`.

    Args:
        entity_types: Declared chain ids mapped to their entity type.
        chain_id: The chain id to validate.
        location: Where the reference appears, e.g. "pocket binder", named in the error.

    Raises:
        ValueError: If the chain id was never declared.
    """
    if chain_id not in entity_types:
        raise ValueError(f"{location} references chain '{chain_id}', which is not declared in sequences")


def render_constraint_spec(constraint: Constraint) -> dict:
    """Render one constraint into boltz's spec form, where atoms and tokens are lists.

    Args:
        constraint: The constraint to render.

    Returns:
        The constraint as boltz's YAML expects it.
    """
    spec = constraint.model_dump(exclude_none=True)
    if constraint.bond:
        spec["bond"]["atom1"] = constraint.bond.atom1.as_list()
        spec["bond"]["atom2"] = constraint.bond.atom2.as_list()
    elif constraint.pocket:
        spec["pocket"]["contacts"] = [contact.as_list() for contact in constraint.pocket.contacts]
    elif constraint.contact:
        spec["contact"]["token1"] = constraint.contact.token1.as_list()
        spec["contact"]["token2"] = constraint.contact.token2.as_list()
    return spec


def render_boltz2_spec(params: Boltz2Params) -> dict:
    """Render validated params into boltz's YAML spec as a dict.

    Templates are omitted here and added by run(), which writes their inline
    content to files and injects the resulting paths.

    Args:
        params: The validated job params.

    Returns:
        The spec dict, ready to be written as boltz's input YAML.
    """
    spec: dict = {
        "version": 1,
        "sequences": [sequence.model_dump(exclude_none=True) for sequence in params.sequences],
    }
    if params.constraints:
        spec["constraints"] = [render_constraint_spec(constraint) for constraint in params.constraints]
    if params.properties:
        spec["properties"] = [prop.model_dump(exclude_none=True) for prop in params.properties]
    return spec


@app.function(
    name="boltz2",
    image=boltz2_image,
    gpu=BOLTZ2_GPU,
    volumes={VOLUME_ROOT: volume},
    timeout=BOLTZ2_TIMEOUT,
    max_containers=BOLTZ2_MAX_CONTAINERS,
    scaledown_window=BOLTZ2_SCALEDOWN_WINDOW,
)
def run(job_id: str, job_name: str | None, params: dict) -> None:
    """Predict one complex with Boltz-2 and persist the result.

    Any templates are materialized first. boltz's `templates` section expects
    on-disk cif or pdb paths, so each template's inline content is written to a
    file in a temp dir and the spec is pointed at that path.

    Args:
        job_id: Unique name identifying the job.
        job_name: The caller's label for the job, recorded in the run log. None
            when the caller did not supply one.
        params: A Boltz2Params dump, revalidated here so the container never
            trusts the payload it was handed.
    """
    import yaml

    logger.info(f"Starting boltz2 prediction: job={job_id}")
    started_at = datetime.now(UTC)
    job_command = ""
    stderr = ""

    try:
        job_params = Boltz2Params.model_validate(params)
        boltz_spec = render_boltz2_spec(job_params)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.yaml"
            output_dir = tmpdir / "output"

            for i, template in enumerate(job_params.templates or []):
                fmt = "cif" if template.cif else "pdb"
                template_path = tmpdir / f"template_{i}.{fmt}"
                template_path.write_text(template.cif or template.pdb or "")
                entry: dict[str, object] = {fmt: str(template_path)}
                for key in ("chain_id", "template_id", "threshold"):
                    if getattr(template, key) is not None:
                        entry[key] = getattr(template, key)
                if template.force:
                    entry["force"] = True
                boltz_spec.setdefault("templates", []).append(entry)

            input_path.write_text(yaml.dump(boltz_spec, sort_keys=False))

            cmd = [
                "boltz",
                "predict",
                str(input_path),
                "--out_dir",
                str(output_dir),
                "--cache",
                VOLUME_BOLTZ2_CACHE,
                "--output_format",
                "mmcif",
                "--use_msa_server",
                "--diffusion_samples",
                str(job_params.diffusion_samples),
                "--recycling_steps",
                str(job_params.recycling_steps),
                "--sampling_steps",
                str(job_params.sampling_steps),
                "--step_scale",
                str(job_params.step_scale),
                "--sampling_steps_affinity",
                str(job_params.sampling_steps_affinity),
                "--diffusion_samples_affinity",
                str(job_params.diffusion_samples_affinity),
            ]
            for flag in (
                "use_potentials",
                "write_full_pae",
                "write_full_pde",
                "no_kernels",
                "affinity_mw_correction",
            ):
                if getattr(job_params, flag):
                    cmd.append(f"--{flag}")
            job_command = " ".join(cmd)

            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            stderr = result.stderr

            if result.returncode != 0:
                logger.error(f"boltz2 stderr: {stderr}")
                raise RuntimeError(f"boltz2 predict exited with code {result.returncode}")

            persist_job_output(job_id, output_dir)

        log = format_run_log(job_id, job_name, "boltz2", BOLTZ2_SPEC, job_command, result.stdout, stderr, started_at)
        mark_job_complete(job_id, log)
        logger.info(f"Done: job={job_id}")
    except Exception as e:
        logger.error(f"Failed: job={job_id}: {e}")
        log = format_run_log(job_id, job_name, "boltz2", BOLTZ2_SPEC, job_command, str(e), stderr, started_at)
        mark_job_failed(job_id, log)
        raise
