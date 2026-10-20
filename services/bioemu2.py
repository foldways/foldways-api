import logging
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import modal
from pydantic import BaseModel, Field, model_validator

from common.utils import format_run_log, mark_job_complete, mark_job_failed, persist_job_output
from config import (
    BIOEMU2_GPU,
    BIOEMU2_MAX_CONTAINERS,
    BIOEMU2_SCALEDOWN_WINDOW,
    BIOEMU2_TIMEOUT,
)
from constants import (
    BIOEMU2_CHECKPOINT,
    BIOEMU2_SPEC,
    PYDANTIC_SPEC,
    PYTHON_3_12,
    SERVICE_SOURCES,
    VOLUME_BIOEMU2_CACHE,
    VOLUME_ROOT,
)
from core import app, volume

logger = logging.getLogger(__name__)


bioemu2_image = (
    modal.Image.debian_slim(python_version=PYTHON_3_12)
    .apt_install("build-essential")
    .uv_pip_install(BIOEMU2_SPEC, PYDANTIC_SPEC)
    .add_local_python_source(*SERVICE_SOURCES)
)


CHAIN_ID_EXAMPLE = "A"
LIGAND_ID_EXAMPLE = "B"
ASYM_IDS_DESC = "Chain ids, one per identical copy of this entity. Each is at most 4 characters."
CHAIN_ID_DESC = "Chain id. Must match an id declared in `entities`."

MODE_COMMANDS = {"structure": "predict-structure", "equilibrium": "sample-equilibrium", "propagate": "propagate"}


# entities parameter types
class ModifiedResidue(BaseModel):
    res_id: int = Field(ge=1, description="Residue index (1-based) of the modified residue.", examples=[3])
    ccd_code: str = Field(description="CCD code of the modified residue.", examples=["TPO"])


class Protein(BaseModel):
    entity_type: Literal["protein"] = Field(description="Entity type.")
    sequence: str = Field(description="Amino-acid sequence in one-letter codes.", examples=["MKTAYIAKQR"])
    asym_ids: list[str] = Field(min_length=1, description=ASYM_IDS_DESC, examples=[[CHAIN_ID_EXAMPLE]])
    mods: list[ModifiedResidue] = Field(default_factory=list, description="Modified residues in the chain.")


class Nucleic(BaseModel):
    entity_type: Literal["rna", "dna"] = Field(description="Entity type.")
    sequence: str = Field(description="Nucleotide sequence.", examples=["GCAUAGC"])
    asym_ids: list[str] = Field(min_length=1, description=ASYM_IDS_DESC, examples=[[CHAIN_ID_EXAMPLE]])
    mods: list[ModifiedResidue] = Field(default_factory=list, description="Modified residues in the chain.")


class Ligand(BaseModel):
    """One non-polymer molecule. Give exactly one of `smiles` or `ccd_code`."""

    entity_type: Literal["ligand"] = Field(description="Entity type.")
    asym_ids: list[str] = Field(min_length=1, description=ASYM_IDS_DESC, examples=[[LIGAND_ID_EXAMPLE]])
    smiles: str | None = Field(
        default=None,
        description="Ligand as a SMILES string (exclusive with ccd_code).",
        examples=["N[C@@H](Cc1ccc(O)cc1)C(=O)O"],
    )
    ccd_code: str | None = Field(
        default=None, description="Ligand as a CCD code (exclusive with smiles).", examples=["ATP"]
    )

    @model_validator(mode="after")
    def require_smiles_or_ccd(self):
        if bool(self.smiles) == bool(self.ccd_code):
            raise ValueError("ligand requires exactly one of 'smiles' or 'ccd_code'")
        return self


Entity = Annotated[Protein | Nucleic | Ligand, Field(discriminator="entity_type")]


# bonds parameter types
class BondAtom(BaseModel):
    asym_id: str = Field(description=CHAIN_ID_DESC, examples=[CHAIN_ID_EXAMPLE])
    res_id: int = Field(ge=1, description="Residue index, 1-based (1 for ligands).", examples=[3])
    res_name: str = Field(description="Residue name, matching the sequence or modification.", examples=["CYS"])
    name: str = Field(description="Atom name, as in the component's CIF.", examples=["SG"])


class Bond(BaseModel):
    """A non-standard covalent bond, such as a disulfide. Peptide bonds are inferred from the sequence."""

    atom1: BondAtom = Field(description="First atom in the bond.")
    atom2: BondAtom = Field(description="Second atom in the bond.")


# contacts parameter types
class ContactToken(BaseModel):
    """One contact endpoint. Give `res_id` for a residue, `atom_name` for a ligand atom, or both."""

    asym_id: str = Field(description=CHAIN_ID_DESC, examples=[CHAIN_ID_EXAMPLE])
    res_id: int | None = Field(default=None, ge=1, description="Residue index, 1-based.", examples=[10])
    atom_name: str | None = Field(
        default=None,
        description="Atom name, required for ligand atoms and atomistic tokens in modified residues.",
    )

    @model_validator(mode="after")
    def require_residue_or_atom(self):
        if self.res_id is None and self.atom_name is None:
            raise ValueError("contact token requires 'res_id', 'atom_name', or both")
        return self


class Contact(BaseModel):
    """A pair of tokens to keep close during sampling. It conditions the model and adds no covalent bond."""

    token1: ContactToken = Field(description="First residue or atom in the contact.")
    token2: ContactToken = Field(description="Second residue or atom in the contact.")
    max_distance: float = Field(
        default=6.0, ge=4.0, le=20.0, description="Maximum distance in Angstroms (supported range 4-20 Å)."
    )


class BioEmu2Params(BaseModel):
    """Parameters BioEmu2 needs to sample structures of one biomolecular system.

    `mode` picks what is sampled. Use `structure` to predict the folded
    structure, `equilibrium` to sample the equilibrium ensemble of
    conformations the system visits, and `propagate` to sample where a starting
    structure moves after `lag_time_ns` nanoseconds.

    Every mode needs the system, given in one of two ways. `entities` declares
    each molecule explicitly, and `bonds` and `contacts` refer back to the
    chain ids it names in `asym_ids`. `pdb_id` instead loads the molecules of
    an RCSB entry.

    Samples are written as one trajectory, `samples.xtc`, with its topology in
    `topology.cif`, and `physicality.json` reports which frames pass the
    physicality checks. To chain a sampling job into a propagation job, send
    the same `entities` or `pdb_id` again with that job's `topology.cif` as
    `start_cif`.

    Multiple sequence alignments are generated automatically by the public MSA
    server, so no MSA needs to be supplied.
    """

    mode: Literal["structure", "equilibrium", "propagate"] = Field(
        default="equilibrium",
        description=(
            "`structure` predicts the folded structure, `equilibrium` samples the equilibrium "
            "ensemble, and `propagate` samples where `start_cif` moves after `lag_time_ns`."
        ),
    )
    entities: list[Entity] | None = Field(
        default=None,
        description=(
            "One entry per entity in the system (exclusive with pdb_id). Each entry sets "
            "`entity_type` to protein, rna, dna, or ligand, so a polymer chain and a small "
            "molecule are both declared here."
        ),
    )
    pdb_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_]{4,12}$",
        description=(
            "RCSB PDB entry to load the system from, as biological assembly 1 with its "
            "ligands (exclusive with entities). Its label chain ids name the chains."
        ),
        examples=["1AKE"],
    )
    bonds: list[Bond] | None = Field(
        default=None,
        description="Optional non-standard covalent bonds. Declare every disulfide here. Requires entities.",
    )
    contacts: list[Contact] | None = Field(
        default=None, description="Optional contact conditioning. Requires entities."
    )
    start_cif: str | None = Field(
        default=None,
        description=(
            "Starting structure for propagate mode, as inline mmCIF content (the file's text, "
            "not a path). Its atoms must match the system exactly, so use the topology.cif of "
            "an earlier bioemu2 job on the same system."
        ),
    )
    lag_time_ns: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Time lag in nanoseconds for propagate mode. 1 to 1000 is the range the model mostly saw in training."
        ),
        examples=[1.0],
    )
    num_samples: int = Field(
        default=8,
        ge=1,
        le=100,
        description="Number of structures to generate. In propagate mode, the number of endpoints sampled from start_cif.",
    )
    batch_size: int | None = Field(
        default=None,
        ge=1,
        description="Structures denoised together. Defaults to num_samples, so lower it if a large system runs out of memory.",
    )
    num_steps: int | None = Field(
        default=None, ge=1, le=1000, description="Denoising steps. Defaults to the sampler's own value."
    )
    sampler: Literal["edm", "edm_gd", "edm_fk", "edm_fkc"] = Field(
        default="edm",
        description=(
            "Sampling algorithm. `edm` is unsteered. The steered samplers apply the default "
            "physicality potentials, with gradient guidance (`edm_gd`), Feynman-Kac "
            "resampling (`edm_fk`), or the Feynman-Kac corrector (`edm_fkc`)."
        ),
    )
    physicality_filter: Literal["high_quality", "high_quality_per_residue", "no_violations", "no_filter"] = Field(
        default="high_quality",
        description="Physicality checks reported per frame in physicality.json. All frames are kept either way.",
    )
    seed: int | None = Field(default=None, description="Random seed. Unset gives different samples on each run.")

    @model_validator(mode="after")
    def validate_mode_inputs(self):
        """Check that the system and starting structure match what `mode` needs."""
        if (self.entities is None) == (self.pdb_id is None):
            raise ValueError("give exactly one of 'entities' or 'pdb_id'")
        if (self.bonds or self.contacts) and self.entities is None:
            raise ValueError("'bonds' and 'contacts' require 'entities'")
        if self.mode == "propagate":
            if self.start_cif is None or self.lag_time_ns is None:
                raise ValueError("propagate mode requires 'start_cif' and 'lag_time_ns'")
        elif self.start_cif is not None or self.lag_time_ns is not None:
            raise ValueError("'start_cif' and 'lag_time_ns' apply only to propagate mode")
        return self

    @model_validator(mode="after")
    def validate_chain_references(self):
        """Cross-check every chain id against `entities` before any GPU work starts.

        bioemu2 enforces these too, but only after a container has spun up and
        loaded the model. Rejecting here turns that into an immediate 422.
        """
        entity_types: dict[str, str] = {}
        for entity in self.entities or []:
            for chain_id in entity.asym_ids:
                if chain_id in entity_types:
                    raise ValueError(f"duplicate chain id '{chain_id}' in entities")
                if not 1 <= len(chain_id) <= 4:
                    raise ValueError(f"chain id '{chain_id}' must be 1 to 4 characters")
                entity_types[chain_id] = entity.entity_type

        for bond in self.bonds or []:
            validate_chain_reference(entity_types, bond.atom1.asym_id, "bond atom1")
            validate_chain_reference(entity_types, bond.atom2.asym_id, "bond atom2")

        for contact in self.contacts or []:
            for location, token in (("contact token1", contact.token1), ("contact token2", contact.token2)):
                validate_chain_reference(entity_types, token.asym_id, location)
                if entity_types[token.asym_id] == "ligand" and token.atom_name is None:
                    raise ValueError(f"{location} on ligand chain '{token.asym_id}' requires 'atom_name'")
        return self


def validate_chain_reference(entity_types: dict[str, str], chain_id: str, location: str) -> None:
    """Validate that a referenced chain id was declared in `entities`.

    Args:
        entity_types: Declared chain ids mapped to their entity type.
        chain_id: The chain id to validate.
        location: Where the reference appears, e.g. "bond atom1", named in the error.

    Raises:
        ValueError: If the chain id was never declared.
    """
    if chain_id not in entity_types:
        raise ValueError(f"{location} references chain '{chain_id}', which is not declared in entities")


def render_bioemu2_system(params: BioEmu2Params) -> dict:
    """Render validated params into bioemu2's system YAML as a dict.

    Args:
        params: The validated job params, with `entities` set.

    Returns:
        The system dict, with any contacts alongside, ready to be written as YAML.
    """
    system: dict = {
        "entities": [entity.model_dump(exclude_none=True) for entity in params.entities or []],
        "bonds": [bond.model_dump() for bond in params.bonds or []],
    }
    if params.contacts:
        system["contacts"] = [contact.model_dump(exclude_none=True) for contact in params.contacts]
    return system


@app.function(
    name="bioemu2",
    image=bioemu2_image,
    gpu=BIOEMU2_GPU,
    volumes={VOLUME_ROOT: volume},
    timeout=BIOEMU2_TIMEOUT,
    max_containers=BIOEMU2_MAX_CONTAINERS,
    scaledown_window=BIOEMU2_SCALEDOWN_WINDOW,
)
def run(job_id: str, job_name: str | None, params: dict) -> None:
    """Sample one system with BioEmu2 and persist the result.

    The system goes to bioemu2 as a YAML file when `entities` is given, or as a
    PDB id. bioemu2 can also read the system from `start_cif` alone, but that
    loses modified residues and extra bonds, so the system is always passed.

    Args:
        job_id: Unique name identifying the job.
        job_name: The caller's label for the job, recorded in the run log. None
            when the caller did not supply one.
        params: A BioEmu2Params dump, revalidated here so the container never
            trusts the payload it was handed.
    """
    import yaml

    logger.info(f"Starting bioemu2 sampling: job={job_id}")
    started_at = datetime.now(UTC)
    job_command = ""
    stderr = ""

    try:
        job_params = BioEmu2Params.model_validate(params)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_dir = tmpdir / "output"

            cmd = ["bioemu2", MODE_COMMANDS[job_params.mode]]
            if job_params.entities is not None:
                system_path = tmpdir / "system.yaml"
                system_path.write_text(yaml.dump(render_bioemu2_system(job_params), sort_keys=False))
                cmd += ["--system", str(system_path)]
            elif job_params.pdb_id is not None:
                cmd += ["--pdb-id", job_params.pdb_id]
            if job_params.start_cif is not None:
                start_path = tmpdir / "start.cif"
                start_path.write_text(job_params.start_cif)
                cmd += ["--x0-cif", str(start_path), "--lag-time-ns", str(job_params.lag_time_ns)]

            cmd += [
                "--out",
                str(output_dir),
                "--checkpoint",
                BIOEMU2_CHECKPOINT,
                "--checkpoint-cache-dir",
                VOLUME_BIOEMU2_CACHE,
                "--num-samples",
                str(job_params.num_samples),
                "--sampler",
                job_params.sampler,
                "--physicality-filter",
                job_params.physicality_filter,
            ]
            for flag, value in (
                ("--batch-size", job_params.batch_size),
                ("--num-steps", job_params.num_steps),
                ("--seed", job_params.seed),
            ):
                if value is not None:
                    cmd += [flag, str(value)]
            job_command = " ".join(cmd)

            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            stderr = result.stderr

            if result.returncode != 0:
                logger.error(f"bioemu2 stderr: {stderr}")
                raise RuntimeError(f"bioemu2 {MODE_COMMANDS[job_params.mode]} exited with code {result.returncode}")

            persist_job_output(job_id, output_dir)

        log = format_run_log(job_id, job_name, "bioemu2", BIOEMU2_SPEC, job_command, result.stdout, stderr, started_at)
        mark_job_complete(job_id, log)
        logger.info(f"Done: job={job_id}")
    except Exception as e:
        logger.error(f"Failed: job={job_id}: {e}")
        log = format_run_log(job_id, job_name, "bioemu2", BIOEMU2_SPEC, job_command, str(e), stderr, started_at)
        mark_job_failed(job_id, log)
        raise
