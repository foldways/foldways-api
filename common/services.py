from dataclasses import dataclass

import modal
from pydantic import BaseModel

from services import (
    bindcraft,
    boltz2,
    boltzgen,
    chai,
    esm3,
    esmc,
    esmfold2,
    immunebuilder,
    intellifold,
    ligandmpnn,
    proteinmpnn,
    protenix,
    solublempnn,
    thermompnn,
    vesm,
)


@dataclass(frozen=True)
class ServiceEntry:
    """One service's wiring: its params model, its Modal function, and a description.

    Every service's `run` takes the same two arguments, `job_id` and a dump of its
    params model, so adding a service needs no dispatch code beyond this entry.
    """

    description: str
    params: type[BaseModel]
    run: modal.Function


SERVICES = {
    "boltz2": ServiceEntry(
        description="Boltz-2 biomolecular structure and binding-affinity prediction.",
        params=boltz2.Boltz2Params,
        run=boltz2.run,
    ),
    "esmc": ServiceEntry(
        description="ESMC protein language model sequence embeddings.",
        params=esmc.ESMCParams,
        run=esmc.run,
    ),
    "esmfold2": ServiceEntry(
        description="ESMFold2 all-atom biomolecular structure prediction.",
        params=esmfold2.ESMFold2Params,
        run=esmfold2.run,
    ),
    "esm3": ServiceEntry(
        description="ESM3 generative protein design across sequence and structure.",
        params=esm3.ESM3Params,
        run=esm3.run,
    ),
    "boltzgen": ServiceEntry(
        description="BoltzGen de novo protein design.",
        params=boltzgen.BoltzGenParams,
        run=boltzgen.run,
    ),
    "proteinmpnn": ServiceEntry(
        description="ProteinMPNN inverse folding, sequence design for a backbone.",
        params=proteinmpnn.ProteinMPNNParams,
        run=proteinmpnn.run,
    ),
    "ligandmpnn": ServiceEntry(
        description="LigandMPNN inverse folding, sequence design for a backbone in its ligand context.",
        params=ligandmpnn.LigandMPNNParams,
        run=ligandmpnn.run,
    ),
    "solublempnn": ServiceEntry(
        description="SolubleMPNN inverse folding, sequence design for a backbone biased toward soluble proteins.",
        params=solublempnn.SolubleMPNNParams,
        run=solublempnn.run,
    ),
    "thermompnn": ServiceEntry(
        description="ThermoMPNN point-mutation stability (ddG) prediction by site-saturation mutagenesis.",
        params=thermompnn.ThermoMPNNParams,
        run=thermompnn.run,
    ),
    "chai": ServiceEntry(
        description="Chai-1 all-atom structure prediction for proteins, ligands, nucleic acids, and glycans.",
        params=chai.ChaiParams,
        run=chai.run,
    ),
    "bindcraft": ServiceEntry(
        description="BindCraft de novo binder design against a target structure.",
        params=bindcraft.BindCraftParams,
        run=bindcraft.run,
    ),
    "vesm": ServiceEntry(
        description="VESM variant effect prediction, log-likelihood-ratio scores for sequence mutations.",
        params=vesm.VESMParams,
        run=vesm.run,
    ),
    "intellifold": ServiceEntry(
        description="IntelliFold all-atom structure prediction for proteins, ligands, and nucleic acids.",
        params=intellifold.IntelliFoldParams,
        run=intellifold.run,
    ),
    "immunebuilder": ServiceEntry(
        description="ImmuneBuilder structure prediction for antibodies, nanobodies, and T-cell receptors.",
        params=immunebuilder.ImmuneBuilderParams,
        run=immunebuilder.run,
    ),
    "protenix": ServiceEntry(
        description="Protenix all-atom structure prediction for proteins, ligands, nucleic acids, and ions.",
        params=protenix.ProtenixParams,
        run=protenix.run,
    ),
}
