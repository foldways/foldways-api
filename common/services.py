from dataclasses import dataclass

import modal
from pydantic import BaseModel

from services import boltz2, boltzgen, esmc, esmfold2, proteinmpnn


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
}
