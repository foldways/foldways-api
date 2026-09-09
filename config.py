import functools
import tomllib
from dataclasses import dataclass, replace

from constants import (
    FOLDWAYS_TOML_CONFIG_FILENAME,
    FOLDWAYS_TOML_CONFIG_REMOTE_PATH,
    GPU_H100,
    GPU_L4,
    GPU_T4,
    HOURS_6,
    MAX_CONTAINERS,
    MINUTES_1,
    MINUTES_10,
    MINUTES_15,
    MINUTES_30,
)


@dataclass(frozen=True)
class RegistryEntry:
    """One service's static wiring, enough to build it without importing it.

    `module` is imported lazily, only when the service is enabled, and `params` is
    the name of its params model within that module.
    """

    module: str
    params: str
    description: str


SERVICE_REGISTRY: dict[str, RegistryEntry] = {
    "boltz2": RegistryEntry(
        "services.boltz2", "Boltz2Params", "Boltz-2 biomolecular structure and binding-affinity prediction."
    ),
    "esmc": RegistryEntry("services.esmc", "ESMCParams", "ESMC protein language model sequence embeddings."),
    "esmfold2": RegistryEntry(
        "services.esmfold2", "ESMFold2Params", "ESMFold2 all-atom biomolecular structure prediction."
    ),
    "esm3": RegistryEntry(
        "services.esm3", "ESM3Params", "ESM3 generative protein design across sequence and structure."
    ),
    "boltzgen": RegistryEntry("services.boltzgen", "BoltzGenParams", "BoltzGen de novo protein design."),
    "proteinmpnn": RegistryEntry(
        "services.proteinmpnn", "ProteinMPNNParams", "ProteinMPNN inverse folding, sequence design for a backbone."
    ),
    "ligandmpnn": RegistryEntry(
        "services.ligandmpnn",
        "LigandMPNNParams",
        "LigandMPNN inverse folding, sequence design for a backbone in its ligand context.",
    ),
    "solublempnn": RegistryEntry(
        "services.solublempnn",
        "SolubleMPNNParams",
        "SolubleMPNN inverse folding, sequence design for a backbone biased toward soluble proteins.",
    ),
    "thermompnn": RegistryEntry(
        "services.thermompnn",
        "ThermoMPNNParams",
        "ThermoMPNN point-mutation stability (ddG) prediction by site-saturation mutagenesis.",
    ),
    "chai": RegistryEntry(
        "services.chai",
        "ChaiParams",
        "Chai-1 all-atom structure prediction for proteins, ligands, nucleic acids, and glycans.",
    ),
    "bindcraft": RegistryEntry(
        "services.bindcraft", "BindCraftParams", "BindCraft de novo binder design against a target structure."
    ),
    "vesm": RegistryEntry(
        "services.vesm",
        "VESMParams",
        "VESM variant effect prediction, log-likelihood-ratio scores for sequence mutations.",
    ),
    "intellifold": RegistryEntry(
        "services.intellifold",
        "IntelliFoldParams",
        "IntelliFold all-atom structure prediction for proteins, ligands, and nucleic acids.",
    ),
    "immunebuilder": RegistryEntry(
        "services.immunebuilder",
        "ImmuneBuilderParams",
        "ImmuneBuilder structure prediction for antibodies, nanobodies, and T-cell receptors.",
    ),
    "protenix": RegistryEntry(
        "services.protenix",
        "ProtenixParams",
        "Protenix all-atom structure prediction for proteins, ligands, nucleic acids, and ions.",
    ),
}


@functools.cache
def load_foldways_toml_config() -> dict:
    """Parse the config file once. An absent file reads as empty, meaning all
    defaults. The mounted copy is tried first, then the repository root."""
    paths = (FOLDWAYS_TOML_CONFIG_REMOTE_PATH, FOLDWAYS_TOML_CONFIG_FILENAME)
    for path in paths:
        try:
            with open(path, "rb") as f:
                return tomllib.load(f)
        except OSError:
            continue
    return {}


def get_enabled_services() -> list[str]:
    """Return the service names to include, in registry order."""
    names = list(SERVICE_REGISTRY)
    enabled = load_foldways_toml_config().get("services", {}).get("enabled", [])
    if not enabled:
        return names
    unknown = sorted(set(enabled) - set(SERVICE_REGISTRY))
    if unknown:
        raise ValueError(
            f"Unknown service(s) in {FOLDWAYS_TOML_CONFIG_FILENAME} [services].enabled: {unknown}. Valid services are: {names}."
        )
    selected = set(enabled)
    return [name for name in names if name in selected]


@dataclass(frozen=True)
class ServiceCompute:
    """Compute settings for one service."""

    gpu: str
    timeout: int
    max_containers: int = MAX_CONTAINERS
    scaledown_window: int = MINUTES_1


SERVICE_DEFAULTS: dict[str, ServiceCompute] = {
    "boltz2": ServiceCompute(GPU_H100, MINUTES_30),
    "esmc": ServiceCompute(GPU_L4, MINUTES_10),
    "esmfold2": ServiceCompute(GPU_H100, MINUTES_30),
    "esm3": ServiceCompute(GPU_H100, MINUTES_15),
    "boltzgen": ServiceCompute(GPU_H100, MINUTES_30),
    "proteinmpnn": ServiceCompute(GPU_L4, MINUTES_10),
    "ligandmpnn": ServiceCompute(GPU_L4, MINUTES_10),
    "solublempnn": ServiceCompute(GPU_L4, MINUTES_10),
    "thermompnn": ServiceCompute(GPU_L4, MINUTES_10),
    "chai": ServiceCompute(GPU_H100, MINUTES_30),
    "bindcraft": ServiceCompute(GPU_H100, HOURS_6),
    "vesm": ServiceCompute(GPU_L4, MINUTES_10),
    "intellifold": ServiceCompute(GPU_H100, MINUTES_30),
    "immunebuilder": ServiceCompute(GPU_T4, MINUTES_15),
    "protenix": ServiceCompute(GPU_H100, MINUTES_30),
}


@functools.cache
def get_service_compute(name: str) -> ServiceCompute:
    """Return a service's compute settings, the file's overrides merged over the
    defaults. An unrecognized key in the file's table raises, so a typo fails loudly."""
    overrides = load_foldways_toml_config().get("services", {}).get(name, {})
    return replace(SERVICE_DEFAULTS[name], **overrides)


BOLTZ2_GPU = get_service_compute("boltz2").gpu
BOLTZ2_TIMEOUT = get_service_compute("boltz2").timeout
BOLTZ2_MAX_CONTAINERS = get_service_compute("boltz2").max_containers
BOLTZ2_SCALEDOWN_WINDOW = get_service_compute("boltz2").scaledown_window

ESMC_GPU = get_service_compute("esmc").gpu
ESMC_TIMEOUT = get_service_compute("esmc").timeout
ESMC_MAX_CONTAINERS = get_service_compute("esmc").max_containers
ESMC_SCALEDOWN_WINDOW = get_service_compute("esmc").scaledown_window

ESMFOLD2_GPU = get_service_compute("esmfold2").gpu
ESMFOLD2_TIMEOUT = get_service_compute("esmfold2").timeout
ESMFOLD2_MAX_CONTAINERS = get_service_compute("esmfold2").max_containers
ESMFOLD2_SCALEDOWN_WINDOW = get_service_compute("esmfold2").scaledown_window

ESM3_GPU = get_service_compute("esm3").gpu
ESM3_TIMEOUT = get_service_compute("esm3").timeout
ESM3_MAX_CONTAINERS = get_service_compute("esm3").max_containers
ESM3_SCALEDOWN_WINDOW = get_service_compute("esm3").scaledown_window

BOLTZGEN_GPU = get_service_compute("boltzgen").gpu
BOLTZGEN_TIMEOUT = get_service_compute("boltzgen").timeout
BOLTZGEN_MAX_CONTAINERS = get_service_compute("boltzgen").max_containers
BOLTZGEN_SCALEDOWN_WINDOW = get_service_compute("boltzgen").scaledown_window

PROTEINMPNN_GPU = get_service_compute("proteinmpnn").gpu
PROTEINMPNN_TIMEOUT = get_service_compute("proteinmpnn").timeout
PROTEINMPNN_MAX_CONTAINERS = get_service_compute("proteinmpnn").max_containers
PROTEINMPNN_SCALEDOWN_WINDOW = get_service_compute("proteinmpnn").scaledown_window

LIGANDMPNN_GPU = get_service_compute("ligandmpnn").gpu
LIGANDMPNN_TIMEOUT = get_service_compute("ligandmpnn").timeout
LIGANDMPNN_MAX_CONTAINERS = get_service_compute("ligandmpnn").max_containers
LIGANDMPNN_SCALEDOWN_WINDOW = get_service_compute("ligandmpnn").scaledown_window

SOLUBLEMPNN_GPU = get_service_compute("solublempnn").gpu
SOLUBLEMPNN_TIMEOUT = get_service_compute("solublempnn").timeout
SOLUBLEMPNN_MAX_CONTAINERS = get_service_compute("solublempnn").max_containers
SOLUBLEMPNN_SCALEDOWN_WINDOW = get_service_compute("solublempnn").scaledown_window

THERMOMPNN_GPU = get_service_compute("thermompnn").gpu
THERMOMPNN_TIMEOUT = get_service_compute("thermompnn").timeout
THERMOMPNN_MAX_CONTAINERS = get_service_compute("thermompnn").max_containers
THERMOMPNN_SCALEDOWN_WINDOW = get_service_compute("thermompnn").scaledown_window

CHAI_GPU = get_service_compute("chai").gpu
CHAI_TIMEOUT = get_service_compute("chai").timeout
CHAI_MAX_CONTAINERS = get_service_compute("chai").max_containers
CHAI_SCALEDOWN_WINDOW = get_service_compute("chai").scaledown_window

BINDCRAFT_GPU = get_service_compute("bindcraft").gpu
BINDCRAFT_TIMEOUT = get_service_compute("bindcraft").timeout
BINDCRAFT_MAX_CONTAINERS = get_service_compute("bindcraft").max_containers
BINDCRAFT_SCALEDOWN_WINDOW = get_service_compute("bindcraft").scaledown_window

VESM_GPU = get_service_compute("vesm").gpu
VESM_TIMEOUT = get_service_compute("vesm").timeout
VESM_MAX_CONTAINERS = get_service_compute("vesm").max_containers
VESM_SCALEDOWN_WINDOW = get_service_compute("vesm").scaledown_window

INTELLIFOLD_GPU = get_service_compute("intellifold").gpu
INTELLIFOLD_TIMEOUT = get_service_compute("intellifold").timeout
INTELLIFOLD_MAX_CONTAINERS = get_service_compute("intellifold").max_containers
INTELLIFOLD_SCALEDOWN_WINDOW = get_service_compute("intellifold").scaledown_window

IMMUNEBUILDER_GPU = get_service_compute("immunebuilder").gpu
IMMUNEBUILDER_TIMEOUT = get_service_compute("immunebuilder").timeout
IMMUNEBUILDER_MAX_CONTAINERS = get_service_compute("immunebuilder").max_containers
IMMUNEBUILDER_SCALEDOWN_WINDOW = get_service_compute("immunebuilder").scaledown_window

PROTENIX_GPU = get_service_compute("protenix").gpu
PROTENIX_TIMEOUT = get_service_compute("protenix").timeout
PROTENIX_MAX_CONTAINERS = get_service_compute("protenix").max_containers
PROTENIX_SCALEDOWN_WINDOW = get_service_compute("protenix").scaledown_window
