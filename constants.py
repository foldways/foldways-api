from enum import StrEnum

APP_NAME = "foldways"
API_VERSION = "0.7.0"
ROUTE_PREFIX = ""
LOCAL_MOCKS_DIR = "mocks"
MOCK_OUTPUT_DIR = "output"
MOCK_REQUEST_FILE = "request.json"
SERVICE_SOURCES = ("config", "constants", "core", "services", "common")
API_SOURCES = SERVICE_SOURCES + ("api",)

# Modal volume config
VOLUME_NAME = "foldways-data"
VOLUME_ROOT = "/data"
VOLUME_OUTPUTS_DIR = f"{VOLUME_ROOT}/outputs"
VOLUME_JOBS_DIR = f"{VOLUME_ROOT}/jobs"
VOLUME_MOCKS_DIR = f"{VOLUME_ROOT}/mocks"
VOLUME_BOLTZ2_CACHE = f"{VOLUME_ROOT}/boltz2_cache"
VOLUME_ESMC_CACHE = f"{VOLUME_ROOT}/esmc_cache"
VOLUME_ESMFOLD2_CACHE = f"{VOLUME_ROOT}/esmfold2_cache"
VOLUME_ESM3_CACHE = f"{VOLUME_ROOT}/esm3_cache"
VOLUME_BOLTZGEN_CACHE = f"{VOLUME_ROOT}/boltzgen_cache"
VOLUME_PROTEINMPNN_CACHE = f"{VOLUME_ROOT}/proteinmpnn_cache"
VOLUME_LIGANDMPNN_CACHE = f"{VOLUME_ROOT}/ligandmpnn_cache"
VOLUME_SOLUBLEMPNN_CACHE = f"{VOLUME_ROOT}/solublempnn_cache"
VOLUME_BINDCRAFT_CACHE = f"{VOLUME_ROOT}/bindcraft_cache"
VOLUME_CHAI_CACHE = f"{VOLUME_ROOT}/chai_cache"

# Modal job config
JOB_COMPLETE_MARKER = "job.log"
JOB_ERROR_MARKER = "error.log"

# Image dependency pins
FASTAPI_SPEC = "fastapi>=0.139.2,<0.140.0"
PYDANTIC_SPEC = "pydantic>=2.13.4,<3.0.0"

# Modal function config
MAX_CONTAINERS = 5
PYTHON_3_10 = "3.10"
PYTHON_3_11 = "3.11"
PYTHON_3_12 = "3.12"
MINUTES_1 = 1 * 60
MINUTES_10 = 10 * 60
MINUTES_15 = 15 * 60
MINUTES_20 = 20 * 60
MINUTES_30 = 30 * 60
MINUTES_40 = 40 * 60
MINUTES_60 = 60 * 60
HOURS_6 = 6 * 60 * 60

# Modal GPU types
GPU_T4 = "T4"
GPU_L4 = "L4"
GPU_A10G = "A10G"
GPU_L40S = "L40S"
GPU_A100 = "A100"
GPU_A100_40GB = "A100-40GB"
GPU_A100_80GB = "A100-80GB"
GPU_RTX_PRO_6000 = "RTX-PRO-6000"
GPU_H100 = "H100"
GPU_H200 = "H200"
GPU_B200 = "B200"

# Boltz-2
BOLTZ2_SPEC = "boltz==2.1.1"
BOLTZ2_WEIGHTS_REPO = "boltz-community/boltz-2"
BOLTZ2_WEIGHTS_REVISION = "6fdef46d763fee7fbb83ca5501ccceff43b85607"

# ESM
ESM_COMMIT_HASH = "917af90b624535eed1e072d343c717e3ec11fef4"
ESMC_SPEC = f"esm @ git+https://github.com/Biohub/esm.git@{ESM_COMMIT_HASH}"
ESMC_300M_WEIGHTS_REPO = "biohub/esmc-300m-2024-12"
ESMC_600M_WEIGHTS_REPO = "biohub/esmc-600m-2024-12"
ESMC_6B_WEIGHTS_REPO = "biohub/esmc-6b-2024-12"
ESMFOLD2_WEIGHTS_REPO = "biohub/ESMFold2"
ESMFOLD2_LM_REPO = "biohub/ESMC-6B"
ESM3_SPEC = ESMC_SPEC
ESM3_WEIGHTS_REPO = "biohub/esm3-sm-open-v1"
ESM3_MODEL_NAME = "esm3-sm-open-v1"

# BoltzGen
BOLTZGEN_SPEC = "boltzgen==0.3.2"
BOLTZGEN_MODEL_REPO = "boltzgen/boltzgen-1"
BOLTZGEN_DATA_REPO = "boltzgen/inference-data"

# LigandMPNN
LIGANDMPNN_REPO = "https://github.com/dauparas/LigandMPNN.git"
LIGANDMPNN_COMMIT = "26ec57ac976ade5379920dbd43c7f97a91cf82de"
LIGANDMPNN_DIR = "/opt/LigandMPNN"
PROTEINMPNN_SPEC = f"LigandMPNN@{LIGANDMPNN_COMMIT}"
LIGANDMPNN_SPEC = f"LigandMPNN@{LIGANDMPNN_COMMIT}"
SOLUBLEMPNN_SPEC = f"LigandMPNN@{LIGANDMPNN_COMMIT}"
PROTEINMPNN_WEIGHTS_URL = "https://files.ipd.uw.edu/pub/ligandmpnn"
PROTEINMPNN_CHECKPOINTS = (
    "proteinmpnn_v_48_002.pt",
    "proteinmpnn_v_48_010.pt",
    "proteinmpnn_v_48_020.pt",
    "proteinmpnn_v_48_030.pt",
)
LIGANDMPNN_CHECKPOINTS = (
    "ligandmpnn_v_32_005_25.pt",
    "ligandmpnn_v_32_010_25.pt",
    "ligandmpnn_v_32_020_25.pt",
    "ligandmpnn_v_32_030_25.pt",
)
SOLUBLEMPNN_CHECKPOINTS = (
    "solublempnn_v_48_002.pt",
    "solublempnn_v_48_010.pt",
    "solublempnn_v_48_020.pt",
    "solublempnn_v_48_030.pt",
)

# Chai-1
CHAI_SPEC = "chai_lab==0.6.1"
CHAI_ASSETS_URL = "https://chaiassets.com/chai1-inference-depencencies"
CHAI_COMPONENTS = (
    "feature_embedding.pt",
    "bond_loss_input_proj.pt",
    "token_embedder.pt",
    "trunk.pt",
    "diffusion_module.pt",
    "confidence_head.pt",
)
CHAI_CONFORMERS_FILE = "conformers_v1.apkl"
CHAI_ESM_URL_PATH = "esm2/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt"
CHAI_ESM_LOCAL_PATH = "esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt"

# ThermoMPNN
THERMOMPNN_REPO = "https://github.com/Kuhlman-Lab/ThermoMPNN.git"
THERMOMPNN_COMMIT = "2b04fd370e399911b1fa5848112cc9013f084110"
THERMOMPNN_DIR = "/opt/ThermoMPNN"
THERMOMPNN_SPEC = f"ThermoMPNN@{THERMOMPNN_COMMIT}"
THERMOMPNN_MODEL_PATH = f"{THERMOMPNN_DIR}/models/thermoMPNN_default.pt"

# BindCraft
BINDCRAFT_REPO = "https://github.com/martinpacesa/BindCraft.git"
BINDCRAFT_COMMIT = "b971db42ba6e091afab63ccb30ae02215150a990"
BINDCRAFT_DIR = "/opt/BindCraft"
BINDCRAFT_SPEC = f"BindCraft@{BINDCRAFT_COMMIT}"
COLABDESIGN_SPEC = "git+https://github.com/sokrypton/ColabDesign.git"
PYROSETTA_FIND_LINKS = "https://west.rosettacommons.org/pyrosetta/quarterly/release.cxx11thread.serialization"
PYROSETTA_SPEC = "pyrosetta==2026.3"
BINDCRAFT_AF2_WEIGHTS_URL = "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
BINDCRAFT_AF2_PARAMS_DIR = f"{VOLUME_BINDCRAFT_CACHE}/params"
BINDCRAFT_AF2_PARAMS_MARKER = "params_model_5_ptm.npz"
BINDCRAFT_SETTINGS_ADVANCED_DIR = f"{BINDCRAFT_DIR}/settings_advanced"
BINDCRAFT_SETTINGS_FILTERS_DIR = f"{BINDCRAFT_DIR}/settings_filters"

# Amino acids
ONE_LETTER_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class JobState(StrEnum):
    """The status values a single job can report."""

    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"
    INIT_FAILED = "init_failed"
    STOPPED = "stopped"
    TIMED_OUT = "timed_out"
    UNKNOWN = "unknown"


class BatchState(StrEnum):
    """The status values a batch can report, aggregated from its jobs."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    COMPLETED_WITH_FAILURES = "completed_with_failures"
