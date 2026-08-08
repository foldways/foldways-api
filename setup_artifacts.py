import logging
from pathlib import Path

import modal

from constants import (
    BINDCRAFT_AF2_PARAMS_DIR,
    BINDCRAFT_AF2_PARAMS_MARKER,
    BINDCRAFT_AF2_WEIGHTS_URL,
    BOLTZ2_WEIGHTS_REPO,
    BOLTZ2_WEIGHTS_REVISION,
    BOLTZGEN_DATA_REPO,
    BOLTZGEN_MODEL_REPO,
    ESM3_WEIGHTS_REPO,
    ESMC_600M_WEIGHTS_REPO,
    ESMFOLD2_LM_REPO,
    ESMFOLD2_WEIGHTS_REPO,
    LIGANDMPNN_CHECKPOINTS,
    LOCAL_MOCKS_DIR,
    MINUTES_20,
    MINUTES_30,
    MINUTES_60,
    PROTEINMPNN_CHECKPOINTS,
    PROTEINMPNN_WEIGHTS_URL,
    SOLUBLEMPNN_CHECKPOINTS,
    VOLUME_BOLTZ2_CACHE,
    VOLUME_BOLTZGEN_CACHE,
    VOLUME_ESM3_CACHE,
    VOLUME_ESMC_CACHE,
    VOLUME_ESMFOLD2_CACHE,
    VOLUME_LIGANDMPNN_CACHE,
    VOLUME_MOCKS_DIR,
    VOLUME_PROTEINMPNN_CACHE,
    VOLUME_ROOT,
    VOLUME_SOLUBLEMPNN_CACHE,
)
from core import app, volume

logger = logging.getLogger(__name__)


download_image = modal.Image.debian_slim().pip_install("huggingface_hub").add_local_python_source("constants", "core")


@app.function(image=download_image, volumes={VOLUME_ROOT: volume}, timeout=MINUTES_20)
def download_boltz2_weights():
    """Pre-stage Boltz-2 weights."""
    from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

    volume.reload()
    cache_path = Path(VOLUME_BOLTZ2_CACHE)
    if cache_path.exists() and any(cache_path.iterdir()):
        logger.info(f"Boltz-2 weights already on the volume, skipping {BOLTZ2_WEIGHTS_REPO}")
        return
    logger.info(f"Downloading Boltz-2 weights: {BOLTZ2_WEIGHTS_REPO}@{BOLTZ2_WEIGHTS_REVISION}")
    snapshot_download(
        repo_id=BOLTZ2_WEIGHTS_REPO,
        revision=BOLTZ2_WEIGHTS_REVISION,
        local_dir=VOLUME_BOLTZ2_CACHE,
    )
    volume.commit()
    logger.info(f"Boltz-2 weights ready at {VOLUME_BOLTZ2_CACHE}")


@app.function(image=download_image, volumes={VOLUME_ROOT: volume}, timeout=MINUTES_20)
def download_esmc_weights():
    """Pre-stage ESMC weights."""
    from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

    volume.reload()
    cache_path = Path(VOLUME_ESMC_CACHE)
    if cache_path.exists() and any(cache_path.iterdir()):
        logger.info(f"ESMC weights already on the volume, skipping {ESMC_600M_WEIGHTS_REPO}")
        return
    logger.info(f"Downloading ESMC weights: {ESMC_600M_WEIGHTS_REPO}")
    snapshot_download(repo_id=ESMC_600M_WEIGHTS_REPO, cache_dir=VOLUME_ESMC_CACHE)
    volume.commit()
    logger.info(f"ESMC weights ready at {VOLUME_ESMC_CACHE}")


@app.function(image=download_image, volumes={VOLUME_ROOT: volume}, timeout=MINUTES_30)
def download_esmfold2_weights():
    """Pre-stage ESMFold2 weights and the ESMC 6B backbone it depends on."""
    from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

    volume.reload()
    cache_path = Path(VOLUME_ESMFOLD2_CACHE)
    if cache_path.exists() and any(cache_path.iterdir()):
        logger.info(f"ESMFold2 weights already on the volume, skipping {ESMFOLD2_WEIGHTS_REPO}")
        return
    for repo in (ESMFOLD2_WEIGHTS_REPO, ESMFOLD2_LM_REPO):
        logger.info(f"Downloading ESMFold2 weights: {repo}")
        snapshot_download(repo_id=repo, cache_dir=VOLUME_ESMFOLD2_CACHE)
    volume.commit()
    logger.info(f"ESMFold2 weights ready at {VOLUME_ESMFOLD2_CACHE}")


@app.function(image=download_image, volumes={VOLUME_ROOT: volume}, timeout=MINUTES_30)
def download_esm3_weights():
    """Pre-stage ESM3-open weights.

    The single repo holds the main model plus the structure and function submodels
    ESM3 loads, so one snapshot covers everything the service resolves at load time.
    """
    from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

    volume.reload()
    cache_path = Path(VOLUME_ESM3_CACHE)
    if cache_path.exists() and any(cache_path.iterdir()):
        logger.info(f"ESM3 weights already on the volume, skipping {ESM3_WEIGHTS_REPO}")
        return
    logger.info(f"Downloading ESM3 weights: {ESM3_WEIGHTS_REPO}")
    snapshot_download(repo_id=ESM3_WEIGHTS_REPO, cache_dir=VOLUME_ESM3_CACHE)
    volume.commit()
    logger.info(f"ESM3 weights ready at {VOLUME_ESM3_CACHE}")


@app.function(image=download_image, volumes={VOLUME_ROOT: volume}, timeout=MINUTES_30)
def download_boltzgen_weights():
    """Pre-stage BoltzGen weights and inference data."""
    from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

    volume.reload()
    cache_path = Path(VOLUME_BOLTZGEN_CACHE)
    if cache_path.exists() and any(cache_path.iterdir()):
        logger.info(f"BoltzGen weights already on the volume, skipping {BOLTZGEN_MODEL_REPO}")
        return
    snapshot_download(repo_id=BOLTZGEN_MODEL_REPO, cache_dir=VOLUME_BOLTZGEN_CACHE)
    snapshot_download(repo_id=BOLTZGEN_DATA_REPO, repo_type="dataset", cache_dir=VOLUME_BOLTZGEN_CACHE)
    volume.commit()
    logger.info(f"BoltzGen weights ready at {VOLUME_BOLTZGEN_CACHE}")


@app.function(image=download_image, volumes={VOLUME_ROOT: volume}, timeout=MINUTES_20)
def download_proteinmpnn_weights():
    """Pre-stage ProteinMPNN checkpoints from the IPD file server."""
    import urllib.request

    volume.reload()
    cache_path = Path(VOLUME_PROTEINMPNN_CACHE)
    if cache_path.exists() and any(cache_path.iterdir()):
        logger.info(f"ProteinMPNN weights already on the volume, skipping {len(PROTEINMPNN_CHECKPOINTS)} checkpoints")
        return
    cache_path.mkdir(parents=True, exist_ok=True)
    for filename in PROTEINMPNN_CHECKPOINTS:
        logger.info(f"Downloading ProteinMPNN checkpoint: {filename}")
        urllib.request.urlretrieve(f"{PROTEINMPNN_WEIGHTS_URL}/{filename}", cache_path / filename)
    volume.commit()
    logger.info(f"ProteinMPNN weights ready at {VOLUME_PROTEINMPNN_CACHE}")


@app.function(image=download_image, volumes={VOLUME_ROOT: volume}, timeout=MINUTES_20)
def download_ligandmpnn_weights():
    """Pre-stage LigandMPNN checkpoints from the IPD file server."""
    import urllib.request

    volume.reload()
    cache_path = Path(VOLUME_LIGANDMPNN_CACHE)
    if cache_path.exists() and any(cache_path.iterdir()):
        logger.info(f"LigandMPNN weights already on the volume, skipping {len(LIGANDMPNN_CHECKPOINTS)} checkpoints")
        return
    cache_path.mkdir(parents=True, exist_ok=True)
    for filename in LIGANDMPNN_CHECKPOINTS:
        logger.info(f"Downloading LigandMPNN checkpoint: {filename}")
        urllib.request.urlretrieve(f"{PROTEINMPNN_WEIGHTS_URL}/{filename}", cache_path / filename)
    volume.commit()
    logger.info(f"LigandMPNN weights ready at {VOLUME_LIGANDMPNN_CACHE}")


@app.function(image=download_image, volumes={VOLUME_ROOT: volume}, timeout=MINUTES_20)
def download_solublempnn_weights():
    """Pre-stage SolubleMPNN checkpoints from the IPD file server."""
    import urllib.request

    volume.reload()
    cache_path = Path(VOLUME_SOLUBLEMPNN_CACHE)
    if cache_path.exists() and any(cache_path.iterdir()):
        logger.info(f"SolubleMPNN weights already on the volume, skipping {len(SOLUBLEMPNN_CHECKPOINTS)} checkpoints")
        return
    cache_path.mkdir(parents=True, exist_ok=True)
    for filename in SOLUBLEMPNN_CHECKPOINTS:
        logger.info(f"Downloading SolubleMPNN checkpoint: {filename}")
        urllib.request.urlretrieve(f"{PROTEINMPNN_WEIGHTS_URL}/{filename}", cache_path / filename)
    volume.commit()
    logger.info(f"SolubleMPNN weights ready at {VOLUME_SOLUBLEMPNN_CACHE}")


@app.function(image=download_image, volumes={VOLUME_ROOT: volume}, timeout=MINUTES_60)
def download_bindcraft_weights():
    """Pre-stage the AlphaFold2 parameters BindCraft designs against."""
    import tarfile
    import tempfile
    import urllib.request

    volume.reload()
    params_path = Path(BINDCRAFT_AF2_PARAMS_DIR)
    if (params_path / BINDCRAFT_AF2_PARAMS_MARKER).exists():
        logger.info("BindCraft AlphaFold2 params already on the volume, skipping")
        return
    params_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading BindCraft AlphaFold2 params: {BINDCRAFT_AF2_WEIGHTS_URL}")
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "alphafold_params.tar"
        urllib.request.urlretrieve(BINDCRAFT_AF2_WEIGHTS_URL, archive_path)
        logger.info(f"Extracting BindCraft AlphaFold2 params to {BINDCRAFT_AF2_PARAMS_DIR}")
        with tarfile.open(archive_path) as tar:
            tar.extractall(params_path, filter="data")
    if not (params_path / BINDCRAFT_AF2_PARAMS_MARKER).exists():
        raise RuntimeError(f"AlphaFold2 params missing {BINDCRAFT_AF2_PARAMS_MARKER} after extraction")
    volume.commit()
    logger.info(f"BindCraft AlphaFold2 params ready at {BINDCRAFT_AF2_PARAMS_DIR}")


def upload_mocks():
    """Copy the local mock fixtures onto the volume for mock job submissions."""
    logger.info("Uploading mock fixtures...")
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(LOCAL_MOCKS_DIR, VOLUME_MOCKS_DIR)
    logger.info(f"Mock fixtures ready at {VOLUME_MOCKS_DIR}")


@app.local_entrypoint()
def main():
    logger.info("Setting up artifacts on the Modal volume...")
    download_boltz2_weights.remote()
    download_esmc_weights.remote()
    download_esmfold2_weights.remote()
    download_esm3_weights.remote()
    download_boltzgen_weights.remote()
    download_proteinmpnn_weights.remote()
    download_ligandmpnn_weights.remote()
    download_solublempnn_weights.remote()
    download_bindcraft_weights.remote()
    upload_mocks()
    logger.info("Artifact setup complete.")
