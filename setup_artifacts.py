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
    CHAI_ASSETS_URL,
    CHAI_COMPONENTS,
    CHAI_CONFORMERS_FILE,
    CHAI_ESM_LOCAL_PATH,
    CHAI_ESM_URL_PATH,
    ESM3_WEIGHTS_REPO,
    ESMC_600M_WEIGHTS_REPO,
    ESMFOLD2_LM_REPO,
    ESMFOLD2_WEIGHTS_REPO,
    IMMUNEBUILDER_WEIGHTS,
    IMMUNEBUILDER_ZENODO_BASE,
    INTELLIFOLD_CCD_FILE,
    INTELLIFOLD_CHECKPOINTS,
    INTELLIFOLD_DATA_FILES,
    INTELLIFOLD_WEIGHTS_REPO,
    LIGANDMPNN_CHECKPOINTS,
    LIGANDMPNN_SC_CHECKPOINT,
    LOCAL_MOCKS_DIR,
    MINUTES_20,
    MINUTES_30,
    MINUTES_60,
    PROTEINMPNN_CHECKPOINTS,
    PROTEINMPNN_WEIGHTS_URL,
    PROTENIX_CACHE_FILES,
    PROTENIX_DOWNLOAD_URL,
    PROTENIX_MODELS,
    SOLUBLEMPNN_CHECKPOINTS,
    VESM_MODELS,
    VESM_WEIGHTS_REPO,
    VOLUME_BOLTZ2_CACHE,
    VOLUME_BOLTZGEN_CACHE,
    VOLUME_CHAI_CACHE,
    VOLUME_ESM3_CACHE,
    VOLUME_ESMC_CACHE,
    VOLUME_ESMFOLD2_CACHE,
    VOLUME_IMMUNEBUILDER_CACHE,
    VOLUME_INTELLIFOLD_CACHE,
    VOLUME_LIGANDMPNN_CACHE,
    VOLUME_MOCKS_DIR,
    VOLUME_MOCKS_SUBDIR,
    VOLUME_PROTEINMPNN_CACHE,
    VOLUME_PROTENIX_CACHE,
    VOLUME_ROOT,
    VOLUME_SOLUBLEMPNN_CACHE,
    VOLUME_VESM_CACHE,
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
        logger.info("Boltz-2 weights already on the volume. Skipping download.")
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
        logger.info("ESMC weights already on the volume. Skipping download.")
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
        logger.info("ESMFold2 weights already on the volume. Skipping download.")
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
        logger.info("ESM3 weights already on the volume. Skipping download.")
        return
    logger.info(f"Downloading ESM3 weights: {ESM3_WEIGHTS_REPO}")
    snapshot_download(repo_id=ESM3_WEIGHTS_REPO, cache_dir=VOLUME_ESM3_CACHE)
    volume.commit()
    logger.info(f"ESM3 weights ready at {VOLUME_ESM3_CACHE}")


@app.function(image=download_image, volumes={VOLUME_ROOT: volume}, timeout=MINUTES_60)
def download_chai_weights():
    """Pre-stage Chai-1 model components, conformers, and the ESM2 embedding weights.

    Chai fetches these lazily from its CDN at inference time into CHAI_DOWNLOADS_DIR.
    Staging them to the volume at the exact paths Chai expects lets the service run
    without any download. The ESM2 file is served under esm2/ but read back from esm/,
    so its local path differs from its URL path.
    """
    import shutil
    import urllib.request

    volume.reload()
    cache_path = Path(VOLUME_CHAI_CACHE)
    downloads = [
        (f"{CHAI_ASSETS_URL}/models_v2/{component}", f"models_v2/{component}") for component in CHAI_COMPONENTS
    ]
    downloads.append((f"{CHAI_ASSETS_URL}/{CHAI_CONFORMERS_FILE}", CHAI_CONFORMERS_FILE))
    downloads.append((f"{CHAI_ASSETS_URL}/{CHAI_ESM_URL_PATH}", CHAI_ESM_LOCAL_PATH))

    missing = [(url, cache_path / rel) for url, rel in downloads if not (cache_path / rel).exists()]
    if not missing:
        logger.info("Chai-1 weights already on the volume. Skipping download.")
        return
    for url, dest in missing:
        logger.info(f"Downloading Chai-1 weights: {url}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "python-requests/2.32"})
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with urllib.request.urlopen(request) as response, open(tmp, "wb") as f:
            shutil.copyfileobj(response, f)
        tmp.rename(dest)
    volume.commit()
    logger.info(f"Chai-1 weights ready at {VOLUME_CHAI_CACHE}")


@app.function(image=download_image, volumes={VOLUME_ROOT: volume}, timeout=MINUTES_30)
def download_boltzgen_weights():
    """Pre-stage BoltzGen weights and inference data."""
    from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

    volume.reload()
    cache_path = Path(VOLUME_BOLTZGEN_CACHE)
    if cache_path.exists() and any(cache_path.iterdir()):
        logger.info("BoltzGen weights already on the volume. Skipping download.")
        return
    snapshot_download(repo_id=BOLTZGEN_MODEL_REPO, cache_dir=VOLUME_BOLTZGEN_CACHE)
    snapshot_download(repo_id=BOLTZGEN_DATA_REPO, repo_type="dataset", cache_dir=VOLUME_BOLTZGEN_CACHE)
    volume.commit()
    logger.info(f"BoltzGen weights ready at {VOLUME_BOLTZGEN_CACHE}")


@app.function(image=download_image, volumes={VOLUME_ROOT: volume}, timeout=MINUTES_20)
def download_proteinmpnn_weights():
    """Pre-stage ProteinMPNN checkpoints and the shared side-chain packer from the IPD server.

    The packer checkpoint is checked per file, so it is added to a volume that was
    already staged with the design checkpoints before packing support existed.
    """
    import urllib.request

    volume.reload()
    cache_path = Path(VOLUME_PROTEINMPNN_CACHE)
    cache_path.mkdir(parents=True, exist_ok=True)
    checkpoints = (*PROTEINMPNN_CHECKPOINTS, LIGANDMPNN_SC_CHECKPOINT)
    missing = [filename for filename in checkpoints if not (cache_path / filename).exists()]
    if not missing:
        logger.info("ProteinMPNN weights already on the volume. Skipping download.")
        return
    for filename in missing:
        logger.info(f"Downloading ProteinMPNN weights: {filename}")
        urllib.request.urlretrieve(f"{PROTEINMPNN_WEIGHTS_URL}/{filename}", cache_path / filename)
    volume.commit()
    logger.info(f"ProteinMPNN weights ready at {VOLUME_PROTEINMPNN_CACHE}")


@app.function(image=download_image, volumes={VOLUME_ROOT: volume}, timeout=MINUTES_20)
def download_ligandmpnn_weights():
    """Pre-stage LigandMPNN checkpoints and the shared side-chain packer."""
    import urllib.request

    volume.reload()
    cache_path = Path(VOLUME_LIGANDMPNN_CACHE)
    cache_path.mkdir(parents=True, exist_ok=True)
    checkpoints = (*LIGANDMPNN_CHECKPOINTS, LIGANDMPNN_SC_CHECKPOINT)
    missing = [filename for filename in checkpoints if not (cache_path / filename).exists()]
    if not missing:
        logger.info("LigandMPNN weights already on the volume. Skipping download.")
        return
    for filename in missing:
        logger.info(f"Downloading LigandMPNN weights: {filename}")
        urllib.request.urlretrieve(f"{PROTEINMPNN_WEIGHTS_URL}/{filename}", cache_path / filename)
    volume.commit()
    logger.info(f"LigandMPNN weights ready at {VOLUME_LIGANDMPNN_CACHE}")


@app.function(image=download_image, volumes={VOLUME_ROOT: volume}, timeout=MINUTES_20)
def download_solublempnn_weights():
    """Pre-stage SolubleMPNN checkpoints and the shared side-chain packer."""
    import urllib.request

    volume.reload()
    cache_path = Path(VOLUME_SOLUBLEMPNN_CACHE)
    cache_path.mkdir(parents=True, exist_ok=True)
    checkpoints = (*SOLUBLEMPNN_CHECKPOINTS, LIGANDMPNN_SC_CHECKPOINT)
    missing = [filename for filename in checkpoints if not (cache_path / filename).exists()]
    if not missing:
        logger.info("SolubleMPNN weights already on the volume. Skipping download.")
        return
    for filename in missing:
        logger.info(f"Downloading SolubleMPNN weights: {filename}")
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
        logger.info("BindCraft weights already on the volume. Skipping download.")
        return
    params_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading BindCraft weights: {BINDCRAFT_AF2_WEIGHTS_URL}")
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "alphafold_params.tar"
        urllib.request.urlretrieve(BINDCRAFT_AF2_WEIGHTS_URL, archive_path)
        logger.info(f"Extracting BindCraft weights to {BINDCRAFT_AF2_PARAMS_DIR}")
        with tarfile.open(archive_path) as tar:
            tar.extractall(params_path, filter="data")
    if not (params_path / BINDCRAFT_AF2_PARAMS_MARKER).exists():
        raise RuntimeError(f"AlphaFold2 params missing {BINDCRAFT_AF2_PARAMS_MARKER} after extraction")
    volume.commit()
    logger.info(f"BindCraft weights ready at {BINDCRAFT_AF2_PARAMS_DIR}")


@app.function(image=download_image, volumes={VOLUME_ROOT: volume}, timeout=MINUTES_30)
def download_vesm_weights():
    """Pre-stage the base ESM2 models and the distilled VESM checkpoints."""
    from huggingface_hub import hf_hub_download, snapshot_download  # pyright: ignore[reportMissingImports]

    volume.reload()
    cache_path = Path(VOLUME_VESM_CACHE)
    if cache_path.exists() and any(cache_path.iterdir()):
        logger.info("VESM weights already on the volume. Skipping download.")
        return
    for base_repo in dict.fromkeys(VESM_MODELS.values()):
        logger.info(f"Downloading VESM weights: {base_repo}")
        snapshot_download(repo_id=base_repo, cache_dir=VOLUME_VESM_CACHE)
    for model_name in VESM_MODELS:
        logger.info(f"Downloading VESM weights: {model_name}.pth")
        hf_hub_download(repo_id=VESM_WEIGHTS_REPO, filename=f"{model_name}.pth", cache_dir=VOLUME_VESM_CACHE)
    volume.commit()
    logger.info(f"VESM weights ready at {VOLUME_VESM_CACHE}")


@app.function(image=download_image, volumes={VOLUME_ROOT: volume}, timeout=MINUTES_30)
def download_intellifold_weights():
    """Pre-stage IntelliFold checkpoints and the CCD dictionary from Hugging Face."""
    from huggingface_hub import hf_hub_download  # pyright: ignore[reportMissingImports]

    volume.reload()
    cache_path = Path(VOLUME_INTELLIFOLD_CACHE)
    cache_path.mkdir(parents=True, exist_ok=True)
    files = (INTELLIFOLD_CCD_FILE, *INTELLIFOLD_CHECKPOINTS, *INTELLIFOLD_DATA_FILES)
    missing = [filename for filename in files if not (cache_path / filename).exists()]
    if not missing:
        logger.info("IntelliFold weights already on the volume. Skipping download.")
        return
    for filename in missing:
        logger.info(f"Downloading IntelliFold weights: {filename}")
        hf_hub_download(repo_id=INTELLIFOLD_WEIGHTS_REPO, filename=filename, local_dir=VOLUME_INTELLIFOLD_CACHE)
    volume.commit()
    logger.info(f"IntelliFold weights ready at {VOLUME_INTELLIFOLD_CACHE}")


@app.function(image=download_image, volumes={VOLUME_ROOT: volume}, timeout=MINUTES_60)
def download_protenix_weights():
    """Pre-stage Protenix checkpoints and the CCD and cluster caches from the ByteDance TOS bucket.

    Protenix reads everything under PROTENIX_ROOT_DIR, checkpoints from checkpoint/ and the
    shared caches from common/, and downloads whatever is missing on the first run. Staging
    them to the volume at those exact paths lets the service run without any download.
    """
    import urllib.request

    volume.reload()
    cache_path = Path(VOLUME_PROTENIX_CACHE)
    checkpoint_dir = cache_path / "checkpoint"
    common_dir = cache_path / "common"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    common_dir.mkdir(parents=True, exist_ok=True)

    downloads = [(f"{PROTENIX_DOWNLOAD_URL}/common/{name}", common_dir / name) for name in PROTENIX_CACHE_FILES]
    downloads += [
        (f"{PROTENIX_DOWNLOAD_URL}/checkpoint/{model}.pt", checkpoint_dir / f"{model}.pt") for model in PROTENIX_MODELS
    ]
    missing = [(url, dest) for url, dest in downloads if not dest.exists()]
    if not missing:
        logger.info("Protenix weights already on the volume. Skipping download.")
        return
    for url, dest in missing:
        logger.info(f"Downloading Protenix weights: {url}")
        urllib.request.urlretrieve(url, dest)
    volume.commit()
    logger.info(f"Protenix weights ready at {VOLUME_PROTENIX_CACHE}")


@app.function(image=download_image, volumes={VOLUME_ROOT: volume}, timeout=MINUTES_60)
def download_immunebuilder_weights():
    """Pre-stage ImmuneBuilder weights straight from the Zenodo records the package points at."""
    import urllib.request

    volume.reload()
    cache_path = Path(VOLUME_IMMUNEBUILDER_CACHE)
    cache_path.mkdir(parents=True, exist_ok=True)
    missing = [(name, record) for name, record in IMMUNEBUILDER_WEIGHTS if not (cache_path / name).exists()]
    if not missing:
        logger.info("ImmuneBuilder weights already on the volume. Skipping download.")
        return
    for name, record in missing:
        url = f"{IMMUNEBUILDER_ZENODO_BASE}/{record}/files/{name}?download=1"
        logger.info(f"Downloading ImmuneBuilder weights: {url}")
        urllib.request.urlretrieve(url, cache_path / name)
    volume.commit()
    logger.info(f"ImmuneBuilder weights ready at {VOLUME_IMMUNEBUILDER_CACHE}")


def upload_mocks():
    """Copy the local mock fixtures onto the volume for mock job submissions."""
    logger.info("Uploading mock fixtures...")
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(LOCAL_MOCKS_DIR, VOLUME_MOCKS_SUBDIR)
    logger.info(f"Mock fixtures ready at {VOLUME_MOCKS_DIR}")


@app.local_entrypoint()
def main():
    logger.info("Setting up artifacts on the Modal volume...")
    download_boltz2_weights.remote()
    download_esmc_weights.remote()
    download_esmfold2_weights.remote()
    download_esm3_weights.remote()
    download_chai_weights.remote()
    download_boltzgen_weights.remote()
    download_proteinmpnn_weights.remote()
    download_ligandmpnn_weights.remote()
    download_solublempnn_weights.remote()
    download_bindcraft_weights.remote()
    download_vesm_weights.remote()
    download_intellifold_weights.remote()
    download_immunebuilder_weights.remote()
    download_protenix_weights.remote()
    upload_mocks()
    logger.info("Artifact setup complete.")
