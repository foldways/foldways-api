import logging
import os
import sys

import modal

from constants import (
    API_SOURCES,
    APP_NAME,
    FASTAPI_SPEC,
    FOLDWAYS_TOML_CONFIG_FILENAME,
    FOLDWAYS_TOML_CONFIG_REMOTE_PATH,
    PYDANTIC_SPEC,
    VOLUME_NAME,
)

# Global logging config
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

app = modal.App(APP_NAME)


def configure_image(image: modal.Image) -> modal.Image:
    """Mount foldways.toml into the image. Requires manual creation, otherwise defaults used."""
    if os.path.exists(FOLDWAYS_TOML_CONFIG_FILENAME):
        image = image.add_local_file(FOLDWAYS_TOML_CONFIG_FILENAME, FOLDWAYS_TOML_CONFIG_REMOTE_PATH)
    return image


cpu_image = configure_image(
    modal.Image.debian_slim().pip_install(FASTAPI_SPEC, PYDANTIC_SPEC).add_local_python_source(*API_SOURCES)
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
