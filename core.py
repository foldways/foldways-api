import logging
import sys

import modal

from constants import API_SOURCES, APP_NAME, FASTAPI_SPEC, PYDANTIC_SPEC, VOLUME_NAME

# Global logging config
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

app = modal.App(APP_NAME)


cpu_image = modal.Image.debian_slim().pip_install(FASTAPI_SPEC, PYDANTIC_SPEC).add_local_python_source(*API_SOURCES)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
