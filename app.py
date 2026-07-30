import modal
from fastapi import FastAPI

from api.batches import router as batches_router
from api.health import router as health_router
from api.jobs import router as jobs_router
from api.services import router as services_router
from constants import API_VERSION, ROUTE_PREFIX, VOLUME_ROOT
from core import app, cpu_image, volume

web_app = FastAPI(
    title="Foldways API",
    description="Protein engineering tools running on Modal.",
    version=API_VERSION,
)

web_app.include_router(health_router, prefix=ROUTE_PREFIX)
web_app.include_router(services_router, prefix=ROUTE_PREFIX)
web_app.include_router(jobs_router, prefix=ROUTE_PREFIX)
web_app.include_router(batches_router, prefix=ROUTE_PREFIX)


@app.function(image=cpu_image, volumes={VOLUME_ROOT: volume})
@modal.asgi_app()
def api():
    return web_app
