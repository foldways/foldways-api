import modal
from fastapi import FastAPI

from api.batches import router as batches_router
from api.health import router as health_router
from api.jobs import router as jobs_router
from api.mcp import mcp_app, run_mcp_session_manager
from api.services import router as services_router
from config import get_modal_proxy_auth_setting
from constants import API_VERSION, ROUTE_PREFIX, VOLUME_ROOT
from core import app, cpu_image, volume

web_app = FastAPI(
    title="Foldways API",
    description="Protein engineering tools running on Modal.",
    version=API_VERSION,
    docs_url=None,
    redoc_url=None,
    lifespan=run_mcp_session_manager,
)

web_app.include_router(health_router, prefix=ROUTE_PREFIX)
web_app.include_router(services_router, prefix=ROUTE_PREFIX)
web_app.include_router(jobs_router, prefix=ROUTE_PREFIX)
web_app.include_router(batches_router, prefix=ROUTE_PREFIX)
web_app.mount(ROUTE_PREFIX or "/", mcp_app)  # last, since it matches every path


@app.function(image=cpu_image, volumes={VOLUME_ROOT: volume})
@modal.asgi_app(requires_proxy_auth=get_modal_proxy_auth_setting())
def api():
    return web_app
