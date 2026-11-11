from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations

from api import batches, jobs, services
from api.schemas import (
    BatchDeleteResponse,
    BatchRequest,
    BatchStatus,
    BatchStopResponse,
    BatchSubmitResponse,
    JobDeleteResponse,
    JobFile,
    JobFiles,
    JobRequest,
    JobStatus,
    JobStopResponse,
    ServiceInfo,
    ServiceSummary,
)
from api.utils import reload_volume
from common.registries import jobs_registry
from common.services import SERVICES
from common.utils import check_job_complete, get_job_status
from config import get_modal_proxy_auth_setting
from constants import API_VERSION, MCP_MAX_READ_BYTES, MCP_SERVER_INSTRUCTIONS, VOLUME_JOBS_DIR

mcp_server = MCPServer(name="Foldways API", instructions=MCP_SERVER_INSTRUCTIONS, version=API_VERSION)


def call_route[T](route: Callable[..., T], *args) -> T:
    """Call one of the API's route functions from an MCP tool and return its result."""
    try:
        return route(*args)
    except HTTPException as e:
        raise ToolError(str(e.detail)) from e


def get_complete_job_dir(job_id: str) -> Path:
    """Return a job's output directory, or raise if the job is unknown or not complete."""
    reload_volume()
    if job_id not in jobs_registry:
        raise ToolError(f"No such job: '{job_id}'")
    if not check_job_complete(job_id):
        status = get_job_status(job_id, jobs_registry[job_id])
        raise ToolError(f"Output not available for '{job_id}' (status={status})")
    return Path(VOLUME_JOBS_DIR) / job_id


@mcp_server.tool()
def list_services() -> list[ServiceSummary]:
    """List the deployed services. Call get_service for the params a service takes."""
    return [ServiceSummary(name=name, description=entry.description) for name, entry in SERVICES.items()]


@mcp_server.tool()
def get_service(name: str) -> ServiceInfo:
    """Describe a service, including the JSON schema of the params to submit."""
    return call_route(services.get_service, name)


@mcp_server.tool()
def submit_job(service: str, params: dict, name: str | None = None, mock: bool = False) -> JobStatus:
    """Submit one job and return its id. Inline file contents such as FASTA or PDB text go in params.

    Set mock to true to return a stored example result without using a GPU.
    """
    reload_volume()
    return call_route(jobs.submit_job, JobRequest(service=service, params=params, name=name, mock=mock))


@mcp_server.tool()
def list_jobs() -> list[JobStatus]:
    """List every job and its status."""
    reload_volume()
    return call_route(jobs.list_jobs)


@mcp_server.tool()
def get_job(job_id: str) -> JobStatus:
    """Get a job's status."""
    reload_volume()
    return call_route(jobs.get_job, job_id)


@mcp_server.tool(annotations=ToolAnnotations(destructive_hint=True))
def stop_job(job_id: str) -> JobStopResponse:
    """Stop a running job. Its record and any output are kept."""
    reload_volume()
    return JobStopResponse(**call_route(jobs.stop_job, job_id))


@mcp_server.tool(annotations=ToolAnnotations(destructive_hint=True))
def delete_job(job_id: str) -> JobDeleteResponse:
    """Stop a job if it is running, then delete it and its output for good."""
    reload_volume()
    return JobDeleteResponse(**call_route(jobs.stop_and_delete_job, job_id))


@mcp_server.tool()
def submit_batch(service: str, params: list[dict], name: str | None = None, mock: bool = False) -> BatchSubmitResponse:
    """Run one service over many param sets, one job per set, and return the batch id and job ids.

    Set mock to true to return stored example results without using a GPU.
    """
    reload_volume()
    return call_route(batches.submit_batch, BatchRequest(service=service, params=params, name=name, mock=mock))


@mcp_server.tool()
def get_batch(batch_id: str) -> BatchStatus:
    """Get a batch's overall status and the status of each job in it."""
    reload_volume()
    return call_route(batches.get_batch, batch_id)


@mcp_server.tool(annotations=ToolAnnotations(destructive_hint=True))
def stop_batch(batch_id: str) -> BatchStopResponse:
    """Stop every running job in a batch. Records and any output are kept."""
    reload_volume()
    return BatchStopResponse(**call_route(batches.stop_batch, batch_id))


@mcp_server.tool(annotations=ToolAnnotations(destructive_hint=True))
def delete_batch(batch_id: str) -> BatchDeleteResponse:
    """Stop a batch, then delete every job in it and their output for good."""
    reload_volume()
    return BatchDeleteResponse(**call_route(batches.delete_batch, batch_id))


@mcp_server.tool()
def list_job_files(job_id: str, ctx: Context) -> JobFiles:
    """List a completed job's output files, with a link and a curl command to download them all as a zip."""
    job_dir = get_complete_job_dir(job_id)
    files = [
        JobFile(path=str(path.relative_to(job_dir)), size_bytes=path.stat().st_size)
        for path in sorted(job_dir.rglob("*"))
        if path.is_file()
    ]
    request = ctx.request_context.request
    base_url = str(request.base_url).rstrip("/") if request is not None else ""
    download_url = f"{base_url}/jobs/{job_id}/download"
    headers = '-H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET" ' if get_modal_proxy_auth_setting() else ""
    download_command = f'curl {headers}-o {job_id}.zip "{download_url}"'
    return JobFiles(job_id=job_id, files=files, download_url=download_url, download_command=download_command)


@mcp_server.tool()
def read_job_file(job_id: str, path: str) -> str:
    """Read one text output file of a completed job, such as a score table or a structure file.

    The path comes from list_job_files. Binary and very large files must be downloaded instead.
    """
    job_dir = get_complete_job_dir(job_id)
    file_path = (job_dir / path).resolve()
    if not file_path.is_relative_to(job_dir.resolve()) or not file_path.is_file():
        raise ToolError(f"No such file in job '{job_id}': '{path}'")
    size = file_path.stat().st_size
    if size > MCP_MAX_READ_BYTES:
        raise ToolError(f"'{path}' is {size} bytes, over the {MCP_MAX_READ_BYTES} byte limit. Download it instead.")
    try:
        return file_path.read_text()
    except UnicodeDecodeError:
        raise ToolError(f"'{path}' is not a text file. Download it instead.")


mcp_app = mcp_server.streamable_http_app(
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@asynccontextmanager
async def run_mcp_session_manager(_: FastAPI) -> AsyncGenerator[None]:
    """Run the MCP session manager, which a mounted app does not start on its own."""
    async with mcp_server.session_manager.run():
        yield
