from fastapi import APIRouter, HTTPException

from api.schemas import ServiceInfo
from common.services import SERVICES

router = APIRouter()


@router.get("/services", response_model=list[ServiceInfo])
def list_services():
    """List the available services and how to call each."""
    services = []
    for name, service_entry in SERVICES.items():
        services.append(
            ServiceInfo(
                name=name,
                description=service_entry.description,
                params_schema=service_entry.params.model_json_schema(),
            )
        )
    return services


@router.get("/services/{name}", response_model=ServiceInfo)
def get_service(name: str):
    """Describe a service: its description and the params to submit."""
    if name not in SERVICES:
        raise HTTPException(404, f"No such service: '{name}'")
    service_entry = SERVICES[name]
    return ServiceInfo(
        name=name,
        description=service_entry.description,
        params_schema=service_entry.params.model_json_schema(),
    )
