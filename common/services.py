import importlib
from dataclasses import dataclass

import modal
from pydantic import BaseModel

from config import SERVICE_REGISTRY, get_enabled_services


@dataclass(frozen=True)
class ServiceEntry:
    """One service's wiring: its params model, its Modal function, and a description.

    Every service's `run` takes the same two arguments, `job_id` and a dump of its
    params model, so adding a service needs no dispatch code beyond a registry entry.
    """

    description: str
    params: type[BaseModel]
    run: modal.Function


def build_services() -> dict[str, ServiceEntry]:
    """Import each enabled service and build its entry.

    Only enabled modules are imported, and importing a service module is what
    registers its Modal function, so a disabled service is never registered and
    never deployed.
    """
    services: dict[str, ServiceEntry] = {}
    for name in get_enabled_services():
        entry = SERVICE_REGISTRY[name]
        module = importlib.import_module(entry.module)
        services[name] = ServiceEntry(
            description=entry.description,
            params=getattr(module, entry.params),
            run=module.run,
        )
    return services


SERVICES = build_services()
