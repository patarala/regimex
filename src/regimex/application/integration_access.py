"""Map the existing HTTP operations to the capabilities of an integration key."""

from regimex.access_contracts import IntegrationScope
from regimex.service_contracts import Forbidden, Principal

_MUTATIONS: dict[tuple[str, str], IntegrationScope] = {
    ("POST", "/api/v1/validate"): "upload",
    ("POST", "/api/v1/snapshots"): "upload",
    ("POST", "/api/v1/analyses"): "analyze",
    ("POST", "/api/v1/jobs/{operation_id}/cancel"): "analyze",
    ("PATCH", "/api/v1/experiments/{snapshot_id}/name"): "upload",
}


def required_scope(method: str, route: str) -> IntegrationScope:
    """Reject integration mutations without an explicitly assigned capability."""
    ordinary_route = route.replace("/api/v1/integration/", "/api/v1/", 1)
    if method in {"GET", "HEAD"}:
        return "read"
    permission = _MUTATIONS.get((method, ordinary_route))
    if permission is None:
        raise Forbidden("This operation is unavailable to integration credentials.")
    return permission


def authorize_request(principal: Principal, method: str, route: str) -> None:
    """Capabilities narrow access; resource ownership checks remain mandatory."""
    if principal.role != "integration":
        return
    if required_scope(method, route) not in principal.integration_scopes:
        raise Forbidden("This integration key does not allow the requested operation.")
