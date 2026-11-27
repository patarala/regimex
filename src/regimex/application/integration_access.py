"""Map the existing HTTP operations to the capabilities of an integration key."""

from regimex.access_contracts import IntegrationScope
from regimex.service_contracts import Forbidden, Principal

# Each shared route is installed under /integration with exactly the listed method and scope.
DELEGATED_OPERATIONS: dict[tuple[str, str], IntegrationScope] = {
    ("GET", "/api/v1/storage"): "read",
    ("GET", "/api/v1/snapshots/{snapshot_id}"): "read",
    ("GET", "/api/v1/snapshots/{snapshot_id}/export"): "read",
    ("GET", "/api/v1/analyses"): "read",
    ("GET", "/api/v1/experiments"): "read",
    ("GET", "/api/v1/analyses/{analysis_id}"): "read",
    ("GET", "/api/v1/analyses/{analysis_id}/result"): "read",
    ("GET", "/api/v1/analyses/{analysis_id}/report"): "read",
    ("GET", "/api/v1/analyses/{analysis_id}/report/preview"): "read",
    ("GET", "/api/v1/analyses/{analysis_id}/comparison"): "read",
    ("GET", "/api/v1/analyses/{analysis_id}/comparison/preview"): "read",
    ("GET", "/api/v1/jobs/{operation_id}"): "read",
    ("GET", "/api/v1/deletions"): "delete",
    ("GET", "/api/v1/deletions/{plan_id}"): "delete",
    ("POST", "/api/v1/validate"): "upload",
    ("POST", "/api/v1/snapshots"): "upload",
    ("POST", "/api/v1/analyses"): "analyze",
    ("POST", "/api/v1/jobs/{operation_id}/cancel"): "analyze",
    ("PATCH", "/api/v1/experiments/{snapshot_id}/name"): "upload",
    ("POST", "/api/v1/analyses/{analysis_id}/deletion-preview"): "delete",
    ("POST", "/api/v1/experiments/{snapshot_id}/deletion-preview"): "delete",
    ("POST", "/api/v1/deletions/{plan_id}"): "delete",
}


def required_scopes(method: str, route: str) -> tuple[IntegrationScope, ...]:
    """Resolve capabilities for a matched public or delegated route."""
    if method == "POST" and route == "/api/v1/integration/analysis-intake":
        return ("upload", "analyze")
    if method == "HEAD":
        method = "GET"
    if method == "GET" and route == "/api/v1/integration/analyses/{analysis_id}/delivery":
        return ("read",)
    ordinary_route = route.replace("/api/v1/integration/", "/api/v1/", 1)
    permission = DELEGATED_OPERATIONS.get((method, ordinary_route))
    if permission is not None:
        return (permission,)
    if method == "GET" and route == ordinary_route:
        return ("read",)
    raise Forbidden("This operation is unavailable to integration credentials.")


def authorize_request(principal: Principal, method: str, route: str) -> None:
    """Capabilities narrow access; resource ownership checks remain mandatory."""
    if principal.role != "integration":
        return
    if method == "GET" and route == "/api/v1/connection":
        # Introspection describes the caller's own key, including absent read
        # permission. It grants no access to models, subjects or observations.
        return
    if not set(required_scopes(method, route)).issubset(principal.integration_scopes):
        raise Forbidden("This integration key does not allow the requested operation.")
