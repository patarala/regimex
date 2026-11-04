"""Public queue availability and explicit operator diagnostic results."""

from typing import Literal

from pydantic import AwareDatetime

from regimex.contracts import StrictModel

QueueName = Literal["analysis", "admin", "maintenance"]


class QueueStatus(StrictModel):
    state: Literal["available", "busy", "unavailable", "unknown"]
    checked_at: AwareDatetime
    message: str
    worker_capability: Literal["not_probed"] = "not_probed"


class DiagnosticCheck(StrictModel):
    name: str
    status: Literal["passed", "failed", "skipped"]
    message: str


class DiagnosticReport(StrictModel):
    checked_at: AwareDatetime
    checks: tuple[DiagnosticCheck, ...]
    model_probe_scope: Literal["local_isolated_process"] = "local_isolated_process"

    @property
    def healthy(self) -> bool:
        return bool(self.checks) and all(check.status == "passed" for check in self.checks)
