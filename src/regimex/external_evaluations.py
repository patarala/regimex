"""Read the selected model's scores from an offline direction comparison."""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from regimex.contracts import Identifier, StrictModel
from regimex.operations import Sha256

Score = Annotated[float, Field(ge=0, le=1)]


EvaluationKind = Literal["native_final", "controlled_transfer"]


class ExternalEvaluation(StrictModel):
    kind: EvaluationKind
    bundle_sha256: Sha256
    freeze_sha256: Sha256
    report_sha256: Sha256
    protocol_sha256: Sha256
    candidate: Identifier
    sample_count: Annotated[int, Field(ge=1, le=2**31 - 1)]
    interval_macro_f1: Score
    point_macro_f1: Score | None = None
    passed: bool


def read_evidence_file(path: Path) -> tuple[dict[str, Any], str]:
    with path.open("rb") as source:
        data = source.read(8 * 1024**2 + 1)
    if len(data) > 8 * 1024**2:
        raise ValueError("External comparison file exceeds 8 MiB")
    report = json.loads(data)
    if not isinstance(report, dict):
        raise ValueError("Expected a JSON object in the external comparison")
    return report, hashlib.sha256(data).hexdigest()


def _controlled_passed(report: dict[str, Any], candidate: str) -> bool:
    # The original single-challenger protocol stored its gates directly.
    checks = report["checks"].get(candidate, report["checks"])
    per_class = checks["per_class"]
    if set(per_class) != {"sideways", "growth", "decline"}:
        raise ValueError("Incomplete controlled checks")
    gates = [checks["pooled"], checks["stability"], *per_class.values()]
    if any(type(gate) is not bool for gate in gates):
        raise ValueError("Invalid controlled checks")
    passed = all(gates)
    if "passed" in report and report["passed"] is not passed:
        raise ValueError("Inconsistent controlled result")
    return passed


def read_external_evaluations(
    manifest_sha256: str,
    *,
    selection_file: Path | None,
    native_final: Path | None,
    controlled_final: Path | None,
) -> tuple[ExternalEvaluation, ...]:
    if selection_file is None:
        if native_final is not None or controlled_final is not None:
            raise ValueError("Final reports require --selection-file to bind them to the bundle")
        return ()
    frozen, freeze_sha = read_evidence_file(selection_file)
    if frozen.get("bundle_sha256") != manifest_sha256:
        raise ValueError("Frozen selection belongs to another bundle")
    results = []
    reports: tuple[tuple[EvaluationKind, Path | None], ...] = (
        ("native_final", native_final),
        ("controlled_transfer", controlled_final),
    )
    for kind, path in reports:
        if path is None:
            continue
        report, report_sha = read_evidence_file(path)
        candidate = frozen.get("challenger")
        if (
            not isinstance(candidate, str)
            or candidate != report.get("challenger")
            or report.get("freeze_sha256") != freeze_sha
            or report.get("bundle_sha256", manifest_sha256) != manifest_sha256
        ):
            raise ValueError("Final report belongs to a different selection or candidate")
        try:
            if kind == "native_final":
                scores = report["scores"][candidate]
                count = report["experiments"]
                interval = scores["interval_macro_f1"]
                point = scores["point_macro_f1"]
                passed = report["native_qualified"]
                protocol = frozen["protocol_sha256"]
            else:
                protocol = report["plan_sha256"]
                if frozen.get("controlled_plan_sha256", protocol) != protocol:
                    raise ValueError("Controlled plan differs from selection")
                count = report["paired_paths"]
                interval = report["results"][candidate]["interval"]["pooled"]["macro_f1"]
                point = None
                passed = _controlled_passed(report, candidate)
            result = ExternalEvaluation(
                kind=kind,
                bundle_sha256=manifest_sha256,
                freeze_sha256=freeze_sha,
                report_sha256=report_sha,
                protocol_sha256=protocol,
                candidate=candidate,
                sample_count=count,
                interval_macro_f1=interval,
                point_macro_f1=point,
                passed=passed,
            )
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise ValueError(f"Invalid {kind} report for the frozen candidate") from exc
        results.append(result)
    return tuple(results)
