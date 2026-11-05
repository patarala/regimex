"""Offline comparison of two committed predictions on one original snapshot."""

import html
import re
from dataclasses import dataclass
from importlib.resources import files
from typing import TYPE_CHECKING

from regimex.application.report import _hash_source, _script_json, validate_analysis_result
from regimex.directions import DirectionInterval
from regimex.exports import ExportV2
from regimex.infrastructure.object_store import ObjectCapacityError, ObjectIntegrityError
from regimex.report_contracts import MAX_REPORT_BYTES
from regimex.report_licenses import license_notices
from regimex.storage import json_bytes

if TYPE_CHECKING:
    from regimex.application.service import AnalysisResult

COMPARISON_RENDERER_VERSION = "analysis-comparison/1.0"


@dataclass(frozen=True, slots=True)
class RenderedComparison:
    html: bytes
    renderer_version: str


def _state(interval: DirectionInterval) -> str:
    return interval.direction.value if interval.direction is not None else interval.status


def _differences(
    first: tuple[DirectionInterval, ...], second: tuple[DirectionInterval, ...]
) -> list[dict[str, object]]:
    """Intersect complete interval partitions without treating gaps as directions."""
    differences: list[dict[str, object]] = []
    left = right = 0
    while left < len(first) and right < len(second):
        a, b = first[left], second[right]
        start, stop = max(a.start, b.start), min(a.stop, b.stop)
        state_a, state_b = _state(a), _state(b)
        if state_a != state_b:
            kind = (
                "direction"
                if a.direction is not None and b.direction is not None
                else "availability"
            )
            if (
                differences
                and differences[-1]["stop"] == start
                and differences[-1]["first"] == state_a
                and differences[-1]["second"] == state_b
            ):
                differences[-1]["stop"] = stop
            else:
                differences.append(
                    {
                        "start": start,
                        "stop": stop,
                        "first": state_a,
                        "second": state_b,
                        "kind": kind,
                    }
                )
        if a.stop == stop:
            left += 1
        if b.stop == stop:
            right += 1
    return differences


def _metadata(result: AnalysisResult, result_sha256: str) -> dict[str, object]:
    return {
        "analysis_id": str(result.analysis_id),
        "model": result.model.model_dump(mode="json"),
        "architecture_id": result.architecture_id,
        "implementation_version": result.implementation_version,
        "decoder_version": result.decoder_version,
        "analysis_profile_id": result.analysis_profile_id,
        "analysis": result.analysis.model_dump(mode="json"),
        "result_sha256": result_sha256,
    }


def render_analysis_comparison(
    export: ExportV2,
    first: AnalysisResult,
    second: AnalysisResult,
    *,
    first_sha256: str,
    second_sha256: str,
) -> RenderedComparison:
    """Compare saved predictions without fitting or loading a model backend.

    The application authorizes both analysis resources before calling this pure
    rendering boundary. SVG rendering needs no scientific or plotting packages.
    """
    export, first = validate_analysis_result(export, first, result_sha256=first_sha256)
    _, second = validate_analysis_result(export, second, result_sha256=second_sha256)
    if first.snapshot_id != second.snapshot_id:
        raise ObjectIntegrityError("comparison analyses must belong to the same saved snapshot")
    if first.analysis_id == second.analysis_id:
        raise ValueError("choose two different analyses to compare")

    series: list[dict[str, object]] = []
    for instrument, a, b in zip(export.instruments, first.sequences, second.sequences, strict=True):
        observations = tuple(
            row for row in export.observations if row.instrument_id == instrument.instrument_id
        )
        series.append(
            {
                "instrument": instrument.model_dump(mode="json"),
                "prices": [row.last_price for row in observations],
                "price_states": [row.price_state for row in observations],
                "intervals": [
                    [
                        {"start": item.start, "stop": item.stop, "state": _state(item)}
                        for item in sequence.intervals
                    ]
                    for sequence in (a, b)
                ],
                "differences": _differences(a.intervals, b.intervals),
            }
        )
    payload = _script_json(
        json_bytes(
            {
                "renderer_version": COMPARISON_RENDERER_VERSION,
                "source": export.source.model_dump(mode="json"),
                "snapshot_id": str(first.snapshot_id),
                "snapshot_sha256": export.content_sha256,
                "time_grid": export.time_grid.model_dump(mode="json"),
                "analyses": [_metadata(first, first_sha256), _metadata(second, second_sha256)],
                "series": series,
            }
        )
    )
    resources = files("regimex").joinpath("templates")
    template = resources.joinpath("analysis-comparison.html").read_text(encoding="utf-8")
    css = resources.joinpath("analysis-comparison.css").read_text(encoding="utf-8")
    script = resources.joinpath("analysis-comparison.js").read_text(encoding="utf-8")
    csp = (
        "default-src 'none'; base-uri 'none'; form-action 'none'; connect-src 'none'; "
        "img-src data:; style-src 'unsafe-inline'; script-src "
        + " ".join(_hash_source(item) for item in (payload, script))
    )
    replacements = {
        "@@LICENSES@@": license_notices(),
        "@@CSP@@": html.escape(csp, quote=True),
        "@@CSS@@": css,
        "@@PAYLOAD@@": payload,
        "@@SCRIPT@@": script,
        "@@RUN@@": html.escape(export.source.run_id),
        "@@FIRST@@": html.escape(first.model.model_version_id),
        "@@SECOND@@": html.escape(second.model.model_version_id),
    }
    document = re.sub(r"@@[A-Z_]+@@", lambda match: replacements[match[0]], template).encode()
    if len(document) > MAX_REPORT_BYTES:
        raise ObjectCapacityError("the complete comparison exceeds its 16 MiB report limit")
    return RenderedComparison(document, COMPARISON_RENDERER_VERSION)
