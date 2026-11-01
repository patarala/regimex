"""Validated, self-contained HTML rendering, called only inside the analysis worker."""

import base64
import hashlib
import html
import re
from dataclasses import dataclass
from importlib.metadata import version
from importlib.resources import files
from typing import TYPE_CHECKING

from regimex.contracts import MAX_INPUT_BYTES, Observation
from regimex.directions import DirectionAnnotations
from regimex.exports import ExportInstrument, ExportObservation, ExportV2, parse_export_v2
from regimex.infrastructure.object_store import ObjectCapacityError, ObjectIntegrityError
from regimex.model_registry import RegistryError
from regimex.storage import json_bytes

if TYPE_CHECKING:
    from regimex.application.service import AnalysisResult

from regimex.report_contracts import MAX_REPORT_BYTES, REPORT_RENDERER_VERSION


@dataclass(frozen=True, slots=True)
class RenderedReport:
    html: bytes
    renderer_version: str
    plotly_version: str


@dataclass(frozen=True, slots=True)
class _Grid:
    experiment_id: str
    export: ExportV2

    @property
    def step_count(self) -> int:
        return self.export.time_grid.step_count

    @property
    def instruments(self) -> tuple[ExportInstrument, ...]:
        return self.export.instruments

    @property
    def observations(self) -> tuple[Observation, ...]:
        return self.export.observations


def _script_json(data: bytes) -> str:
    # JSON is inert data, but HTML still recognizes a closing script element.
    return (
        data.decode("utf-8").replace("<", r"\u003c").replace(">", r"\u003e").replace("&", r"\u0026")
    )


def _hash_source(value: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(value.encode()).digest()).decode() + "'"


def _number(value: float | int | None, *, percent: bool = False) -> str:
    if value is None:
        return "unknown"
    return f"{value * 100 if percent else value:.6g}"


def _interval_table(result: AnalysisResult) -> str:
    rows = []
    for sequence in result.sequences:
        for interval in sequence.intervals:
            stats = interval.statistics
            values = (
                sequence.instrument_id,
                f"[{interval.start}, {interval.stop})",
                interval.direction or interval.status,
                _number(stats.price_change_percent),
                _number(stats.return_std, percent=True),
                _number(stats.trade_count),
                _number(stats.trade_volume),
                _number(stats.trade_active_fraction, percent=True),
                _number(stats.price_coverage, percent=True),
                _number(stats.trade_count_coverage, percent=True),
                _number(stats.trade_volume_coverage, percent=True),
                str(stats.longest_no_trade_run),
            )
            rows.append(
                "<tr>"
                + "".join(f"<td>{html.escape(str(value))}</td>" for value in values)
                + "</tr>"
            )
    return "".join(rows)


def _provenance(export: ExportV2, result: AnalysisResult, result_sha256: str) -> str:
    values = [
        ("Source", f"{export.source.source_id} · {export.source.source_version}"),
        ("Snapshot", export.source.snapshot_id),
        ("Model", result.model.model_version_id),
        ("Analysis", str(result.analysis_id)),
        ("Input SHA-256", export.content_sha256),
        ("Result SHA-256", result_sha256),
    ]
    if export.source.run_id != export.source.snapshot_id:
        values.insert(2, ("Run", export.source.run_id))
    return "".join(
        f"<dt>{html.escape(name)}</dt><dd>{html.escape(value)}</dd>" for name, value in values
    )


def validate_analysis_result(
    export: ExportV2, result: AnalysisResult, *, result_sha256: str
) -> tuple[ExportV2, AnalysisResult]:
    """Validate a saved prediction against its exact original observations.

    Both report formats use this boundary before displaying any selected example.
    No labels, reference set, generator configuration or model fitting is read.
    """
    # Revalidation prevents a mutated nested dict or model_copy bypass from
    # converting an invalid provenance object into a trusted report.
    from regimex.application.service import AnalysisResult

    export_bytes = json_bytes(export.model_dump(mode="json"))
    result_bytes = json_bytes(result.model_dump(mode="json"))
    if len(export_bytes) > MAX_INPUT_BYTES or len(result_bytes) > MAX_INPUT_BYTES:
        raise ObjectCapacityError("analysis report inputs exceed their accepted size limits")
    export = parse_export_v2(export_bytes)
    result = AnalysisResult.model_validate_json(result_bytes)
    if (
        hashlib.sha256(result_bytes).hexdigest() != result_sha256
        or export.content_sha256 != result.snapshot_sha256
        or result.source != export.source
        or json_bytes(result.time_grid.model_dump(mode="json"))
        != json_bytes(export.time_grid.model_dump(mode="json"))
        or result.instruments != export.instruments
        or hashlib.sha256(json_bytes(result.inference_runtime.model_dump(mode="json"))).hexdigest()
        != result.model.inference_policy_sha256
    ):
        raise ObjectIntegrityError("report data, result and provenance identities differ")
    DirectionAnnotations(
        experiment_id=export.content_sha256,
        policy_config=result.analysis.model_dump(mode="json"),
        intervals=tuple(
            interval for sequence in result.sequences for interval in sequence.intervals
        ),
    ).validate_for(_Grid(export.content_sha256, export))
    return export, result


def render_analysis_report(
    export: ExportV2, result: AnalysisResult, *, result_sha256: str
) -> RenderedReport:
    """Render the whole requested snapshot with its verified saved predictions."""
    export, result = validate_analysis_result(export, result, result_sha256=result_sha256)
    # Plotly stays optional for CLI/API startup. Missing visualization capability
    # fails this worker attempt explicitly, before the result is published.
    try:
        # Plotly's offline bundle helper does not ship a typed signature.
        from plotly.offline import get_plotlyjs  # type: ignore[import-untyped]
    except ImportError:
        raise RegistryError("The analysis worker requires the visualization extra.") from None

    plotly_js = get_plotlyjs()
    plotly_version = version("plotly")
    if "</script" in plotly_js.lower():
        raise ObjectIntegrityError("embedded plotting runtime contains an unsafe HTML delimiter")
    resources = files("regimex").joinpath("templates")
    template = resources.joinpath("analysis-report.html").read_text()
    css = resources.joinpath("analysis-report.css").read_text()
    script = resources.joinpath("analysis-report.js").read_text()
    # Store repeated row keys and instrument identifiers once. This lossless
    # plain JSON representation keeps every accepted full-grid report bounded
    # without relying on browser decompression support.
    fields = tuple(ExportObservation.model_fields)
    fields = tuple(name for name in fields if name != "instrument_id")
    observations = []
    sequences = []
    for sequence in result.sequences:
        observations.append(
            {
                "instrument_id": sequence.instrument_id,
                "rows": [
                    [getattr(row, name) for name in fields]
                    for row in export.observations
                    if row.instrument_id == sequence.instrument_id
                ],
            }
        )
        item = sequence.model_dump(mode="json")
        for interval in item["intervals"]:
            interval.pop("instrument_id")
        sequences.append(item)
    payload = _script_json(
        json_bytes(
            {
                "export": export.model_dump(mode="json", exclude={"observations"}),
                "result": result.model_dump(mode="json", exclude={"sequences"}),
                "observation_fields": fields,
                "observations": observations,
                "sequences": sequences,
                "report": {
                    "renderer_version": REPORT_RENDERER_VERSION,
                    "plotly_version": plotly_version,
                    "result_sha256": result_sha256,
                    "selection": "whole requested completed snapshot",
                },
            }
        )
    )
    csp = (
        "default-src 'none'; base-uri 'none'; form-action 'none'; connect-src 'none'; "
        "img-src data: blob:; style-src 'unsafe-inline'; script-src "
        + " ".join(_hash_source(item) for item in (plotly_js, payload, script))
    )
    replacements = {
        "@@CSP@@": html.escape(csp, quote=True),
        "@@CSS@@": css,
        "@@TITLE@@": html.escape(str(result.analysis_id)),
        "@@PLOTLY@@": plotly_js,
        "@@PAYLOAD@@": payload,
        "@@SCRIPT@@": script,
        "@@INTERVALS@@": _interval_table(result),
        "@@PROVENANCE@@": _provenance(export, result, result_sha256),
    }
    # Substitute once: user identifiers may themselves contain template tokens.
    # Replacing sequentially could otherwise corrupt JSON or inject later assets.
    template = re.sub(r"@@[A-Z_]+@@", lambda match: replacements[match[0]], template)
    rendered = template.encode("utf-8")
    if len(rendered) > MAX_REPORT_BYTES:
        raise ObjectCapacityError("the complete offline report exceeds its 16 MiB object limit")
    return RenderedReport(rendered, REPORT_RENDERER_VERSION, plotly_version)
