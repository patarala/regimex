"""Portable offline reports for current source exports and independent human review."""

import json
from collections.abc import Mapping
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

from plotly.offline import get_plotlyjs  # type: ignore[import-untyped]

from regimex.data_catalog import DatasetManifest, SourceSnapshot
from regimex.directions import AnalysisConfig, summarize_interval
from regimex.exports import ExportV2
from regimex.inspection import (
    OBSERVABLE_FIELDS,
    REPORT_VERSION,
    Inspection,
    inspect_snapshots,
)
from regimex.storage import digest, output_directory, write_json

if TYPE_CHECKING:
    from regimex.application.service import AnalysisResult

REGIME_COLORS = {
    "growth": "#059669",
    "decline": "#e11d48",
    "sideways": "#64748b",
    "unclassified": "#cbd5e1",
}


def _script_json(value: object) -> str:
    # JSON is embedded in an HTML script element, whose parser recognizes </script>.
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _write_direction_assets(target: Path, payload: Mapping[str, object]) -> None:
    templates = files("regimex").joinpath("templates")
    (target / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    (target / "index.html").write_text(
        templates.joinpath("direction-report.html")
        .read_text(encoding="utf-8")
        .replace("__REPORT_JSON__", _script_json(payload)),
        encoding="utf-8",
    )
    for extension in ("js", "css"):
        (target / f"report.{extension}").write_text(
            templates.joinpath(f"direction-report.{extension}").read_text(encoding="utf-8"),
            encoding="utf-8",
        )


def _write_manifest(target: Path, *, kind: str, source_sha256: str) -> None:
    write_json(
        target / "report-manifest.json",
        {
            "version": "report-manifest/1.0",
            "report_version": REPORT_VERSION,
            "kind": kind,
            "source_sha256": source_sha256,
            "plotly_version": version("plotly"),
            "files": {
                path.name: digest(path) for path in sorted(target.iterdir()) if path.is_file()
            },
        },
    )


def render_snapshot_report(
    snapshots: tuple[SourceSnapshot, ...],
    output: Path,
    *,
    results: tuple[AnalysisResult, ...],
    dataset: DatasetManifest | None = None,
    sample_count: int = 12,
    experiment_id: str | None = None,
) -> Inspection:
    """Render saved model predictions after validating their original observations."""
    inspection = inspect_snapshots(
        snapshots,
        results=results,
        dataset=dataset,
        sample_count=sample_count,
        experiment_id=experiment_id,
    )
    payload = {
        "kind": "prediction",
        "summary": inspection.summary,
        "samples": [series.payload() for series in inspection.samples],
        "colors": REGIME_COLORS,
    }
    with output_directory(output) as target:
        _write_direction_assets(target, payload)
        write_json(target / "summary.json", inspection.summary)
        write_json(target / "examples.json", payload["samples"])
        write_json(target / "snapshots.json", [item.model_dump(mode="json") for item in snapshots])
        write_json(target / "results.json", [item.model_dump(mode="json") for item in results])
        _write_manifest(
            target,
            kind="snapshot-predictions",
            source_sha256=str(inspection.summary["source_content_sha256"]),
        )
    return inspection


def _reference_samples(export: ExportV2) -> list[dict[str, object]]:
    samples = []
    for instrument in export.instruments:
        rows = tuple(
            row for row in export.observations if row.instrument_id == instrument.instrument_id
        )
        samples.append(
            {
                "experiment_id": export.source.snapshot_id,
                "instrument_id": instrument.instrument_id,
                "base_asset": instrument.base_asset,
                "quote_asset": instrument.quote_asset,
                "initial_price": instrument.initial_price,
                "time_grid": export.time_grid.model_dump(mode="json"),
                "source": export.source.model_dump(mode="json"),
                "steps": [row.step for row in rows],
                "observations": {
                    field: [getattr(row, field) for row in rows] for field in OBSERVABLE_FIELDS
                },
                "availability": {
                    field: [getattr(row, field) for row in rows]
                    for field in ("price_state", "bid_state", "ask_state")
                },
                "intervals": [],
                "statistics": summarize_interval(rows, 0, len(rows)).model_dump(mode="json"),
            }
        )
    return samples


def review_reference(export: ExportV2, output: Path, config: AnalysisConfig | None = None) -> None:
    """Render original observations without model output, seeds, or automatic labels."""
    export = ExportV2.model_validate_json(export.model_dump_json())
    config = AnalysisConfig.model_validate_json((config or AnalysisConfig()).model_dump_json())
    payload: dict[str, object] = {
        "kind": "reference",
        "samples": _reference_samples(export),
        "experiment_hash": export.content_sha256,
        "analysis_config": config.model_dump(mode="json"),
        "colors": REGIME_COLORS,
    }
    with output_directory(output) as target:
        _write_direction_assets(target, payload)
        write_json(target / "export.json", export.model_dump(mode="json"))
        # Deliberately incomplete: a reviewer must supply decisions and interval statistics.
        write_json(
            target / "annotation-draft.json",
            {
                "schema_version": "3.0",
                "experiment_id": export.source.snapshot_id,
                "policy": "human-directions/1.0",
                "policy_config": config.model_dump(mode="json"),
                "reference_kind": "human",
                "reviewer": None,
                "intervals": [],
            },
        )
        _write_manifest(target, kind="reference-review", source_sha256=export.content_sha256)
