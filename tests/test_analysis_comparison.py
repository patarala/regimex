"""Saved comparisons preserve input identity, availability and offline behavior."""

import hashlib
import json
import re
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from support.report_fixtures import prediction_for

from regimex.application.analysis_comparison import render_analysis_comparison
from regimex.application.service import AnalysisResult, PredictionSequence
from regimex.directions import (
    AnalysisConfig,
    Direction,
    DirectionInterval,
    observation_blocks,
    summarize_interval,
)
from regimex.exports import ExportPayloadV2, ExportV2, seal_export
from regimex.infrastructure.object_store import ObjectIntegrityError
from regimex.report_contracts import MAX_REPORT_BYTES
from regimex.storage import json_bytes


def _export(*, missing: bool = False, steps: int = 32) -> ExportV2:
    value = json.loads((Path(__file__).parent / "fixtures" / "experiment-v2.json").read_text())
    value.pop("content_sha256")
    value["source"]["run_id"] = "</script><script>window.untrusted=true</script>"
    value["time_grid"].update(step_count=steps, completed_horizon=100.0 + steps * 0.5)
    value["instruments"].append(dict(value["instruments"][0], instrument_id="B/Q", base_asset="B"))
    template = value["observations"][0]
    value["observations"] = [
        dict(
            template,
            instrument_id=instrument["instrument_id"],
            step=step,
            last_price=100.0 + step / 10,
            price_state="observed",
            trade_count=1,
            trade_volume=1.0,
        )
        for instrument in value["instruments"]
        for step in range(steps)
    ]
    if missing:
        for row in value["observations"]:
            if row["step"] in (8, 9):
                row.update(
                    last_price=None, price_state="unknown", trade_count=None, trade_volume=None
                )
            if row["step"] in (14, 15):
                row.update(
                    last_price=101.3, price_state="carried", trade_count=0, trade_volume=-0.0
                )
    return seal_export(ExportPayloadV2.model_validate_json(json.dumps(value)))


def _digest(result: AnalysisResult) -> str:
    return hashlib.sha256(json_bytes(result.model_dump(mode="json"))).hexdigest()


def _render(export: ExportV2, first: AnalysisResult, second: AnalysisResult) -> bytes:
    return render_analysis_comparison(
        export, first, second, first_sha256=_digest(first), second_sha256=_digest(second)
    ).html


def _payload(report: bytes) -> dict[str, Any]:
    match = re.search(
        rb'<script id="comparison-data" type="application/json">(.*?)</script>', report, re.S
    )
    assert match is not None
    return cast(dict[str, Any], json.loads(match[1]))


def _with_intervals(
    result: AnalysisResult,
    export: ExportV2,
    partitions: tuple[tuple[int, int, Direction], ...],
) -> AnalysisResult:
    sequences = []
    for instrument in export.instruments:
        rows = tuple(
            row for row in export.observations if row.instrument_id == instrument.instrument_id
        )
        sequences.append(
            PredictionSequence(
                instrument_id=instrument.instrument_id,
                probabilities=((0.2, 0.4, 0.4),) * len(rows),
                intervals=tuple(
                    DirectionInterval(
                        instrument_id=instrument.instrument_id,
                        start=start,
                        stop=stop,
                        direction=direction,
                        status="observed",
                        statistics=summarize_interval(rows, start, stop),
                    )
                    for start, stop, direction in partitions
                ),
            )
        )
    return result.model_copy(update={"sequences": tuple(sequences)})


def test_comparison_is_offline_and_preserves_missing_carried_prices_and_safe_labels() -> None:
    export = _export(missing=True)
    first = prediction_for(export, direction=Direction.GROWTH)
    second = prediction_for(export, direction=Direction.DECLINE)
    first = first.model_copy(
        update={
            "model": first.model.model_copy(update={"model_version_id": "@@SCRIPT@@ </script>"})
        }
    )
    report = _render(export, first, second)
    document = report.decode()
    payload = _payload(report)
    assert payload["renderer_version"] == "analysis-comparison/1.0"
    assert payload["snapshot_id"] == str(first.snapshot_id)
    assert payload["series"][0]["prices"][8:10] == [None, None]
    assert payload["series"][0]["prices"][14:16] == [101.3, 101.3]
    assert payload["series"][0]["price_states"][14:16] == ["carried", "carried"]
    assert payload["analyses"][0]["model"]["model_version_id"] == "@@SCRIPT@@ </script>"
    assert "window.untrusted=true</script>" not in document
    assert "connect-src &#x27;none&#x27;" in document
    assert not re.search(r'<(?:script|link)[^>]+(?:src|href)=["\']https?://', document)
    notices = re.search(r"<!-- Software licenses\n(.*?)\n  -->", document, re.S)
    assert notices is not None
    license_text = distribution("regimex").read_text("licenses/LICENSE")
    assert license_text and license_text in notices[1]
    assert "# Plotly" not in notices[1]
    script = (
        Path(__file__).parents[1] / "src/regimex/templates/analysis-comparison.js"
    ).read_bytes()
    assert b"<script>" + script + b"</script>" in report
    assert "Differences do not measure accuracy." in document
    assert len(report) < MAX_REPORT_BYTES


def test_difference_ranges_follow_original_boundaries_without_fragmenting_equal_states() -> None:
    export = _export()
    first = _with_intervals(
        prediction_for(export),
        export,
        ((0, 16, Direction.GROWTH), (16, 32, Direction.DECLINE)),
    )
    second = _with_intervals(
        prediction_for(export, direction=Direction.SIDEWAYS),
        export,
        ((0, 8, Direction.GROWTH), (8, 24, Direction.SIDEWAYS), (24, 32, Direction.DECLINE)),
    )
    payload = _payload(_render(export, first, second))
    assert payload["series"][0]["differences"] == [
        {"start": 8, "stop": 16, "first": "growth", "second": "sideways", "kind": "direction"},
        {"start": 16, "stop": 24, "first": "decline", "second": "sideways", "kind": "direction"},
    ]


def test_distinct_runs_of_the_same_model_can_agree_completely() -> None:
    export = _export()
    first = prediction_for(export)
    second = first.model_copy(update={"analysis_id": uuid4()})
    payload = _payload(_render(export, first, second))
    assert all(series["differences"] == [] for series in payload["series"])


def test_comparison_distinguishes_availability_policies_from_direction_disagreement() -> None:
    export = _export(missing=True)
    first = prediction_for(export)
    config = AnalysisConfig(min_duration=4, minimum_price_coverage=1.0)
    sequences = []
    for instrument in export.instruments:
        rows = tuple(
            row for row in export.observations if row.instrument_id == instrument.instrument_id
        )
        sequences.append(
            PredictionSequence(
                instrument_id=instrument.instrument_id,
                probabilities=((0.2, 0.4, 0.4),) * len(rows),
                intervals=tuple(
                    DirectionInterval(
                        instrument_id=instrument.instrument_id,
                        start=start,
                        stop=stop,
                        direction=Direction.GROWTH if status == "observed" else None,
                        status=status,
                        statistics=summarize_interval(rows, start, stop),
                    )
                    for start, stop, status in observation_blocks(rows, config)
                ),
            )
        )
    second = first.model_copy(
        update={"analysis_id": uuid4(), "analysis": config, "sequences": tuple(sequences)}
    )
    payload = _payload(_render(export, first, second))
    assert payload["series"][0]["differences"] == [
        {
            "start": 0,
            "stop": 32,
            "first": "growth",
            "second": "insufficient_data",
            "kind": "availability",
        }
    ]


@pytest.mark.parametrize(
    "change", ["digest", "snapshot", "content", "grid", "statistics", "same_analysis"]
)
def test_comparison_rejects_invalid_or_unrelated_saved_results(change: str) -> None:
    export = _export()
    first = prediction_for(export)
    value = prediction_for(export, direction=Direction.DECLINE).model_dump(mode="json")
    if change == "snapshot":
        value["snapshot_id"] = str(uuid4())
    elif change == "content":
        value["snapshot_sha256"] = "f" * 64
    elif change == "grid":
        value["time_grid"]["origin"] += 1
        value["time_grid"]["completed_horizon"] += 1
    elif change == "statistics":
        value["sequences"][0]["intervals"][0]["statistics"]["trade_count"] = 100000
    elif change == "same_analysis":
        value["analysis_id"] = str(first.analysis_id)
    second = AnalysisResult.model_validate_json(json.dumps(value))
    with pytest.raises((ObjectIntegrityError, ValueError)):
        render_analysis_comparison(
            export,
            first,
            second,
            first_sha256=_digest(first),
            second_sha256="f" * 64 if change == "digest" else _digest(second),
        )


def test_maximum_two_instrument_grid_has_a_bounded_complete_comparison() -> None:
    export = _export(steps=4096)
    first = prediction_for(export)
    second = prediction_for(export, direction=Direction.DECLINE)
    report = _render(export, first, second)
    payload = _payload(report)
    assert len(report) < MAX_REPORT_BYTES
    assert len(payload["series"]) == 2
    assert all(len(series["prices"]) == 4096 for series in payload["series"])
    assert all(series["differences"][0]["stop"] == 4096 for series in payload["series"])


def test_comparison_rendering_needs_no_optional_scientific_or_plotting_dependencies(
    tmp_path: Path,
) -> None:
    export = _export()
    first = prediction_for(export)
    second = prediction_for(export, direction=Direction.DECLINE)
    source = tmp_path / "inputs.json"
    source.write_bytes(
        json_bytes(
            {
                "export": export.model_dump(mode="json"),
                "first": first.model_dump(mode="json"),
                "second": second.model_dump(mode="json"),
                "first_sha256": _digest(first),
                "second_sha256": _digest(second),
            }
        )
    )
    script = """
import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'torch', 'sklearn', 'plotly'}:
            raise AssertionError('optional dependency loaded: ' + fullname)
sys.meta_path.insert(0, Block())
from regimex.application.analysis_comparison import render_analysis_comparison
from regimex.application.service import AnalysisResult
from regimex.exports import ExportV2
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text())
rendered = render_analysis_comparison(
    ExportV2.model_validate_json(json.dumps(data['export'])),
    AnalysisResult.model_validate_json(json.dumps(data['first'])),
    AnalysisResult.model_validate_json(json.dumps(data['second'])),
    first_sha256=data['first_sha256'], second_sha256=data['second_sha256'],
)
assert b'analysis-comparison/1.0' in rendered.html
"""
    subprocess.run([sys.executable, "-c", script, str(source)], check=True, timeout=15)
