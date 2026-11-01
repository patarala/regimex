"""Offline full-grid observations, escaped provenance and strict saved identities."""

import hashlib
import json
import re
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip("plotly")

from regimex.application.report import render_analysis_report
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
from regimex.model_registry import InferenceRuntime
from regimex.operations import ResolvedModel
from regimex.report_contracts import MAX_REPORT_BYTES
from regimex.storage import json_bytes


def report_inputs(
    *, steps: int = 32, long_ids: bool = False, missing: bool = False
) -> tuple[ExportV2, AnalysisResult]:
    value = json.loads((Path(__file__).parent / "fixtures" / "experiment-v2.json").read_text())
    value.pop("content_sha256")
    value["time_grid"].update(step_count=steps, completed_horizon=100.0 + steps * 0.5)
    value["source"]["run_id"] = "</script><script>window.untrusted=true</script>"
    first = value["instruments"][0]
    first["instrument_id"] = "🚀" * 128 if long_ids else "A/Q"
    second = dict(first, instrument_id="🌙" * 128 if long_ids else "B/Q", base_asset="B")
    value["instruments"] = [first, second]
    template = value["observations"][0]
    value["observations"] = [
        dict(
            template,
            instrument_id=instrument["instrument_id"],
            step=step,
            last_price=100.0,
            price_state="observed",
            trade_count=1,
            trade_volume=1.0,
        )
        for instrument in value["instruments"]
        for step in range(steps)
    ]
    if missing:
        value["capabilities"]["historical_quotes"].update(
            mode="retained_snapshots",
            coverage="partial",
            provenance=(
                "Artificial report fixture with unknown, observed-empty and observed quote sides."
            ),
        )
        for row in value["observations"]:
            if row["step"] == 8:
                row.update(bid_state="empty", ask_state="empty")
            elif row["step"] == 9:
                row.update(
                    best_bid=99.0, best_ask=101.0, bid_state="observed", ask_state="observed"
                )
            if row["step"] in (5, 6):
                row.update(price_state="carried", trade_count=0, trade_volume=-0.0)
            elif row["step"] in (20, 21):
                row.update(
                    last_price=None, price_state="unknown", trade_count=None, trade_volume=None
                )
    export = seal_export(ExportPayloadV2.model_validate_json(json.dumps(value)))
    runtime = InferenceRuntime()
    policy_sha = hashlib.sha256(json_bytes(runtime.model_dump(mode="json"))).hexdigest()
    sequences = []
    for instrument in export.instruments:
        rows = tuple(
            row for row in export.observations if row.instrument_id == instrument.instrument_id
        )
        config = AnalysisConfig(min_duration=2, minimum_observations=2)
        blocks = (
            observation_blocks(rows, config)
            if missing
            else tuple((step, step + 2, "observed") for step in range(0, steps, 2))
        )
        # Adjacent labels differ so the largest report still has valid minimum-duration
        # intervals, rather than fragments that the interval contract must merge.
        intervals = tuple(
            DirectionInterval(
                instrument_id=instrument.instrument_id,
                start=start,
                stop=stop,
                direction=(Direction.SIDEWAYS, Direction.GROWTH)[index % 2]
                if status == "observed"
                else None,
                status=status,
                statistics=summarize_interval(rows, start, stop),
            )
            for index, (start, stop, status) in enumerate(blocks)
        )
        sequences.append(
            PredictionSequence(
                instrument_id=instrument.instrument_id,
                probabilities=tuple(
                    (0.01, 0.98, 0.01)
                    if interval.direction == Direction.GROWTH
                    else (0.98, 0.01, 0.01)
                    for interval in intervals
                    for _ in range(interval.start, interval.stop)
                ),
                intervals=intervals,
            )
        )
    result = AnalysisResult(
        analysis_id=uuid4(),
        snapshot_id=uuid4(),
        snapshot_sha256=export.content_sha256,
        source=export.source,
        model=ResolvedModel(
            model_version_id="report-fixture",
            bundle_sha256="a" * 64,
            inference_policy_sha256=policy_sha,
        ),
        architecture_id="report-fixture",
        implementation_version="1.0",
        decoder_version="duration-viterbi/1.0",
        analysis_profile_id="report-fixture",
        analysis=AnalysisConfig(min_duration=2, minimum_observations=2),
        inference_runtime=runtime,
        class_names=("sideways", "growth", "decline"),
        time_grid=export.time_grid,
        instruments=export.instruments,
        sequences=tuple(sequences),
    )
    return export, result


def digest(result: AnalysisResult) -> str:
    return hashlib.sha256(json_bytes(result.model_dump(mode="json"))).hexdigest()


def embedded_data(report: bytes) -> dict[str, object]:
    match = re.search(
        rb'<script id="analysis-data" type="application/json">(.*?)</script>', report, re.S
    )
    assert match is not None
    return json.loads(match[1])  # type: ignore[no-any-return]


def test_report_is_offline_escaped_and_preserves_original_zero_and_missing_values() -> None:
    export, result = report_inputs(missing=True)
    rendered = render_analysis_report(export, result, result_sha256=digest(result))
    document = rendered.html.decode()
    assert len(rendered.html) <= MAX_REPORT_BYTES
    assert rendered.renderer_version == "analysis-html/3.0"
    assert '<footer class="provenance"' in document
    assert "Reading the plots" not in document and '<pre id="provenance"' not in document
    assert "whole requested completed snapshot" in document
    assert "window.untrusted=true</script>" not in document
    assert "default-src &#x27;none&#x27;" in document and "connect-src &#x27;none&#x27;" in document
    assert not re.search(r'<(?:script|link)[^>]+(?:src|href)=["\']https?://', document)
    payload = embedded_data(rendered.html)
    sequences = payload["sequences"]
    assert isinstance(sequences, list)
    assert all("events" not in sequence for sequence in sequences)
    fields = payload["observation_fields"]
    blocks = payload["observations"]
    assert isinstance(fields, list) and isinstance(blocks, list)
    first = blocks[0]
    assert isinstance(first, dict)
    rows = first["rows"]
    assert isinstance(rows, list)
    assert rows[0][fields.index("buy_order_count")] == 0
    assert rows[0][fields.index("best_bid")] is None
    assert rows[0][fields.index("bid_state")] == "unknown"
    assert rows[5][fields.index("trade_count")] == 0
    assert rows[20][fields.index("last_price")] is None
    assert rows[8][fields.index("best_bid")] is None
    assert rows[8][fields.index("bid_state")] == "empty"
    assert len(rows) == export.time_grid.step_count
    provenance = payload["report"]
    assert isinstance(provenance, dict)
    assert provenance["result_sha256"] == digest(result)


def test_maximum_grid_with_long_unicode_identities_fits_without_browser_decompression() -> None:
    export, result = report_inputs(steps=4096, long_ids=True)
    rendered = render_analysis_report(export, result, result_sha256=digest(result))
    assert len(json_bytes(export.model_dump(mode="json"))) <= MAX_REPORT_BYTES
    assert len(rendered.html) <= MAX_REPORT_BYTES
    payload = embedded_data(rendered.html)
    blocks = payload["observations"]
    assert isinstance(blocks, list) and len(blocks) == 2
    assert (
        "DecompressionStream"
        not in (Path(__file__).parents[1] / "src/regimex/templates/analysis-report.js").read_text()
    )


@pytest.mark.parametrize("change", ["digest", "snapshot", "runtime", "statistics"])
def test_report_rejects_mismatched_saved_provenance_or_observed_statistics(change: str) -> None:
    export, result = report_inputs()
    value = result.model_dump(mode="json")
    if change == "snapshot":
        value["snapshot_sha256"] = "f" * 64
    elif change == "runtime":
        value["inference_runtime"]["threads"] = 3
    elif change == "statistics":
        value["sequences"][0]["intervals"][0]["statistics"]["trade_count"] = 200
    result = AnalysisResult.model_validate_json(json.dumps(value))
    with pytest.raises((ObjectIntegrityError, ValueError)):
        render_analysis_report(
            export, result, result_sha256="f" * 64 if change == "digest" else digest(result)
        )
