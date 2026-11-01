"""Completed snapshot identity, executed horizons, and unsupported-version rejection."""

import json
from copy import deepcopy
from decimal import localcontext
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from regimex.contracts import MAX_INPUT_BYTES
from regimex.exports import (
    ExportPayloadV2,
    ExportV2,
    ModelTimeGrid,
    load_export_v2,
    parse_export_v2,
    seal_export,
)


def payload() -> dict[str, Any]:
    return {
        "schema_version": "2.1",
        "source": {
            "source_id": "test-simulator",
            "source_version": "1.0",
            "configuration_id": "configuration-1",
            "run_id": "run-1",
            "snapshot_id": "snapshot-1",
            "state_revision": "revision-1",
            "run_status": "completed",
            "consistency": "atomic_completed_snapshot",
        },
        "producer": {"name": "fixture", "version": "1.0"},
        "time_grid": {
            "origin": 100.0,
            "unit": "model_second",
            "bin_width": 0.5,
            "completed_horizon": 101.0,
            "step_count": 2,
            "interval_bounds": "[start,stop)",
            "terminal_bin": "full_bins_only",
        },
        "capabilities": {
            "execution_prices": "partial",
            "trade_aggregates": "partial",
            "accepted_order_aggregates": "complete",
            "historical_quotes": {
                "mode": "unavailable",
                "coverage": "unavailable",
                "provenance": "The source retains no historical quote snapshots.",
                "complete_event_history_sha256": None,
                "reconstruction_policy": None,
            },
            "price_policy": "initial_then_carry",
            "trade_count_policy": "one_per_partial_execution",
            "order_count_policy": "accepted_submissions_including_immediate_fills",
            "rejected_order_policy": "excluded",
        },
        "instruments": [
            {
                "instrument_id": "A/Q",
                "base_asset": "A",
                "quote_asset": "Q",
                "initial_price": 100.0,
                "units": {
                    "price_unit": "quote_asset_per_base_asset",
                    "volume_unit": "base_asset",
                    "price_decimal_places": 2,
                    "volume_decimal_places": 3,
                },
            }
        ],
        "observations": [
            {
                "instrument_id": "A/Q",
                "step": step,
                "last_price": 100.0 if step == 0 else None,
                "trade_count": 0 if step == 0 else None,
                "trade_volume": 0.0 if step == 0 else None,
                "buy_order_count": 0,
                "buy_order_volume": 0.0,
                "sell_order_count": 0,
                "sell_order_volume": 0.0,
                "best_bid": None,
                "best_ask": None,
                "price_state": "carried" if step == 0 else "unknown",
                "bid_state": "unknown",
                "ask_state": "unknown",
            }
            for step in range(2)
        ],
    }


def validated(data: dict[str, Any]) -> ExportPayloadV2:
    return ExportPayloadV2.model_validate_json(json.dumps(data))


def test_v2_roundtrip_preserves_zero_missing_and_trailing_inactivity(tmp_path: Path) -> None:
    snapshot = seal_export(validated(payload()))
    target = tmp_path / "export.json"
    target.write_text(snapshot.model_dump_json(indent=2))
    restored = load_export_v2(target)
    assert restored == snapshot
    assert restored.observations[0].trade_count == 0
    assert restored.observations[1].trade_count is None
    assert restored.observations[0].price_state == "carried"
    assert restored.time_grid.completed_horizon == 101.0
    assert len(restored.observations) == restored.time_grid.step_count


def test_public_v2_fixture_loads() -> None:
    fixture = Path(__file__).parent / "fixtures" / "experiment-v2.json"
    assert load_export_v2(fixture).source.run_id == "run-1"


@pytest.mark.parametrize("version", ["1.0", "2.0", "3.0"])
def test_unsupported_exports_are_rejected_without_migration(version: str) -> None:
    data = payload()
    data["schema_version"] = version
    with pytest.raises(ValidationError, match="schema_version"):
        validated(data)


def test_removed_import_metadata_is_rejected() -> None:
    data = payload()
    data["v1_import"] = None
    with pytest.raises(ValidationError, match="Extra inputs"):
        validated(data)


def test_content_identity_ignores_json_layout_and_detects_metadata_changes() -> None:
    snapshot = seal_export(validated(payload()))
    data = snapshot.model_dump(mode="json")
    assert parse_export_v2(json.dumps(data, indent=4).encode()) == snapshot
    data["source"]["run_id"] = "rerun-2"
    with pytest.raises(ValidationError, match="SHA-256 mismatch"):
        parse_export_v2(json.dumps(data).encode())
    rerun = payload()
    rerun["source"]["run_id"] = "rerun-2"
    rerun["source"]["snapshot_id"] = "snapshot-2"
    assert seal_export(validated(rerun)).content_sha256 != snapshot.content_sha256
    with pytest.raises(ValidationError, match="frozen"):
        snapshot.source.run_id = "mutated"


@pytest.mark.parametrize("status", ["running", "cancelled", "stopped", "failed"])
def test_only_completed_atomic_runs_are_accepted(status: str) -> None:
    data = payload()
    data["source"]["run_status"] = status
    with pytest.raises(ValidationError, match="run_status"):
        validated(data)


@pytest.mark.parametrize(
    "change",
    [
        {"completed_horizon": 100.75},
        {"completed_horizon": 101.5},
        {"completed_horizon": 100.0},
        {"bin_width": 0.0},
        {"origin": float("nan")},
        {"step_count": 4097},
        {"interval_bounds": "[start,stop]"},
    ],
)
def test_grid_covers_exact_completed_horizon_without_truncation(change: dict[str, Any]) -> None:
    grid = payload()["time_grid"] | change
    with pytest.raises(ValidationError):
        ModelTimeGrid.model_validate_json(json.dumps(grid))


def test_explicit_partial_terminal_bin_and_decimal_boundaries() -> None:
    data = payload()
    data["time_grid"].update(terminal_bin="include_partial_bin", completed_horizon=100.75)
    assert validated(data).time_grid.step_count == 2
    data["time_grid"]["completed_horizon"] = 100.5
    with pytest.raises(ValidationError, match="terminal bin"):
        validated(data)
    grid = payload()["time_grid"] | {
        "origin": 0.0,
        "bin_width": 0.1,
        "completed_horizon": 0.3,
        "step_count": 3,
    }
    assert ModelTimeGrid.model_validate_json(json.dumps(grid)).step_count == 3


@pytest.mark.parametrize(
    "origin,terminal_bin",
    [(1e-30, "full_bins_only"), (-1e-30, "include_partial_bin")],
)
def test_grid_does_not_round_away_a_horizon_mismatch(origin: float, terminal_bin: str) -> None:
    grid = payload()["time_grid"] | {
        "origin": origin,
        "bin_width": 1.0,
        "completed_horizon": 1.0,
        "step_count": 1,
        "terminal_bin": terminal_bin,
    }
    with pytest.raises(ValidationError, match="completed horizon"):
        ModelTimeGrid.model_validate_json(json.dumps(grid))


def test_grid_validation_is_independent_of_the_callers_decimal_context() -> None:
    grid = payload()["time_grid"] | {
        "origin": 0.0,
        "bin_width": 0.125,
        "completed_horizon": 0.375,
        "step_count": 3,
    }
    with localcontext() as context:
        context.prec = 2
        assert ModelTimeGrid.model_validate_json(json.dumps(grid)).step_count == 3


@pytest.mark.parametrize("field", ["initial_price", "last_price", "trade_volume"])
def test_declared_precision_cannot_be_bypassed_by_decimal_context(field: str) -> None:
    data = payload()
    row = data["observations"][0]
    if field == "initial_price":
        data["instruments"][0][field] = 100.001
        row["last_price"] = 100.001
    else:
        row.update(trade_count=1, trade_volume=0.125, price_state="observed")
        row[field] = 0.1234 if field == "trade_volume" else 100.001
    with localcontext() as context:
        context.prec = 3
        with pytest.raises(ValidationError, match="numeric precision"):
            validated(data)


@pytest.mark.parametrize("case", ["duplicate", "missing", "unknown", "out_of_order", "too_many"])
def test_every_instrument_requires_an_ordered_complete_grid(case: str) -> None:
    data = payload()
    if case == "duplicate":
        data["observations"][1]["step"] = 0
    elif case == "missing":
        data["observations"].pop()
    elif case == "unknown":
        data["observations"][1]["instrument_id"] = "UNKNOWN"
    elif case == "out_of_order":
        data["observations"].reverse()
    else:
        data["instruments"] *= 3
    with pytest.raises(ValidationError):
        validated(data)


def test_two_instruments_share_the_grid_and_quote_asset() -> None:
    data = payload()
    instrument = deepcopy(data["instruments"][0])
    instrument.update(instrument_id="B/Q", base_asset="B")
    data["instruments"].append(instrument)
    rows = deepcopy(data["observations"])
    for row in rows:
        row["instrument_id"] = "B/Q"
    data["observations"].extend(rows)
    assert len(validated(data).observations) == 4
    data["instruments"][1]["quote_asset"] = "OTHER"
    with pytest.raises(ValidationError, match="share a quote"):
        validated(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("last_price", 100.001),
        ("trade_volume", 0.0001),
        ("price_state", "observed"),
        ("bid_state", "empty"),
        ("best_bid", 99.0),
        ("trade_count", True),
        ("seed", 11),
        ("reference_intervals", []),
    ],
)
def test_precision_missingness_and_observation_only_boundary(field: str, value: object) -> None:
    data = payload()
    data["observations"][0][field] = value
    if field == "trade_volume":
        data["observations"][0].update(trade_count=1, price_state="observed")
    with pytest.raises(ValidationError):
        validated(data)


def test_empty_historical_side_is_distinct_from_unknown_history() -> None:
    data = payload()
    history = data["capabilities"]["historical_quotes"]
    history.update(mode="retained_snapshots", coverage="partial", provenance="Retained bin closes.")
    data["observations"][0].update(best_bid=99.5, bid_state="observed", ask_state="empty")
    restored = seal_export(validated(data))
    assert restored.observations[0].ask_state == "empty"
    assert restored.observations[1].ask_state == "unknown"
    assert restored.observations[0].best_ask is restored.observations[1].best_ask is None
    history["coverage"] = "complete"
    with pytest.raises(ValidationError, match="unknown sides"):
        validated(data)


def test_quote_reconstruction_requires_complete_event_history_and_policy() -> None:
    data = payload()
    history = data["capabilities"]["historical_quotes"]
    history.update(mode="reconstructed_event_history", coverage="partial")
    with pytest.raises(ValidationError, match="event-history digest"):
        validated(data)
    history["complete_event_history_sha256"] = "a" * 64
    with pytest.raises(ValidationError, match="reconstruction policy"):
        validated(data)
    history["reconstruction_policy"] = "replay-full-order-events-v1"
    assert validated(data).capabilities.historical_quotes.mode == "reconstructed_event_history"


@pytest.mark.parametrize("coverage", ["complete", "unavailable"])
def test_aggregate_capabilities_cannot_hide_null_or_fabricate_known_values(coverage: str) -> None:
    data = payload()
    data["capabilities"]["trade_aggregates"] = coverage
    with pytest.raises(ValidationError, match="aggregate capabilities"):
        validated(data)


def test_carried_price_cannot_change_without_an_execution() -> None:
    data = payload()
    data["observations"][1].update(last_price=101.0, trade_count=0, price_state="carried")
    with pytest.raises(ValidationError, match="carried price differs"):
        validated(data)


@pytest.mark.parametrize("activity_field", ["trade_count", "trade_volume"])
@pytest.mark.parametrize("price_state", ["observed", "carried", "no_execution"])
def test_known_execution_totals_constrain_price_state_without_imputing_missing_totals(
    activity_field: str, price_state: str
) -> None:
    data = payload()
    if price_state == "no_execution":
        data["capabilities"]["price_policy"] = "execution_in_bin_only"
    row = data["observations"][0]
    row.update(
        trade_count=None,
        trade_volume=None,
        price_state=price_state,
        last_price=None if price_state == "no_execution" else 100.0,
    )
    requires_execution = price_state == "observed"
    known_total = 1 if requires_execution else 0
    row[activity_field] = known_total if activity_field == "trade_count" else float(known_total)
    result = seal_export(validated(data))
    counterpart = "trade_volume" if activity_field == "trade_count" else "trade_count"
    assert getattr(result.observations[0], counterpart) is None
    assert getattr(result.observations[0], activity_field) == known_total
    conflicting_total = 0 if requires_execution else 1
    row[activity_field] = (
        conflicting_total if activity_field == "trade_count" else float(conflicting_total)
    )
    with pytest.raises(ValidationError, match=r"requires.*execution"):
        validated(data)


@pytest.mark.parametrize("activity_field", ["trade_count", "trade_volume"])
def test_known_inactivity_preserves_price_history_when_other_totals_are_missing(
    activity_field: str,
) -> None:
    data = payload()
    data["time_grid"].update(step_count=3, completed_horizon=101.5)
    data["observations"][1][activity_field] = 0 if activity_field == "trade_count" else 0.0
    trailing = deepcopy(data["observations"][0])
    trailing.update(step=2, last_price=199.0)
    data["observations"].append(trailing)
    with pytest.raises(ValidationError, match="carried price differs"):
        validated(data)
    trailing["last_price"] = 100.0
    result = seal_export(validated(data))
    assert result.observations[1].last_price is None
    counterpart = "trade_volume" if activity_field == "trade_count" else "trade_count"
    assert getattr(result.observations[1], counterpart) is None
    assert result.observations[2].last_price == 100.0


def test_unknown_activity_may_hide_an_execution_and_change_later_carried_price() -> None:
    data = payload()
    data["time_grid"].update(step_count=3, completed_horizon=101.5)
    trailing = deepcopy(data["observations"][0])
    trailing.update(step=2, last_price=199.0)
    data["observations"].append(trailing)
    result = seal_export(validated(data))
    assert result.observations[1].trade_count is result.observations[1].trade_volume is None
    assert result.observations[2].last_price == 199.0


def test_partial_executions_and_trailing_inactivity_keep_distinct_order_totals() -> None:
    data = payload()
    data["time_grid"].update(step_count=3, completed_horizon=101.5)
    data["capabilities"].update(execution_prices="complete", trade_aggregates="complete")
    data["observations"][1].update(
        last_price=101.0,
        price_state="observed",
        trade_count=2,
        trade_volume=0.123,
        buy_order_count=1,
        buy_order_volume=0.123,
    )
    trailing = deepcopy(data["observations"][0])
    trailing.update(step=2, last_price=101.0)
    data["observations"].append(trailing)
    result = seal_export(validated(data))
    assert result.observations[1].trade_count == 2
    assert result.observations[1].buy_order_count == 1
    assert result.observations[2].trade_count == 0
    assert result.observations[2].last_price == 101.0


def test_execution_in_bin_prices_can_be_absent_without_being_unknown() -> None:
    data = payload()
    data["capabilities"].update(price_policy="execution_in_bin_only", execution_prices="complete")
    for row in data["observations"]:
        row.update(last_price=None, price_state="no_execution", trade_count=0, trade_volume=0.0)
    result = seal_export(validated(data))
    assert all(row.price_state == "no_execution" for row in result.observations)
    assert all(row.last_price is None for row in result.observations)


def test_maximum_grid_is_preserved_without_resampling() -> None:
    data = payload()
    data["time_grid"].update(step_count=4096, completed_horizon=2148.0)
    template = data["observations"][0]
    data["observations"] = [template | {"step": step} for step in range(4096)]
    result = seal_export(validated(data))
    assert len(result.observations) == 4096
    assert result.observations[-1].step == 4095


def test_bounds_precede_parsing_and_no_automatic_version_fallback(tmp_path: Path) -> None:
    oversized = b"!" * (MAX_INPUT_BYTES + 1)
    with pytest.raises(ValueError, match="size limit"):
        parse_export_v2(oversized)
    target = tmp_path / "oversized.json"
    target.write_bytes(oversized)
    with pytest.raises(ValueError, match="size limit"):
        load_export_v2(target)
    data = payload()
    data["schema_version"] = "3.0"
    with pytest.raises(ValidationError, match="schema_version"):
        validated(data)


def test_export_schema_contains_no_supervision_or_generator_inputs() -> None:
    schema = ExportV2.model_json_schema()
    assert set(schema["properties"]) == {
        "schema_version",
        "source",
        "producer",
        "time_grid",
        "capabilities",
        "instruments",
        "observations",
        "content_sha256",
    }
    assert schema["additionalProperties"] is False
