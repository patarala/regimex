"""Known inactivity remains separate from missing execution totals and prices."""

import json
from pathlib import Path

import pytest

from regimex.contracts import Observation
from regimex.data_catalog import SnapshotGrid, analysis_profile_hash
from regimex.directions import AnalysisConfig, DirectionAnnotations, observation_blocks
from regimex.exports import ExportPayloadV2, ExportV2, parse_export_v2, seal_export
from regimex.observation_availability import known_no_executions

np = pytest.importorskip("numpy")
segmentation = pytest.importorskip("regimex.segmentation")


def inactive_export(
    *, count: int | None, volume: float | None, state: str, steps: int = 32
) -> ExportV2:
    payload = json.loads((Path(__file__).parent / "fixtures" / "experiment-v2.json").read_bytes())
    payload.pop("content_sha256")
    payload["time_grid"].update(step_count=steps, completed_horizon=100.0 + steps * 0.5)
    payload["capabilities"].update(
        trade_aggregates="partial",
        price_policy="execution_in_bin_only" if state == "no_execution" else "initial_then_carry",
    )
    template = payload["observations"][0]
    payload["observations"] = [
        dict(
            template,
            step=step,
            trade_count=count,
            trade_volume=volume,
            price_state=state,
            last_price=100.0 if state in ("carried", "observed") else None,
        )
        for step in range(steps)
    ]
    export = seal_export(ExportPayloadV2.model_validate_json(json.dumps(payload)))
    return parse_export_v2(export.model_dump_json().encode())


@pytest.mark.parametrize(
    "count,volume,state,current,previous",
    [
        (None, 0.0, "carried", "no_trades", "observed"),
        (None, None, "carried", "no_trades", "observed"),
        (None, None, "no_execution", "no_trades", "insufficient_data"),
        (None, 0.0, "unknown", "no_trades", "insufficient_data"),
        (0, None, "unknown", "no_trades", "insufficient_data"),
        (None, None, "unknown", "insufficient_data", "insufficient_data"),
        (None, None, "observed", "observed", "observed"),
    ],
)
def test_sealed_exports_have_versioned_availability_without_imputing_counts(
    count: int | None, volume: float | None, state: str, current: str, previous: str
) -> None:
    export = inactive_export(count=count, volume=volume, state=state)
    grid = SnapshotGrid(
        experiment_id=export.content_sha256,
        step_count=export.time_grid.step_count,
        instruments=export.instruments,
        observations=export.observations,
    )
    for config, expected in (
        (AnalysisConfig(), current),
        (AnalysisConfig(version="2.0"), previous),
    ):
        assert observation_blocks(export.observations, config) == ((0, 32, expected),)
        annotation = segmentation.annotate(grid, config)
        restored = DirectionAnnotations.model_validate_json(annotation.model_dump_json())
        restored.validate_for(grid)
        predicted = segmentation.decode(
            np.tile([0.0, 1.0, 0.0], (32, 1)), export.observations, config
        )
        assert restored.intervals[0].status == predicted[0].status == expected
        assert (predicted[0].direction is None) == (expected != "observed")
        assert segmentation.eligible_mask(export.observations, config).sum() == (
            32 if expected == "observed" else 0
        )
        statistics = predicted[0].statistics
        assert statistics.trade_count == (None if count is None else 32 * count)
        assert statistics.trade_count_coverage == (0 if count is None else 1)
        assert statistics.trade_volume == (None if volume is None else 32 * volume)
        if count is None:
            assert statistics.trade_active_fraction is None
    assert all(
        row.trade_count == count and row.trade_volume == volume for row in export.observations
    )


def test_plain_observations_recognize_zero_volume_without_export_states() -> None:
    export = inactive_export(count=None, volume=0.0, state="carried")
    rows = tuple(
        Observation.model_validate(
            row.model_dump(exclude={"price_state", "bid_state", "ask_state"})
        )
        for row in export.observations
    )
    assert all(known_no_executions(row) for row in rows)
    assert observation_blocks(rows, AnalysisConfig()) == ((0, 32, "no_trades"),)
    unknown = rows[0].model_copy(update={"trade_volume": None})
    assert not known_no_executions(unknown)


def test_short_inactivity_stays_inside_observed_intervals() -> None:
    active = inactive_export(count=1, volume=1.0, state="observed", steps=64)
    inactive = inactive_export(count=None, volume=None, state="carried", steps=64)
    rows = active.observations[:20] + inactive.observations[20:24] + active.observations[24:]
    assert observation_blocks(rows, AnalysisConfig()) == ((0, 64, "observed"),)
    assert all(known_no_executions(row) for row in rows[20:24])


def test_profile_versions_are_distinct_and_roundtrip_without_rewriting() -> None:
    previous = AnalysisConfig.model_validate_json('{"version":"2.0"}')
    current = AnalysisConfig()
    profiles = (previous, AnalysisConfig(version="2.1"), current)
    assert current.version == "2.2"
    assert len({analysis_profile_hash(profile) for profile in profiles}) == 3
    for profile in profiles:
        assert AnalysisConfig.model_validate_json(profile.model_dump_json()) == profile
