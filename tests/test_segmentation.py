"""Direction semantics, honest observation coverage, and interval decoding."""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from regimex.contracts import Instrument, Observation
from regimex.directions import (
    AnalysisConfig,
    Direction,
    DirectionAnnotations,
    DirectionInterval,
    observation_blocks,
    summarize_interval,
)

np = pytest.importorskip("numpy")
segmentation = pytest.importorskip("regimex.segmentation")


@dataclass(frozen=True)
class AnalyticGrid:
    experiment_id: str
    step_count: int
    instruments: tuple[Instrument, ...]
    observations: tuple[Observation, ...]


def experiment(
    prices: Sequence[float | None], counts: Sequence[int | None] | None = None
) -> AnalyticGrid:
    counts = counts if counts is not None else [1] * len(prices)
    return AnalyticGrid(
        experiment_id="analytic",
        step_count=len(prices),
        instruments=(
            Instrument(instrument_id="A/Q", base_asset="A", quote_asset="Q", initial_price=100.0),
        ),
        observations=tuple(
            Observation(
                instrument_id="A/Q",
                step=step,
                last_price=price,
                trade_count=count,
                trade_volume=float(count) if count is not None else None,
                buy_order_count=None,
                buy_order_volume=None,
                sell_order_count=None,
                sell_order_volume=None,
                best_bid=None,
                best_ask=None,
            )
            for step, (price, count) in enumerate(zip(prices, counts, strict=True))
        ),
    )


@pytest.mark.parametrize("factor,direction", [(1.1, Direction.GROWTH), (0.9, Direction.DECLINE)])
def test_slow_sustained_movement_covers_the_whole_experiment(
    factor: float, direction: Direction
) -> None:
    data = experiment([100 * math.exp(math.log(factor) * step / 511) for step in range(512)])
    annotations = segmentation.annotate(data)
    assert len(annotations.intervals) == 1
    interval = annotations.intervals[0]
    assert (interval.start, interval.stop, interval.direction) == (0, 512, direction)
    assert interval.statistics.price_change_percent == pytest.approx(100 * (factor - 1))
    assert annotations.reference_kind == "automatic"


@pytest.mark.parametrize("noisy", [False, True])
def test_flat_and_oscillating_flat_prices_have_one_sideways_interval(noisy: bool) -> None:
    data = experiment(
        [100 * math.exp(0.01 * math.sin(step / 2)) if noisy else 100.0 for step in range(512)]
    )
    annotations = segmentation.annotate(data)
    assert len(annotations.intervals) == 1
    assert annotations.intervals[0].direction == Direction.SIDEWAYS
    assert annotations.intervals[0].statistics.return_std is not None
    if noisy:
        assert annotations.intervals[0].statistics.return_std > 0.001


def test_reversal_has_two_sustained_directions_and_no_tiny_fragments() -> None:
    data = experiment([100 * math.exp(0.001 * min(step, 511 - step)) for step in range(512)])
    annotations = segmentation.annotate(data)
    assert [item.direction for item in annotations.intervals] == [
        Direction.GROWTH,
        Direction.DECLINE,
    ]
    assert abs(annotations.intervals[0].stop - 256) <= 2
    assert all(item.stop - item.start >= 32 for item in annotations.intervals)


def test_growth_then_plateau_preserves_both_interpretations() -> None:
    data = experiment([100 * math.exp(0.001 * min(step, 255)) for step in range(512)])
    annotations = segmentation.annotate(data)
    assert [item.direction for item in annotations.intervals] == [
        Direction.GROWTH,
        Direction.SIDEWAYS,
    ]
    assert abs(annotations.intervals[0].stop - 256) <= 2


def test_isolated_level_change_does_not_create_a_one_step_direction() -> None:
    data = experiment([100.0] * 256 + [120.0] * 256)
    annotations = segmentation.annotate(data)
    assert len(annotations.intervals) == 1
    assert annotations.intervals[0].direction == Direction.SIDEWAYS
    assert annotations.intervals[0].statistics.price_change_percent == pytest.approx(20)


@pytest.mark.parametrize("length", [16, 32, 48, 63, 64, 96])
@pytest.mark.parametrize("offset", [-0.2, 0.2])
def test_short_level_offset_does_not_supply_sustained_direction(length: int, offset: float) -> None:
    data = experiment([100 * math.exp(offset * (step >= length // 2)) for step in range(length)])
    result = segmentation.annotate(data)
    assert [(item.start, item.stop, item.direction) for item in result.intervals] == [
        (0, length, Direction.SIDEWAYS)
    ]
    assert result.intervals[0].statistics.price_change_percent == pytest.approx(
        100 * math.expm1(offset)
    )


@pytest.mark.parametrize("length", [16, 32, 48, 63, 64, 96])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("offset", [-0.2, 0.0, 0.2])
def test_short_sustained_drift_survives_an_independent_level_offset(
    length: int, sign: int, offset: float
) -> None:
    data = experiment(
        [
            100 * math.exp(sign * 0.01 * step / (length - 1) + offset * (step >= length // 2))
            for step in range(length)
        ]
    )
    result = segmentation.annotate(data)
    assert [(item.start, item.stop, item.direction) for item in result.intervals] == [
        (0, length, Direction.GROWTH if sign > 0 else Direction.DECLINE)
    ]


@pytest.mark.parametrize("length", [16, 32, 48, 63, 64, 96])
def test_short_reversal_does_not_invent_a_net_direction(length: int) -> None:
    data = experiment(
        [
            100 * math.exp(0.1 * min(step, length - 1 - step) / (length - 1))
            for step in range(length)
        ]
    )
    result = segmentation.annotate(data)
    expected = [Direction.SIDEWAYS] if length < 64 else [Direction.GROWTH, Direction.DECLINE]
    assert [item.direction for item in result.intervals] == expected
    result.validate_for(data)


@pytest.mark.parametrize("length", [16, 32, 48, 63, 64, 96])
@pytest.mark.parametrize("sign", [-1, 0, 1])
def test_short_noisy_drift_and_oscillating_plateau(length: int, sign: int) -> None:
    data = experiment(
        [
            100 * math.exp(sign * 0.01 * step / (length - 1) + 0.001 * math.sin(step * 1.23))
            for step in range(length)
        ]
    )
    result = segmentation.annotate(data)
    direction = (
        Direction.GROWTH if sign > 0 else Direction.DECLINE if sign < 0 else Direction.SIDEWAYS
    )
    assert [(item.start, item.stop, item.direction) for item in result.intervals] == [
        (0, length, direction)
    ]


@pytest.mark.parametrize("price_scale", [0.01, 100.0, 1000000.0])
@pytest.mark.parametrize("movement", [-0.01, 0.0, 0.01])
def test_level_offset_evidence_preserves_timestamps_missingness_and_price_units(
    price_scale: float, movement: float
) -> None:
    prices = [
        price_scale * math.exp(movement * step / 47 + 0.2 * (step >= 24)) if step % 7 != 3 else None
        for step in range(48)
    ]
    data = experiment([None] * 64 + prices + [None] * 64)
    result = segmentation.annotate(data)
    expected = (
        Direction.GROWTH
        if movement > 0
        else Direction.DECLINE
        if movement < 0
        else Direction.SIDEWAYS
    )
    assert [(item.start, item.stop, item.direction) for item in result.intervals] == [
        (0, 64, None),
        (64, 112, expected),
        (112, 176, None),
    ]
    assert result.intervals[1].statistics.price_coverage == pytest.approx(41 / 48)
    result.validate_for(data)


def drift_with_level_offset(drift_per_step: float, jump_factor: float) -> AnalyticGrid:
    """A smooth log-price slope with one independent midpoint level offset."""
    return experiment(
        [
            100 * math.exp(drift_per_step * step) * (jump_factor if step >= 256 else 1)
            for step in range(512)
        ]
    )


@pytest.mark.parametrize("drift_per_step", [-0.000001, 0.000001])
@pytest.mark.parametrize("jump_factor", [0.8, 1.2])
def test_level_offset_does_not_turn_subthreshold_drift_into_direction(
    drift_per_step: float, jump_factor: float
) -> None:
    config = AnalysisConfig()
    assert abs(math.expm1(drift_per_step * 511)) < config.minimum_relative_change
    for data in (
        drift_with_level_offset(drift_per_step, 1.0),
        drift_with_level_offset(0.0, jump_factor),
        drift_with_level_offset(drift_per_step, jump_factor),
    ):
        annotations = segmentation.annotate(data, config)
        assert [(item.start, item.stop, item.direction) for item in annotations.intervals] == [
            (0, 512, Direction.SIDEWAYS)
        ]


@pytest.mark.parametrize(
    "drift_per_step, jump_factor, direction",
    [(0.0001, 0.8, Direction.GROWTH), (-0.0001, 1.2, Direction.DECLINE)],
)
def test_opposing_level_offset_does_not_reverse_supported_drift(
    drift_per_step: float, jump_factor: float, direction: Direction
) -> None:
    annotations = segmentation.annotate(drift_with_level_offset(drift_per_step, jump_factor))
    assert [(item.start, item.stop, item.direction) for item in annotations.intervals] == [
        (0, 512, direction)
    ]
    change = annotations.intervals[0].statistics.price_change_percent
    assert change is not None
    assert change * drift_per_step < 0


@pytest.mark.parametrize("drift_per_step", [-0.000001, 0.0, 0.000001])
@pytest.mark.parametrize("offset", [-0.1, 0.1])
def test_repeated_resolved_level_offsets_do_not_accumulate_as_trend(
    drift_per_step: float, offset: float
) -> None:
    data = experiment(
        [100 * math.exp(drift_per_step * step + offset * (step // 128)) for step in range(512)]
    )
    annotations = segmentation.annotate(data)
    assert [(item.start, item.stop, item.direction) for item in annotations.intervals] == [
        (0, 512, Direction.SIDEWAYS)
    ]


@pytest.mark.parametrize(
    "drift_per_step, direction", [(0.000015, Direction.GROWTH), (-0.000015, Direction.DECLINE)]
)
def test_supported_movement_accumulates_across_multiple_fitted_levels(
    drift_per_step: float, direction: Direction
) -> None:
    """Each 128-bin slope is subthreshold; their sustained sum exceeds 0.5%."""
    config = AnalysisConfig()
    assert abs(math.expm1(drift_per_step * 127)) < config.minimum_relative_change
    assert abs(math.expm1(drift_per_step * 508)) > config.minimum_relative_change
    data = experiment(
        [100 * math.exp(drift_per_step * step + 0.2 * ((step // 128) % 2)) for step in range(512)]
    )
    annotations = segmentation.annotate(data, config)
    assert [(item.start, item.stop, item.direction) for item in annotations.intervals] == [
        (0, 512, direction)
    ]


@pytest.mark.parametrize("sign", [-1, 1])
def test_flat_piece_breaks_accumulation_of_subthreshold_slopes(sign: int) -> None:
    data = experiment(
        [
            100
            * math.exp(
                sign
                * (
                    0.00003 * min(step, 127)
                    + 0.2 * (step >= 128)
                    + 0.2 * (step >= 384)
                    + 0.00003 * max(step - 384, 0)
                )
            )
            for step in range(512)
        ]
    )
    annotations = segmentation.annotate(data)
    assert [(item.start, item.stop, item.direction) for item in annotations.intervals] == [
        (0, 512, Direction.SIDEWAYS)
    ]


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("price_scale", [0.01, 100.0, 1000000.0])
def test_noisy_drift_with_level_offset_is_invariant_to_price_units(
    sign: int, price_scale: float
) -> None:
    data = experiment(
        [
            price_scale
            * math.exp(sign * 0.0001 * step + 0.001 * math.sin(step * 1.23) - 0.2 * (step >= 256))
            for step in range(512)
        ]
    )
    annotations = segmentation.annotate(data)
    assert [(item.start, item.stop, item.direction) for item in annotations.intervals] == [
        (0, 512, Direction.GROWTH if sign > 0 else Direction.DECLINE)
    ]


@pytest.mark.parametrize("settings", [{"version": "1.0"}, {"jump_threshold": 0.025}])
def test_removed_price_event_configuration_is_rejected(settings: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AnalysisConfig.model_validate(settings)


@pytest.mark.parametrize("change", ["previous_version", "events"])
def test_previous_annotation_shape_is_rejected(change: str) -> None:
    annotations = segmentation.annotate(experiment([100.0] * 32))
    payload = annotations.model_dump(mode="json")
    assert payload["schema_version"] == "3.0"
    assert "events" not in payload
    if change == "previous_version":
        payload["schema_version"] = "2.0"
    else:
        payload["events"] = []
    with pytest.raises(ValidationError):
        DirectionAnnotations.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("previous", ["1.0", "2.0"])
def test_previous_automatic_policy_is_not_silently_reinterpreted(previous: str) -> None:
    payload = segmentation.annotate(experiment([100.0] * 32)).model_dump(mode="json")
    assert payload["policy"] == "retrospective-directions/3.0"
    payload["policy"] = f"retrospective-directions/{previous}"
    with pytest.raises(ValidationError, match="automatic policy"):
        DirectionAnnotations.model_validate_json(json.dumps(payload))


def test_known_no_trades_are_not_observed_sideways_or_missing_counts() -> None:
    no_trades = segmentation.annotate(experiment([100.0] * 64, [0] * 64)).intervals[0]
    unknown = segmentation.annotate(experiment([100.0] * 64, [None] * 64)).intervals[0]
    missing = segmentation.annotate(experiment([None] * 64, [None] * 64)).intervals[0]
    assert no_trades.status == "no_trades" and no_trades.direction is None
    assert no_trades.statistics.trade_count == 0
    assert no_trades.statistics.trade_active_fraction == 0
    assert unknown.status == "observed" and unknown.direction == Direction.SIDEWAYS
    assert unknown.statistics.trade_count is None
    assert unknown.statistics.trade_active_fraction is None
    assert unknown.statistics.trade_count_coverage == 0
    assert missing.status == "insufficient_data" and missing.direction is None
    assert missing.statistics.price_change_percent is None
    assert missing.statistics.return_std is None


def test_short_pauses_and_missing_values_reduce_coverage_without_fragmentation() -> None:
    prices: list[float | None] = [100 + step / 100 for step in range(128)]
    prices[30] = None
    prices[31] = None
    counts: list[int | None] = [1] * 128
    counts[70:72] = [0, 0]
    data = experiment(prices, counts)
    annotations = segmentation.annotate(data)
    assert len(annotations.intervals) == 1
    interval = annotations.intervals[0]
    assert interval.direction == Direction.GROWTH
    assert interval.statistics.price_coverage == 126 / 128
    assert interval.statistics.return_count == 124
    assert interval.statistics.longest_no_trade_run == 2
    assert segmentation.eligible_mask(data.observations, AnalysisConfig()).all()


def test_sustained_gaps_create_explicit_unavailable_intervals() -> None:
    data = experiment([100.0] * 40 + [None] * 40 + [100.0] * 40, [1] * 80 + [0] * 40)
    annotations = segmentation.annotate(data)
    assert [(item.start, item.stop, item.status) for item in annotations.intervals] == [
        (0, 40, "observed"),
        (40, 80, "insufficient_data"),
        (80, 120, "no_trades"),
    ]
    assert segmentation.eligible_mask(data.observations, AnalysisConfig()).sum() == 40


def test_sparse_prices_are_insufficient_without_imputing_a_trend() -> None:
    data = experiment([100.0 if step % 2 == 0 else None for step in range(64)])
    assert observation_blocks(data.observations, AnalysisConfig()) == (
        (0, 64, "insufficient_data"),
    )
    interval = segmentation.annotate(data).intervals[0]
    assert interval.direction is None and interval.statistics.price_coverage == 0.5
    assert interval.statistics.return_count == 0
    assert interval.statistics.return_std is None


def test_observed_sums_disclose_unknown_coverage() -> None:
    data = experiment([None, 100.0, 101.0, None], [None, 0, 3, None])
    stats = summarize_interval(data.observations, 0, 4)
    assert stats.trade_count == 3 and stats.trade_volume == 3.0
    assert stats.trade_count_coverage == 0.5 and stats.trade_volume_coverage == 0.5
    assert stats.trade_active_fraction == 0.5
    assert (stats.first_price_step, stats.last_price_step) == (1, 2)
    assert stats.price_change_percent == pytest.approx(1)
    assert stats.return_count == 1 and stats.return_std is None


def test_decoder_removes_isolated_score_flips_and_keeps_sustained_reversal() -> None:
    data = experiment([100.0] * 128)
    scores = np.full((128, 3), 0.01)
    scores[:64, 1] = 0.98
    scores[64:, 2] = 0.98
    scores[20] = [0.98, 0.01, 0.01]
    scores[90:92] = [0.01, 0.98, 0.01]
    intervals = segmentation.decode(scores, data.observations)
    assert [(item.start, item.stop, item.direction) for item in intervals] == [
        (0, 64, Direction.GROWTH),
        (64, 128, Direction.DECLINE),
    ]


def test_decoder_and_annotation_share_availability_and_validate_scores() -> None:
    data = experiment([100.0] * 32 + [None] * 32 + [100.0] * 32, [1] * 64 + [0] * 32)
    scores = np.tile([0.1, 0.8, 0.1], (96, 1))
    decoded = segmentation.decode(scores, data.observations)
    expected = segmentation.annotate(data).intervals
    assert [(item.start, item.stop, item.status) for item in decoded] == [
        (item.start, item.stop, item.status) for item in expected
    ]
    with pytest.raises(ValueError, match="sum to one"):
        segmentation.decode(np.zeros((96, 3)), data.observations)
    with pytest.raises(ValueError, match="finite"):
        segmentation.decode(np.full((96, 3), np.nan), data.observations)
    with pytest.raises(ValueError, match="one row"):
        segmentation.decode(scores[:-1], data.observations)


def test_human_references_require_explicit_provenance_and_keep_human_directions() -> None:
    data = experiment([100.0] * 64)
    automatic = segmentation.annotate(data)
    payload = automatic.model_dump(mode="json")
    payload["reference_kind"] = "human"
    with pytest.raises(ValidationError, match="reviewer"):
        DirectionAnnotations.model_validate_json(json.dumps(payload))
    payload["policy"] = "human-directions/1.0"
    payload["reviewer"] = "reviewer-1"
    payload["intervals"][0]["direction"] = "growth"
    human = DirectionAnnotations.model_validate_json(json.dumps(payload))
    human.validate_for(data)
    assert human.intervals[0].direction == Direction.GROWTH
    assert human.reference_kind == "human"
    payload["reviewer"] = "   "
    with pytest.raises(ValidationError, match="reviewer"):
        DirectionAnnotations.model_validate_json(json.dumps(payload))
    automatic_payload = automatic.model_dump(mode="json")
    automatic_payload["policy"] = "human-directions/1.0"
    with pytest.raises(ValidationError, match="automatic policy"):
        DirectionAnnotations.model_validate_json(json.dumps(automatic_payload))


def test_annotation_validation_rejects_gaps_and_fabricated_descriptors() -> None:
    data = experiment([100.0] * 64)
    annotations = segmentation.annotate(data)
    item = annotations.intervals[0]
    gap = annotations.model_copy(update={"intervals": (item.model_copy(update={"start": 1}),)})
    with pytest.raises(ValueError, match="without gaps"):
        gap.validate_for(data)
    fabricated = item.statistics.model_copy(update={"trade_count": 999})
    bad = annotations.model_copy(
        update={"intervals": (item.model_copy(update={"statistics": fabricated}),)}
    )
    with pytest.raises(ValueError, match="does not match"):
        bad.validate_for(data)


@pytest.mark.parametrize("reference_kind", ["automatic", "human"])
@pytest.mark.parametrize(
    "bounds, message",
    [
        ([(0, 4, Direction.GROWTH), (4, 96, Direction.SIDEWAYS)], "minimum duration"),
        ([(0, 32, Direction.SIDEWAYS), (32, 96, Direction.SIDEWAYS)], "must be merged"),
    ],
)
def test_imported_references_follow_duration_and_merging_contracts(
    reference_kind: str, bounds: list[tuple[int, int, Direction]], message: str
) -> None:
    data = experiment([100.0] * 96)
    payload = segmentation.annotate(data).model_dump(mode="json")
    if reference_kind == "human":
        payload.update(reference_kind="human", policy="human-directions/1.0", reviewer="reviewer-1")
    payload["intervals"] = [
        DirectionInterval(
            instrument_id="A/Q",
            start=start,
            stop=stop,
            direction=direction,
            status="observed",
            statistics=summarize_interval(data.observations, start, stop),
        ).model_dump(mode="json")
        for start, stop, direction in bounds
    ]
    imported = DirectionAnnotations.model_validate_json(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        imported.validate_for(data)


def test_short_availability_blocks_allow_one_direction_but_not_shorter_fragments() -> None:
    data = experiment([100.0] * 16 + [None] * 32 + [100.0] * 16)
    annotations = segmentation.annotate(data)
    assert [(item.start, item.stop, item.status) for item in annotations.intervals] == [
        (0, 16, "observed"),
        (16, 48, "insufficient_data"),
        (48, 64, "observed"),
    ]
    annotations.validate_for(data)
    split = tuple(
        DirectionInterval(
            instrument_id="A/Q",
            start=start,
            stop=stop,
            direction=direction,
            status="observed",
            statistics=summarize_interval(data.observations, start, stop),
        )
        for start, stop, direction in [(0, 8, Direction.GROWTH), (8, 16, Direction.SIDEWAYS)]
    )
    fragmented = annotations.model_copy(update={"intervals": (*split, *annotations.intervals[1:])})
    with pytest.raises(ValueError, match="minimum duration"):
        fragmented.validate_for(data)


def test_unavailable_references_cannot_invent_extra_boundaries() -> None:
    data = experiment([100.0] * 64, [0] * 64)
    annotations = segmentation.annotate(data)
    fragments = tuple(
        DirectionInterval(
            instrument_id="A/Q",
            start=start,
            stop=stop,
            direction=None,
            status="no_trades",
            statistics=summarize_interval(data.observations, start, stop),
        )
        for start, stop in [(0, 32), (32, 64)]
    )
    fragmented = annotations.model_copy(update={"intervals": fragments})
    with pytest.raises(ValueError, match="must be merged"):
        fragmented.validate_for(data)


def test_annotation_is_deterministic_without_generator_metadata() -> None:
    data = experiment([100 + step / 100 for step in range(128)])
    first = segmentation.annotate(data)
    second = segmentation.annotate(data)
    assert first.model_dump_json() == second.model_dump_json()
