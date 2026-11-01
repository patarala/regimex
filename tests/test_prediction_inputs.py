"""Inference boundaries preserve observations and exclude supervision metadata."""

from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, cast

import pytest

np = pytest.importorskip("numpy")

from regimex.contracts import Observation
from regimex.exports import load_export_v2
from regimex.features import FEATURE_NAMES, NUMERIC_NAMES, compute_features
from regimex.model_contracts import (
    ObservationInputs,
    TrainingInputs,
    require_observation_inputs,
)
from regimex.prediction_inputs import (
    PredictionInputs,
    PredictionSequenceInfo,
    PreparedPredictionInputs,
)


def fixed_inputs() -> PredictionInputs:
    parts = []
    lengths = (7, 11)
    for index, length in enumerate(lengths):
        rows = tuple(
            Observation(
                instrument_id="A/Q",
                step=step,
                last_price=None if step == 2 else 100.0 + index * 20 + step / 10,
                trade_count=0 if step == 0 else 1,
                trade_volume=0.0 if step == 0 else 0.125,
                buy_order_count=None,
                buy_order_volume=None,
                sell_order_count=0,
                sell_order_volume=0.0,
                best_bid=None,
                best_ask=None,
            )
            for step in range(length)
        )
        parts.append(compute_features(rows))
    return PredictionInputs(
        values=np.concatenate(parts),
        offsets=np.array([0, lengths[0], sum(lengths)], dtype=np.int64),
        info=tuple(
            PredictionSequenceInfo(
                experiment_id=f"input-{index}", instrument_id="A/Q", step_count=length
            )
            for index, length in enumerate(lengths)
        ),
    )


def fixed_observations() -> ObservationInputs:
    export = load_export_v2(Path(__file__).parent / "fixtures" / "experiment-v2.json")
    return ObservationInputs(
        info=(
            PredictionSequenceInfo(
                experiment_id=export.source.snapshot_id,
                instrument_id=export.instruments[0].instrument_id,
                step_count=export.time_grid.step_count,
            ),
        ),
        observations=(export.observations,),
        time_grids=(export.time_grid,),
        instruments=export.instruments,
    )


def test_inference_contracts_have_no_targets_references_seeds_or_generator_configuration() -> None:
    raw = fixed_inputs()
    assert {field.name for field in fields(raw)} == {"values", "offsets", "info"}
    assert set(type(raw.info[0]).model_fields) == {"experiment_id", "instrument_id", "step_count"}
    observations = fixed_observations()
    assert {field.name for field in fields(observations)} == {
        "info",
        "observations",
        "time_grids",
        "instruments",
    }
    for container in (raw, observations):
        assert not hasattr(container, "targets")
        assert not hasattr(container, "references")
        assert not hasattr(container, "seed")
        assert not hasattr(container, "generator")


def test_numeric_inputs_copy_mutable_arrays_before_freezing_them() -> None:
    original = fixed_inputs()
    values, offsets = original.values.copy(), original.offsets.copy()
    raw = PredictionInputs(values, offsets, original.info)
    values[0, 0] = 5.0
    offsets[1] = 1
    assert raw.values[0, 0] == 0.0
    assert raw.offsets[1] == 7
    with pytest.raises(ValueError, match="read-only"):
        raw.values[0, 0] = 5.0
    with pytest.raises(ValueError, match="read-only"):
        raw.offsets[1] = 1


@pytest.mark.parametrize(
    "case",
    [
        "shape",
        "dtype",
        "infinity",
        "mask",
        "mask_nan",
        "start",
        "stop",
        "order",
        "length",
        "metadata",
    ],
)
def test_raw_prediction_boundary_rejects_corrupt_values_and_alignment(case: str) -> None:
    source = fixed_inputs()
    values = source.values.copy()
    offsets = source.offsets.copy()
    info = source.info
    if case == "shape":
        values = values[:, :-1]
    elif case == "dtype":
        values = values.astype(np.float64)
    elif case == "infinity":
        values[0, 0] = np.inf
    elif case == "mask":
        values[0, len(NUMERIC_NAMES)] = 1.0
    elif case == "mask_nan":
        values[0, -1] = np.nan
    elif case == "start":
        offsets[0] = 1
    elif case == "stop":
        offsets[-1] -= 1
    elif case == "order":
        offsets[1] = -1
    elif case == "length":
        info = (info[0].model_copy(update={"step_count": 8}), info[1])
    else:
        info = (info[0].model_copy(update={"reference_boundaries": (2, 4)}), info[1])
    with pytest.raises(ValueError):
        PredictionInputs(values, offsets, info)


def test_prepared_backend_input_requires_finite_values_and_matching_lengths() -> None:
    source = fixed_inputs()
    with pytest.raises(ValueError, match="finite"):
        PreparedPredictionInputs(source.values, source.offsets, source.info)
    complete = np.nan_to_num(source.values)
    prepared = PreparedPredictionInputs(complete, source.offsets, source.info)
    assert np.isfinite(prepared.values).all()
    complete[0, 0] = 99.0
    assert prepared.values[0, 0] == 0.0
    with pytest.raises(ValueError, match="offsets"):
        PreparedPredictionInputs(complete[:-1], prepared.offsets, prepared.info)


@pytest.mark.parametrize("extra", [{"seed": 1}, {"step_count": -1}, {"step_count": True}])
def test_constructed_metadata_cannot_bypass_the_observation_only_boundary(
    extra: dict[str, object],
) -> None:
    raw = fixed_inputs()
    changed = raw.info[0].model_copy(update=extra)
    with pytest.raises(ValueError):
        PredictionInputs(raw.values, raw.offsets, (changed, raw.info[1]))


@pytest.mark.parametrize("change", [{"training_labels": (1, 1)}, {"volume_decimal_places": -1}])
def test_nested_units_cannot_bypass_the_observation_only_boundary(
    change: dict[str, object],
) -> None:
    original = fixed_observations()
    instrument = original.instruments[0]
    invalid_units = instrument.units.model_copy(update=change)
    with pytest.raises(ValueError):
        replace(
            original,
            instruments=(instrument.model_copy(update={"units": invalid_units}),),
        )


def test_model_entry_points_explicitly_reject_training_containers() -> None:
    pytest.importorskip("torch")
    from regimex.baseline import LinearModel, aggregate
    from regimex.temporal import TCNClassifier, predict_tcn, temporal_features

    observations = fixed_observations()
    source = cast(
        "Any", TrainingInputs(observations, np.zeros(observations.row_count, dtype=np.int64))
    )
    width = len(FEATURE_NAMES) * 4
    linear = LinearModel(np.zeros((3, width)), np.zeros(3), np.zeros(width), np.ones(width), ())
    temporal = TCNClassifier(50, 4, 0.0)
    operations: tuple[Callable[[], object], ...] = (
        lambda: require_observation_inputs(source),
        lambda: aggregate(source, (4,)),
        lambda: linear.predict(source),
        lambda: temporal_features(source),
        lambda: predict_tcn(temporal, source),
    )
    for operation in operations:
        with pytest.raises(TypeError, match="observation-only"):
            operation()
