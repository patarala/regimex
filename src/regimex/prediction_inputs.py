"""Validated numeric inference inputs with no training targets or simulator metadata."""

from dataclasses import dataclass
from typing import Annotated

import numpy as np
from pydantic import Field

from regimex.contracts import MAX_STEPS, Identifier, StrictModel
from regimex.features import FEATURE_NAMES, FloatArray, IntArray


class PredictionSequenceInfo(StrictModel):
    experiment_id: Identifier
    instrument_id: Identifier
    step_count: Annotated[int, Field(ge=1, le=MAX_STEPS)]


def _validate_grid(
    values: FloatArray, offsets: IntArray, info: tuple[PredictionSequenceInfo, ...]
) -> None:
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.float32
        or values.ndim != 2
        or not len(values)
        or not values.shape[1]
        or np.isinf(values).any()
    ):
        raise ValueError("prediction values must be a nonempty float32 matrix without infinities")
    if (
        not isinstance(info, tuple)
        or not info
        or any(type(item) is not PredictionSequenceInfo for item in info)
        or len({(item.experiment_id, item.instrument_id) for item in info}) != len(info)
    ):
        raise ValueError(
            "prediction sequence identities must be unique and contain no training metadata"
        )
    for item in info:
        # Pydantic's model_copy/model_construct can bypass validation; do not let
        # those paths smuggle extra fields or invalid lengths into inference.
        if set(item.__dict__) != set(PredictionSequenceInfo.model_fields):
            raise ValueError("prediction sequence metadata contains undeclared fields")
        PredictionSequenceInfo.model_validate(item.__dict__)
    if (
        not isinstance(offsets, np.ndarray)
        or offsets.dtype != np.int64
        or offsets.shape != (len(info) + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(values)
        or not np.array_equal(np.diff(offsets), [item.step_count for item in info])
    ):
        raise ValueError("prediction offsets must preserve every original sequence and timestamp")


def _freeze(values: FloatArray, offsets: IntArray) -> tuple[FloatArray, IntArray]:
    # Copy at the boundary so later mutations of a training/evaluation container
    # cannot change an input after its shape and missingness have been validated.
    copied_values = values.copy()
    copied_offsets = offsets.copy()
    copied_values.flags.writeable = False
    copied_offsets.flags.writeable = False
    return copied_values, copied_offsets


@dataclass(frozen=True, slots=True)
class PredictionInputs:
    """Raw observation features on the original grid, including explicit missing masks.

    Sequence IDs are for alignment only; they are never numerical model features.
    Labels, references, random seeds, and generator settings have no fields here.
    """

    values: FloatArray
    offsets: IntArray
    info: tuple[PredictionSequenceInfo, ...]

    def __post_init__(self) -> None:
        _validate_grid(self.values, self.offsets, self.info)
        if self.values.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                "raw prediction inputs require the unchanged observation feature schema"
            )
        numeric_width = len(FEATURE_NAMES) // 2
        if not np.array_equal(
            self.values[:, numeric_width:], np.isnan(self.values[:, :numeric_width])
        ):
            raise ValueError("prediction missing-value masks disagree with observed values")
        values, offsets = _freeze(self.values, self.offsets)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "offsets", offsets)


@dataclass(frozen=True, slots=True)
class PreparedPredictionInputs:
    """Finite backend features after applying restored preprocessing.

    Backend representations may have different widths. Their explicit missing
    indicators remain in the numeric representation, as in the original models.
    """

    values: FloatArray
    offsets: IntArray
    info: tuple[PredictionSequenceInfo, ...]

    def __post_init__(self) -> None:
        _validate_grid(self.values, self.offsets, self.info)
        if not np.isfinite(self.values).all():
            raise ValueError("prepared prediction inputs must be finite after preprocessing")
        values, offsets = _freeze(self.values, self.offsets)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "offsets", offsets)


def require_raw_inputs(inputs: PredictionInputs) -> None:
    if type(inputs) is not PredictionInputs:
        raise TypeError("prediction requires observation-only PredictionInputs")


def require_prepared_inputs(inputs: PreparedPredictionInputs) -> None:
    if type(inputs) is not PreparedPredictionInputs:
        raise TypeError("backend inference requires observation-only PreparedPredictionInputs")
