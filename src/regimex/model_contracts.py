"""Worker contracts shared by installed scientific backends and application code."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import JsonValue

from regimex.contracts import Identifier, StrictModel
from regimex.directions import AnalysisConfig
from regimex.exports import ExportInstrument, ExportObservation, InstrumentUnits, ModelTimeGrid
from regimex.features import DoubleArray, IntArray
from regimex.model_bundles import TrainingEvidence
from regimex.model_registry import InferenceRuntime, TrainingPolicy, TrainingRuntime
from regimex.operations import Sha256
from regimex.prediction_inputs import PredictionSequenceInfo

SelectionEvaluator = Callable[[DoubleArray], float]


@dataclass(frozen=True, slots=True)
class ObservationInputs:
    """Complete raw observation series; feature preparation belongs to each backend.

    Missing values and observed zero counts remain distinct in these immutable
    rows. Times, units, and sequence IDs describe observations, never generation.
    """

    info: tuple[PredictionSequenceInfo, ...]
    observations: tuple[tuple[ExportObservation, ...], ...]
    time_grids: tuple[ModelTimeGrid, ...]
    instruments: tuple[ExportInstrument, ...]

    def __post_init__(self) -> None:
        if (
            type(self.info) is not tuple
            or not self.info
            or any(
                type(items) is not tuple or len(items) != len(self.info)
                for items in (self.observations, self.time_grids, self.instruments)
            )
        ):
            raise ValueError(
                "observation inputs require aligned nonempty immutable sequence tuples"
            )
        identities = set()
        experiment_grids: dict[str, ModelTimeGrid] = {}
        for info, rows, grid, instrument in zip(
            self.info, self.observations, self.time_grids, self.instruments, strict=True
        ):
            for value, model in (
                (info, PredictionSequenceInfo),
                (grid, ModelTimeGrid),
                (instrument, ExportInstrument),
            ):
                if type(value) is not model or set(value.__dict__) != set(model.model_fields):
                    raise ValueError(
                        "observation metadata contains incompatible or undeclared fields"
                    )
                model.model_validate(value.__dict__)
            # Validating an instrument does not revalidate an existing nested
            # Pydantic instance, including fields added by model_copy().
            units = instrument.units
            if type(units) is not InstrumentUnits or set(units.__dict__) != set(
                InstrumentUnits.model_fields
            ):
                raise ValueError("observation units contain incompatible or undeclared fields")
            InstrumentUnits.model_validate(units.__dict__)
            identity = (info.experiment_id, info.instrument_id)
            if (
                identity in identities
                or type(rows) is not tuple
                or len(rows) != info.step_count
                or grid.step_count != info.step_count
                or instrument.instrument_id != info.instrument_id
            ):
                raise ValueError(
                    "observation identities, instruments, and original time grids disagree"
                )
            identities.add(identity)
            if experiment_grids.setdefault(info.experiment_id, grid) != grid:
                raise ValueError(
                    "instruments from one experiment must share the original time grid"
                )
            for step, row in enumerate(rows):
                if (
                    type(row) is not ExportObservation
                    or set(row.__dict__) != set(ExportObservation.model_fields)
                    or row.step != step
                    or row.instrument_id != info.instrument_id
                ):
                    raise ValueError(
                        "observations must preserve every original contiguous timestamp"
                    )
                ExportObservation.model_validate(row.__dict__)

    @property
    def row_count(self) -> int:
        return sum(item.step_count for item in self.info)


def require_observation_inputs(inputs: ObservationInputs) -> None:
    if type(inputs) is not ObservationInputs:
        raise TypeError("backend prediction requires observation-only ObservationInputs")


@dataclass(frozen=True, slots=True)
class TrainingInputs:
    """Observation features and aligned targets, separate from inference inputs.

    The application establishes seed-group isolation before constructing these
    values. A backend cannot read reference intervals or a final partition here.
    """

    observations: ObservationInputs
    targets: IntArray

    def __post_init__(self) -> None:
        require_observation_inputs(self.observations)
        if (
            not isinstance(self.targets, np.ndarray)
            or self.targets.dtype != np.int64
            or self.targets.shape != (self.observations.row_count,)
            or not np.isin(self.targets, [-1, 0, 1, 2]).all()
        ):
            raise ValueError("training targets must be aligned int64 labels -1, 0, 1, or 2")
        targets = self.targets.copy()
        targets.flags.writeable = False
        object.__setattr__(self, "targets", targets)


class LoadedPredictor(Protocol):
    @property
    def architecture_id(self) -> str: ...

    @property
    def implementation_version(self) -> str: ...

    @property
    def class_names(self) -> tuple[str, ...]: ...

    @property
    def analysis(self) -> AnalysisConfig: ...

    def predict(self, inputs: ObservationInputs, *, runtime: InferenceRuntime) -> DoubleArray:
        """Return finite normalized probabilities on the exact original grid."""
        ...


@dataclass(frozen=True, slots=True)
class FitResult:
    predictor: LoadedPredictor
    history: tuple[dict[str, JsonValue], ...]
    selected_seed: int
    validation_score: float
    selected_settings: dict[str, JsonValue]
    environment: dict[str, JsonValue]


class BundleProvenance(StrictModel):
    model_version_id: Identifier
    analysis_profile: Identifier
    analysis_profile_sha256: Sha256
    training: TrainingEvidence
    build_id: Identifier
    environment_sha256: Sha256


@runtime_checkable
class ModelBackend(Protocol):
    @property
    def architecture_id(self) -> str: ...

    @property
    def implementation_version(self) -> str: ...

    def fit(
        self,
        training: TrainingInputs,
        validation: ObservationInputs,
        *,
        parameters: StrictModel,
        policy: TrainingPolicy,
        analysis: AnalysisConfig,
        runtime: TrainingRuntime,
        evaluator: SelectionEvaluator,
    ) -> FitResult: ...

    def write_bundle(
        self,
        result: FitResult,
        root: Path,
        *,
        provenance: BundleProvenance,
        analysis: AnalysisConfig,
    ) -> str:
        """Write a complete candidate bundle and return its manifest SHA-256."""
        ...

    def load_bundle(
        self, root: Path, *, manifest_sha256: str, runtime: InferenceRuntime
    ) -> LoadedPredictor: ...
