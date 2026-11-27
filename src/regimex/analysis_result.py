"""Saved direction predictions on the complete instrument and time grid."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, model_validator

from regimex.contracts import MAX_STEPS, Identifier, StrictModel
from regimex.directions import AnalysisConfig, DirectionInterval
from regimex.exports import ExportInstrument, ModelTimeGrid, SourceIdentity
from regimex.model_registry import InferenceRuntime
from regimex.operations import ResolvedModel


class PredictionSequence(StrictModel):
    """Predictions preserve every input bin, including unavailable price periods."""

    instrument_id: Identifier
    probabilities: Annotated[
        tuple[tuple[float, float, float], ...], Field(min_length=1, max_length=MAX_STEPS)
    ]
    intervals: Annotated[tuple[DirectionInterval, ...], Field(min_length=1, max_length=MAX_STEPS)]

    @model_validator(mode="after")
    def full_grid(self) -> PredictionSequence:
        if not self.probabilities or not self.intervals:
            raise ValueError("predictions require the complete nonempty observation grid")
        for row in self.probabilities:
            if any(value < 0 or value > 1 for value in row) or abs(sum(row) - 1) > 1e-6:
                raise ValueError("probability rows must be normalized and bounded")
        previous = 0
        for interval in self.intervals:
            if interval.instrument_id != self.instrument_id or interval.start != previous:
                raise ValueError("prediction intervals must cover one instrument without gaps")
            previous = interval.stop
        if previous != len(self.probabilities):
            raise ValueError("prediction intervals must cover every probability row")
        return self


class AnalysisResult(StrictModel):
    result_version: Literal["2.0"] = "2.0"
    analysis_id: UUID
    snapshot_id: UUID
    snapshot_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    source: SourceIdentity
    model: ResolvedModel
    architecture_id: Identifier
    implementation_version: Identifier
    decoder_version: Identifier
    analysis_profile_id: Identifier
    analysis: AnalysisConfig
    inference_runtime: InferenceRuntime
    class_names: tuple[Literal["sideways"], Literal["growth"], Literal["decline"]]
    time_grid: ModelTimeGrid
    instruments: Annotated[tuple[ExportInstrument, ...], Field(min_length=1, max_length=2)]
    sequences: Annotated[tuple[PredictionSequence, ...], Field(min_length=1, max_length=2)]

    @model_validator(mode="after")
    def original_grid(self) -> AnalysisResult:
        instrument_ids = tuple(item.instrument_id for item in self.instruments)
        if len(set(instrument_ids)) != len(instrument_ids):
            raise ValueError("analysis instrument identifiers must be unique")
        if instrument_ids != tuple(item.instrument_id for item in self.sequences) or any(
            len(item.probabilities) != self.time_grid.step_count for item in self.sequences
        ):
            raise ValueError("results must preserve the complete input instrument/time grid")
        return self
