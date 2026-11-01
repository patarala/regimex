"""Explicit saved predictions for report behavior tests, without fitting a model."""

from uuid import NAMESPACE_URL, uuid5

from regimex.application.service import AnalysisResult, PredictionSequence
from regimex.data_catalog import content_hash
from regimex.directions import (
    AnalysisConfig,
    Direction,
    DirectionInterval,
    observation_blocks,
    summarize_interval,
)
from regimex.exports import ExportV2
from regimex.model_registry import InferenceRuntime
from regimex.operations import ResolvedModel


def prediction_for(export: ExportV2, *, direction: Direction = Direction.GROWTH) -> AnalysisResult:
    analysis = AnalysisConfig(min_duration=4)
    runtime = InferenceRuntime()
    probabilities = {
        Direction.SIDEWAYS: (0.9, 0.05, 0.05),
        Direction.GROWTH: (0.05, 0.9, 0.05),
        Direction.DECLINE: (0.05, 0.05, 0.9),
    }[direction]
    sequences = []
    for instrument in export.instruments:
        rows = tuple(
            row for row in export.observations if row.instrument_id == instrument.instrument_id
        )
        sequences.append(
            PredictionSequence(
                instrument_id=instrument.instrument_id,
                probabilities=(probabilities,) * len(rows),
                intervals=tuple(
                    DirectionInterval(
                        instrument_id=instrument.instrument_id,
                        start=start,
                        stop=stop,
                        direction=direction if status == "observed" else None,
                        status=status,
                        statistics=summarize_interval(rows, start, stop),
                    )
                    for start, stop, status in observation_blocks(rows, analysis)
                ),
            )
        )
    return AnalysisResult(
        analysis_id=uuid5(NAMESPACE_URL, f"prediction:{export.content_sha256}:{direction}"),
        snapshot_id=uuid5(NAMESPACE_URL, f"snapshot:{export.content_sha256}"),
        snapshot_sha256=export.content_sha256,
        source=export.source,
        model=ResolvedModel(
            model_version_id="report-behavior-fixture",
            bundle_sha256="a" * 64,
            inference_policy_sha256=content_hash(runtime.model_dump(mode="json")),
        ),
        architecture_id="fixture",
        implementation_version="1.0",
        decoder_version="duration-viterbi/1.0",
        analysis_profile_id="fixture-scale",
        analysis=analysis,
        inference_runtime=runtime,
        class_names=("sideways", "growth", "decline"),
        time_grid=export.time_grid,
        instruments=export.instruments,
        sequences=tuple(sequences),
    )
