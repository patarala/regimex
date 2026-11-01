"""Render comparable saved evaluations without running models or reading datasets."""

import json
import math
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from regimex.contracts import Identifier
from regimex.data_catalog import content_hash
from regimex.directions import DIRECTION_NAMES
from regimex.operations import Sha256
from regimex.storage import digest, output_directory, write_json

COMPARISON_VERSION = "model-comparison/2.0"
Score = Annotated[float, Field(ge=0, le=1)]
Count = Annotated[int, Field(ge=0)]
DirectionName = Literal["sideways", "growth", "decline"]


class _Metric(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, allow_inf_nan=False)


class _ClassScore(_Metric):
    precision: Score
    recall: Score
    f1: Score
    support: Count
    predicted: Count
    matched: Count

    @model_validator(mode="after")
    def valid_counts(self) -> Self:
        if self.matched > min(self.support, self.predicted):
            raise ValueError("matched count exceeds reference or prediction support")
        denominator = self.support + self.predicted
        expected = 2 * self.matched / denominator if denominator else 0.0
        precision = self.matched / self.predicted if self.predicted else 0.0
        recall = self.matched / self.support if self.support else 0.0
        if not all(
            math.isclose(actual, wanted, abs_tol=1e-12)
            for actual, wanted in (
                (self.f1, expected),
                (self.precision, precision),
                (self.recall, recall),
            )
        ):
            raise ValueError("class scores disagree with their counts")
        return self


class _DirectionScores(_Metric):
    macro_f1: Score
    per_class: dict[DirectionName, _ClassScore]
    class_names: list[DirectionName]
    iou_threshold: Score | None = None

    @model_validator(mode="after")
    def complete_classes(self) -> Self:
        if tuple(self.class_names) != DIRECTION_NAMES or set(self.per_class) != set(
            DIRECTION_NAMES
        ):
            raise ValueError("evaluation must contain the declared three direction classes")
        if not math.isclose(
            self.macro_f1, sum(item.f1 for item in self.per_class.values()) / 3, abs_tol=1e-12
        ):
            raise ValueError("macro-F1 disagrees with the three class scores")
        return self


class _BoundaryScores(_Metric):
    precision: Score
    recall: Score
    f1: Score
    tolerance_steps: Count
    matched: Count
    reference_count: Count
    predicted_count: Count
    defined: bool

    @model_validator(mode="after")
    def consistent_counts(self) -> Self:
        values = _ClassScore(
            precision=self.precision,
            recall=self.recall,
            f1=self.f1,
            support=self.reference_count,
            predicted=self.predicted_count,
            matched=self.matched,
        )
        if self.defined != bool(values.support + values.predicted):
            raise ValueError("boundary defined flag disagrees with boundary counts")
        if self.tolerance_steps != 4:
            raise ValueError("unsupported boundary matching tolerance")
        return self


class _LagSummary(_Metric):
    count: Count
    mean_steps: float | None
    median_steps: float | None
    mean_absolute_steps: Annotated[float, Field(ge=0)] | None
    p90_absolute_steps: Annotated[float, Field(ge=0)] | None
    minimum_steps: int | None
    maximum_steps: int | None
    early_count: Count
    on_time_count: Count
    late_count: Count


class _Alignment(_Metric):
    matched: Count
    unmatched_reference_count: Count
    unmatched_prediction_count: Count
    signed_lag_steps: _LagSummary


class _TypedBoundaryScores(_BoundaryScores):
    alignment: _Alignment

    @model_validator(mode="after")
    def consistent_alignment(self) -> Self:
        aligned = self.alignment
        if (
            aligned.matched < self.matched
            or aligned.matched + aligned.unmatched_reference_count != self.reference_count
            or aligned.matched + aligned.unmatched_prediction_count != self.predicted_count
            or aligned.signed_lag_steps.count != aligned.matched
        ):
            raise ValueError("typed transition alignment disagrees with boundary counts")
        return self


class _LengthSummary(_Metric):
    interval_count: Count
    minimum_steps: Count | None
    median_steps: Annotated[float, Field(ge=0)] | None
    maximum_steps: Count | None
    one_or_two_step_fraction: Score | None

    @model_validator(mode="after")
    def ordered_durations(self) -> Self:
        values = (
            self.minimum_steps,
            self.median_steps,
            self.maximum_steps,
            self.one_or_two_step_fraction,
        )
        if self.interval_count == 0:
            if any(value is not None for value in values):
                raise ValueError("empty interval sets cannot have duration statistics")
        elif (
            any(value is None for value in values)
            or self.minimum_steps is None
            or self.minimum_steps < 1
            or self.median_steps is None
            or self.maximum_steps is None
            or not self.minimum_steps <= self.median_steps <= self.maximum_steps
        ):
            raise ValueError("interval duration statistics are incomplete or unordered")
        return self


class _Fragmentation(_Metric):
    reference: _LengthSummary
    predicted: _LengthSummary
    predicted_to_reference_interval_ratio: Annotated[float, Field(ge=0)] | None

    @model_validator(mode="after")
    def count_ratio(self) -> Self:
        expected = (
            self.predicted.interval_count / self.reference.interval_count
            if self.reference.interval_count
            else None
        )
        if expected is None:
            if self.predicted_to_reference_interval_ratio is not None:
                raise ValueError("interval ratio is undefined without reference intervals")
        elif self.predicted_to_reference_interval_ratio is None or not math.isclose(
            expected, self.predicted_to_reference_interval_ratio, abs_tol=1e-12
        ):
            raise ValueError("fragmentation ratio disagrees with interval counts")
        return self


class _Coverage(_Metric):
    observed_step_count: Annotated[int, Field(gt=0)]
    reference_classified_step_count: Count
    predicted_classified_step_count: Count
    reference_classified_fraction: Score
    predicted_classified_fraction: Score
    eligible_prediction_coverage: Score | None

    @model_validator(mode="after")
    def valid_coverage(self) -> Self:
        for count, fraction in (
            (self.reference_classified_step_count, self.reference_classified_fraction),
            (self.predicted_classified_step_count, self.predicted_classified_fraction),
        ):
            if count > self.observed_step_count or not math.isclose(
                fraction, count / self.observed_step_count, abs_tol=1e-12
            ):
                raise ValueError("classified coverage disagrees with observed grid counts")
        if (self.eligible_prediction_coverage is None) != (
            self.reference_classified_step_count == 0
        ):
            raise ValueError("eligible coverage must be defined exactly when references exist")
        return self


class _Confidence(_Metric):
    seed: Count
    repetitions: Annotated[int, Field(gt=0)]
    confidence_level: Score
    experiment_count: Annotated[int, Field(gt=0)]
    segment_macro_f1_percentile_interval: list[Score]
    point_macro_f1_percentile_interval: list[Score]
    boundary_f1_percentile_interval: list[Score]
    typed_boundary_f1_percentile_interval: list[Score]

    @model_validator(mode="after")
    def ordered_limits(self) -> Self:
        if self.confidence_level != 0.95:
            raise ValueError("comparison requires 95% confidence intervals")
        for interval in (
            self.segment_macro_f1_percentile_interval,
            self.point_macro_f1_percentile_interval,
            self.boundary_f1_percentile_interval,
            self.typed_boundary_f1_percentile_interval,
        ):
            if len(interval) != 2 or interval[0] > interval[1]:
                raise ValueError("confidence intervals require two ordered endpoints")
        return self


class _Experiment(_Metric):
    experiment_id: str
    reference_grid_sha256: Sha256
    segment_macro_f1: Score
    point_macro_f1: Score
    boundary_f1: Score
    segment_counts: list[list[Count]]
    point_counts: list[list[Count]]
    boundary_counts: list[Count]
    typed_boundary_counts: list[Count]
    typed_boundary_f1: Score
    typed_boundary_lags: list[int]

    @model_validator(mode="after")
    def consistent_scores(self) -> Self:
        def f1(counts: list[int]) -> float:
            matched, false_positive, false_negative = counts
            denominator = 2 * matched + false_positive + false_negative
            return 2 * matched / denominator if denominator else 0.0

        for counts, score in (
            (self.segment_counts, self.segment_macro_f1),
            (self.point_counts, self.point_macro_f1),
        ):
            if len(counts) != 3 or any(len(row) != 3 for row in counts):
                raise ValueError("experiment direction counts require three rows of three values")
            if not math.isclose(sum(f1(row) for row in counts) / 3, score, abs_tol=1e-12):
                raise ValueError("experiment F1 disagrees with its counts")
        for boundary_counts, score in (
            (self.boundary_counts, self.boundary_f1),
            (self.typed_boundary_counts, self.typed_boundary_f1),
        ):
            if len(boundary_counts) != 3 or not math.isclose(
                f1(boundary_counts), score, abs_tol=1e-12
            ):
                raise ValueError("experiment boundary F1 disagrees with its counts")
        matched, false_positive, false_negative = self.typed_boundary_counts
        if (
            not matched
            <= len(self.typed_boundary_lags)
            <= matched + min(false_positive, false_negative)
        ):
            raise ValueError("experiment alignment size disagrees with typed boundary counts")
        return self


class _Evaluation(_Metric):
    evaluation_version: Literal["2.3"]
    scope: Literal["fixed_model_validation", "controlled_final_benchmark"]
    model_version_id: Identifier
    bundle_sha256: Sha256
    dataset_sha256: Sha256
    reference_sha256: Sha256
    reference_policy: Identifier
    reference_kind: Literal["automatic", "human"]
    training_reference_kind: Literal["automatic", "human"]
    training_reference_policy: Identifier
    training_reference_sha256: Sha256
    validation_selection_reference_sha256: Sha256
    analysis_profile_sha256: Sha256
    decoder_version: Identifier
    inference_runtime: dict[str, JsonValue]
    segment_macro_f1: Score
    point_macro_f1: Score
    segment_metrics: _DirectionScores
    point_metrics: _DirectionScores
    boundary_metrics: _BoundaryScores
    typed_boundary_metrics: _TypedBoundaryScores
    fragmentation: _Fragmentation
    coverage: _Coverage
    experiment_count: Annotated[int, Field(gt=0)]
    instrument_sequence_count: Annotated[int, Field(gt=0)]
    confidence_intervals: _Confidence
    experiment_statistics: list[_Experiment]

    @model_validator(mode="after")
    def consistent_summary(self) -> Self:
        if (
            self.segment_macro_f1 != self.segment_metrics.macro_f1
            or self.point_macro_f1 != self.point_metrics.macro_f1
            or len({item.experiment_id for item in self.experiment_statistics})
            != self.experiment_count
            or len(self.experiment_statistics) != self.experiment_count
            or self.confidence_intervals.experiment_count != self.experiment_count
        ):
            raise ValueError("evaluation summary differs from class or experiment metrics")
        if self.segment_metrics.iou_threshold != 0.5:
            raise ValueError("unsupported interval matching IoU threshold")
        reference_steps = sum(item.support for item in self.point_metrics.per_class.values())
        eligible_predicted_steps = sum(
            item.predicted for item in self.point_metrics.per_class.values()
        )
        if (
            reference_steps != self.coverage.reference_classified_step_count
            or eligible_predicted_steps > self.coverage.predicted_classified_step_count
            or (
                reference_steps
                and not math.isclose(
                    eligible_predicted_steps / reference_steps,
                    self.coverage.eligible_prediction_coverage or 0.0,
                    abs_tol=1e-12,
                )
            )
            or sum(item.support for item in self.segment_metrics.per_class.values())
            != self.fragmentation.reference.interval_count
            or sum(item.predicted for item in self.segment_metrics.per_class.values())
            != self.fragmentation.predicted.interval_count
        ):
            raise ValueError("coverage or interval totals disagree with class counts")
        for key, metrics in (
            ("segment_counts", self.segment_metrics),
            ("point_counts", self.point_metrics),
        ):
            for index, name in enumerate(self.segment_metrics.class_names):
                score = metrics.per_class[name]
                actual = [
                    sum(getattr(item, key)[index][column] for item in self.experiment_statistics)
                    for column in range(3)
                ]
                if actual != [
                    score.matched,
                    score.predicted - score.matched,
                    score.support - score.matched,
                ]:
                    raise ValueError("experiment direction counts disagree with evaluation totals")
        for key, boundaries in (
            ("boundary_counts", self.boundary_metrics),
            ("typed_boundary_counts", self.typed_boundary_metrics),
        ):
            totals = [
                sum(getattr(item, key)[column] for item in self.experiment_statistics)
                for column in range(3)
            ]
            if totals != [
                boundaries.matched,
                boundaries.predicted_count - boundaries.matched,
                boundaries.reference_count - boundaries.matched,
            ]:
                raise ValueError("experiment boundary counts disagree with evaluation totals")
        from regimex.evaluation import lag_summary

        lags = [lag for item in self.experiment_statistics for lag in item.typed_boundary_lags]
        if self.typed_boundary_metrics.alignment.signed_lag_steps.model_dump() != lag_summary(lags):
            raise ValueError("typed lag summary disagrees with experiment alignments")
        return self


def _paired_comparison(reference: _Evaluation, candidate: _Evaluation) -> dict[str, JsonValue]:
    import numpy as np

    from regimex.evaluation import ExperimentCounts, paired_bootstrap

    def counts(value: _Evaluation) -> dict[str, ExperimentCounts]:
        return {
            item.experiment_id: (
                np.asarray(item.segment_counts, dtype=np.int64),
                np.asarray(item.point_counts, dtype=np.int64),
                np.asarray(item.boundary_counts, dtype=np.int64),
                np.asarray(item.typed_boundary_counts, dtype=np.int64),
            )
            for item in value.experiment_statistics
        }

    return {
        "reference_model_version_id": reference.model_version_id,
        "candidate_model_version_id": candidate.model_version_id,
        **paired_bootstrap(
            counts(reference),
            counts(candidate),
            repetitions=reference.confidence_intervals.repetitions,
            seed=reference.confidence_intervals.seed,
        ),
    }


def render_model_comparison(evaluations: tuple[dict[str, JsonValue], ...], output: Path) -> None:
    """Create an offline comparison of fixed models evaluated on identical references.

    Inputs are the saved JSON reports returned by the administrative evaluation
    workflow. This renderer neither authorizes nor performs final evaluation.
    """
    parsed = tuple(_Evaluation.model_validate(item) for item in evaluations)
    if len(parsed) < 2:
        raise ValueError("comparison requires at least two fixed-model evaluations")
    if len({item.model_version_id for item in parsed}) != len(parsed):
        raise ValueError("comparison model version IDs must be unique")
    first = parsed[0]
    shared = (
        "evaluation_version",
        "scope",
        "dataset_sha256",
        "reference_sha256",
        "reference_policy",
        "reference_kind",
        "analysis_profile_sha256",
        "decoder_version",
        "experiment_count",
        "instrument_sequence_count",
    )
    for item in parsed[1:]:
        if any(getattr(item, key) != getattr(first, key) for key in shared):
            raise ValueError("comparison requires the same dataset, references and analysis policy")
        reference_experiments = {
            entry.experiment_id: entry for entry in first.experiment_statistics
        }
        for entry in item.experiment_statistics:
            expected = reference_experiments.get(entry.experiment_id)
            if expected is None or (
                entry.reference_grid_sha256 != expected.reference_grid_sha256
                or any(
                    actual[0] + actual[2] != wanted[0] + wanted[2]
                    for name in ("segment_counts", "point_counts")
                    for actual, wanted in zip(
                        getattr(entry, name), getattr(expected, name), strict=True
                    )
                )
                or entry.boundary_counts[0] + entry.boundary_counts[2]
                != expected.boundary_counts[0] + expected.boundary_counts[2]
                or entry.typed_boundary_counts[0] + entry.typed_boundary_counts[2]
                != expected.typed_boundary_counts[0] + expected.typed_boundary_counts[2]
            ):
                raise ValueError("comparison reference populations differ within an experiment")
        if (
            {entry.experiment_id for entry in item.experiment_statistics}
            != {entry.experiment_id for entry in first.experiment_statistics}
            or item.boundary_metrics.tolerance_steps != first.boundary_metrics.tolerance_steps
            or item.fragmentation.reference != first.fragmentation.reference
            or item.coverage.observed_step_count != first.coverage.observed_step_count
            or item.coverage.reference_classified_step_count
            != first.coverage.reference_classified_step_count
            or any(
                getattr(item.confidence_intervals, key) != getattr(first.confidence_intervals, key)
                for key in ("seed", "repetitions", "confidence_level")
            )
            or any(
                getattr(item, name).per_class[direction].support
                != getattr(first, name).per_class[direction].support
                for name in ("segment_metrics", "point_metrics")
                for direction in DIRECTION_NAMES
            )
        ):
            raise ValueError("comparison evaluation populations or uncertainty policies differ")
    payload = {
        "version": COMPARISON_VERSION,
        "evaluations": [item.model_dump(mode="json") for item in parsed],
        "evaluation_sha256": [content_hash(item) for item in evaluations],
        "paired_comparisons": [_paired_comparison(first, item) for item in parsed[1:]],
    }
    encoded = (
        json.dumps(payload, ensure_ascii=True, allow_nan=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    templates = files("regimex").joinpath("templates")
    with output_directory(output) as target:
        (target / "index.html").write_text(
            templates.joinpath("model-comparison.html")
            .read_text(encoding="utf-8")
            .replace("__COMPARISON_JSON__", encoded),
            encoding="utf-8",
        )
        for suffix in ("js", "css"):
            (target / f"comparison.{suffix}").write_text(
                templates.joinpath(f"model-comparison.{suffix}").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        write_json(target / "evaluations.json", list(evaluations))
        write_json(target / "paired-comparisons.json", payload["paired_comparisons"])
        write_json(
            target / "report-manifest.json",
            {
                "version": COMPARISON_VERSION,
                "evaluation_sha256": payload["evaluation_sha256"],
                "files": {path.name: digest(path) for path in sorted(target.iterdir())},
            },
        )
