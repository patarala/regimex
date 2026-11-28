"""Paired interval F1 from saved validation counts, with no model runtime dependency."""

from collections.abc import Iterator, Sequence
from math import ceil
from operator import itemgetter
from random import Random
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from regimex.contracts import StrictModel
from regimex.operations import Sha256

Count = Annotated[int, Field(strict=True, ge=0)]
Counts = tuple[Count, Count, Count]


def paired_draws(size: int, repetitions: int, seed: int) -> Iterator[list[int]]:
    """Draw the same whole-experiment indices for both models in a stable order."""
    rng = Random(seed)
    population = range(size)
    for _ in range(repetitions):
        yield rng.choices(population, k=size)


class _Recorded(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _Experiment(_Recorded):
    experiment_id: str
    reference_grid_sha256: Sha256
    segment_counts: tuple[Counts, Counts, Counts]


class _Metric(_Recorded):
    iou_threshold: Annotated[float, Field(ge=0, le=1)]
    class_names: tuple[Literal["sideways"], Literal["growth"], Literal["decline"]]


class _Sampling(_Recorded):
    seed: Count
    repetitions: Annotated[int, Field(ge=1, le=1000)]
    confidence_level: Annotated[float, Field(ge=0.95, le=0.95)]
    segment_draws_pcg64: tuple[Annotated[float, Field(ge=0, le=1)], ...] | None = None

    @model_validator(mode="after")
    def complete_draws(self) -> Self:
        if (
            self.segment_draws_pcg64 is not None
            and len(self.segment_draws_pcg64) != self.repetitions
        ):
            raise ValueError("saved bootstrap draws must cover every repetition")
        return self


class _Evidence(_Recorded):
    scope: Literal["fixed_model_validation"]
    evaluation_version: Literal["2.3", "2.4", "2.5"]
    dataset_sha256: Sha256
    reference_sha256: Sha256
    analysis_profile_sha256: Sha256
    decoder_version: str
    experiment_count: Annotated[int, Field(gt=0)]
    segment_metrics: _Metric
    confidence_intervals: _Sampling
    experiment_statistics: tuple[_Experiment, ...]

    @model_validator(mode="after")
    def complete_population(self) -> Self:
        ids = {row.experiment_id for row in self.experiment_statistics}
        if len(ids) != self.experiment_count or len(ids) != len(self.experiment_statistics):
            raise ValueError("incomplete validation population")
        return self


class PairedValidationScore(StrictModel):
    status: Literal[
        "available", "missing_evaluation", "insufficient_evidence", "different_population"
    ]
    experiment_count: int | None = None
    scores: tuple[float, float] | None = None
    delta: float | None = None
    percentile_interval: tuple[float, float] | None = None


def _score_counts(columns: list[tuple[list[int], list[int]]], sample: Sequence[int]) -> float:
    score = 0.0
    select = itemgetter(*sample)
    for matched, total in columns:
        denominator = sum(select(total)) if len(sample) > 1 else total[sample[0]]
        if denominator:
            hits = sum(select(matched)) if len(sample) > 1 else matched[sample[0]]
            score += 2 * hits / denominator
    return score / 3


def _columns(rows: list[_Experiment]) -> list[tuple[list[int], list[int]]]:
    columns = []
    for index in range(3):
        counts = [row.segment_counts[index] for row in rows]
        columns.append(([tp for tp, _, _ in counts], [2 * tp + fp + fn for tp, fp, fn in counts]))
    return columns


def compare_interval_scores(first: object, second: object) -> PairedValidationScore:
    """Resample fixed-weight evaluation evidence, including all instruments of each experiment."""
    try:
        a, b = _Evidence.model_validate(first), _Evidence.model_validate(second)
    except ValidationError:
        return PairedValidationScore(status="insufficient_evidence")
    fields = (
        "dataset_sha256",
        "reference_sha256",
        "analysis_profile_sha256",
        "decoder_version",
        "segment_metrics",
        "experiment_count",
    )
    if any(getattr(a, key) != getattr(b, key) for key in fields):
        return PairedValidationScore(status="different_population")
    sampling = a.confidence_intervals
    if (sampling.seed, sampling.repetitions) != (
        b.confidence_intervals.seed,
        b.confidence_intervals.repetitions,
    ):
        return PairedValidationScore(status="different_population")
    left = sorted(a.experiment_statistics, key=lambda row: row.experiment_id)
    right = sorted(b.experiment_statistics, key=lambda row: row.experiment_id)
    for x, y in zip(left, right, strict=True):
        if (x.experiment_id, x.reference_grid_sha256) != (
            y.experiment_id,
            y.reference_grid_sha256,
        ) or any(
            m + fn != n + gn
            for (m, _, fn), (n, _, gn) in zip(x.segment_counts, y.segment_counts, strict=True)
        ):
            return PairedValidationScore(status="different_population")
    left_columns, right_columns = _columns(left), _columns(right)
    size = len(left)

    def difference(sample: Sequence[int]) -> float:
        return _score_counts(right_columns, sample) - _score_counts(left_columns, sample)

    interval = None
    if size > 1:
        left_draws = sampling.segment_draws_pcg64
        right_draws = b.confidence_intervals.segment_draws_pcg64
        if left_draws is not None and right_draws is not None:
            values = sorted(y - x for x, y in zip(left_draws, right_draws, strict=True))
        else:
            values = sorted(
                difference(sample)
                for sample in paired_draws(size, sampling.repetitions, sampling.seed)
            )
        # Interpolate both tails at the same distance from their nearest endpoint.
        position = (len(values) - 1) * 0.025
        low, high = int(position), ceil(position)
        fraction = position - low
        interval = (
            values[low] + fraction * (values[high] - values[low]),
            values[-low - 1] + fraction * (values[-high - 1] - values[-low - 1]),
        )
    population = range(size)
    scores = (_score_counts(left_columns, population), _score_counts(right_columns, population))
    return PairedValidationScore(
        status="available",
        experiment_count=size,
        scores=scores,
        delta=scores[1] - scores[0],
        percentile_interval=interval,
    )
