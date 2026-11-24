"""Pooled interval scores from per-group counts supplied by an offline evaluator."""

import hashlib
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from regimex.contracts import Count, StrictModel
from regimex.data_catalog import content_hash
from regimex.model_bundles import TrainingEvidence
from regimex.operations import Sha256

type Reference = Literal["simulator", "known_direction"]
type Split = Literal["train", "validation", "independent"]
type IntervalMetric = Literal["decoded_segment_macro_f1/iou-0.50"]
type IntervalCounts = tuple[Count, Count, Count]


class GroupCounts(StrictModel):
    group_sha256: Sha256
    # The evaluator pools all instruments and paired views that share a group.
    counts: tuple[IntervalCounts, IntervalCounts, IntervalCounts]


class CohortIdentity(StrictModel):
    reference: Reference
    split: Split
    reference_sha256: Sha256


class MeasuredCohort(CohortIdentity):
    groups: Annotated[tuple[GroupCounts, ...], Field(min_length=1)]


class DirectionScore(CohortIdentity):
    group_count: Annotated[int, Field(gt=0)]
    interval_macro_f1: Annotated[float, Field(ge=0, le=1)]
    cohort_sha256: Sha256
    measurement_sha256: Sha256


class DirectionQualityEvidence(StrictModel):
    metric: IntervalMetric
    bundle_sha256: Sha256
    protocol_sha256: Sha256
    report_sha256: Sha256
    rows: Annotated[tuple[DirectionScore, ...], Field(min_length=1, max_length=6)]

    def validate_for(self, bundle_sha256: str, training: TrainingEvidence) -> None:
        if self.bundle_sha256 != bundle_sha256:
            raise ValueError("Direction scores belong to another bundle.")
        train_labels = [training.labels, *(p.labels for p in training.parent_models)]
        validation_labels = [
            training.validation_references,
            *(p.validation_references for p in training.parent_models),
        ]
        catalogued = {r.content_sha256: r.kind for r in (*train_labels, *validation_labels)}
        train = {r.content_sha256 for r in train_labels}
        train.update(s.references_sha256 for s in training.additional_training_sources)
        validation = {r.content_sha256 for r in validation_labels}
        for row in self.rows:
            if row.split == "train" and row.reference_sha256 not in train:
                raise ValueError("Training scores must identify references used to fit the model.")
            if row.split == "validation" and row.reference_sha256 not in validation:
                raise ValueError("Validation scores must identify the model selection references.")
            if (
                row.split == "validation"
                and row.reference_sha256 == training.validation_references.content_sha256
                and not math.isclose(
                    row.interval_macro_f1,
                    training.validation_segment_macro_f1,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    "Full validation scores must reproduce the bundle's selection score."
                )
            if row.split == "independent" and row.reference_sha256 in train | validation:
                raise ValueError("Independent scores refer to training or validation labels.")
            kind = catalogued.get(row.reference_sha256)
            if kind is not None and (kind != "automatic" or row.reference != "simulator"):
                raise ValueError("Direction reference type differs from the bundle's labels.")


class DirectionQualityReport(StrictModel):
    metric: IntervalMetric
    class_names: tuple[Literal["sideways"], Literal["growth"], Literal["decline"]]
    bundle_sha256: Sha256
    protocol_sha256: Sha256
    rows: Annotated[tuple[MeasuredCohort, ...], Field(min_length=1, max_length=6)]

    def summarize(self, report_sha256: str) -> DirectionQualityEvidence:
        scores: list[DirectionScore] = []
        cohorts: set[tuple[Reference, Split]] = set()
        splits: dict[str, Split] = {}
        reference_kinds: dict[str, Reference] = {}
        for row in self.rows:
            cell = (row.reference, row.split)
            if cell in cohorts:
                raise ValueError("Each reference and split must have a single direction score.")
            cohorts.add(cell)
            if reference_kinds.setdefault(row.reference_sha256, row.reference) != row.reference:
                raise ValueError(
                    "A reference digest cannot identify two different reference types."
                )
            groups = sorted(row.groups, key=lambda group: group.group_sha256)
            ids = [group.group_sha256 for group in groups]
            if len(ids) != len(set(ids)):
                raise ValueError("A group may only contribute once to the same direction score.")
            for identity in ids:
                if splits.setdefault(identity, row.split) != row.split:
                    raise ValueError("Direction report splits contain overlapping groups.")
            totals = [[0, 0, 0] for _ in range(3)]
            for group in groups:
                for total, counts in zip(totals, group.counts, strict=True):
                    for index, count in enumerate(counts):
                        total[index] += count
            if not any(tp + fn for tp, _, fn in totals):
                raise ValueError("Direction scores require reference intervals.")
            f1 = [2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0 for tp, fp, fn in totals]
            scores.append(
                DirectionScore(
                    reference=row.reference,
                    split=row.split,
                    reference_sha256=row.reference_sha256,
                    group_count=len(ids),
                    interval_macro_f1=sum(f1) / 3,
                    cohort_sha256=content_hash(ids),
                    measurement_sha256=content_hash([g.model_dump(mode="json") for g in groups]),
                )
            )
        return DirectionQualityEvidence(
            metric=self.metric,
            bundle_sha256=self.bundle_sha256,
            protocol_sha256=self.protocol_sha256,
            report_sha256=report_sha256,
            rows=tuple(scores),
        )


def read_direction_quality(path: Path) -> DirectionQualityEvidence:
    from regimex.service_client import read_file

    data = read_file(path, 8 * 1024**2)
    return DirectionQualityReport.model_validate_json(data).summarize(
        hashlib.sha256(data).hexdigest()
    )
