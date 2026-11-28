"""Checkpoint records shared by epoch-trained models."""

from itertools import groupby
from typing import Literal, cast

from pydantic import Field

from regimex.backends.checkpoints import ValidationCheckpoints
from regimex.contracts import Seed, StrictModel
from regimex.model_bundles import RepeatOutcome
from regimex.model_registry import TrainingPolicy


class EpochRecord(StrictModel):
    seed: Seed
    epoch: int = Field(ge=1, le=1000)
    training_loss: float = Field(ge=0)
    validation_segment_macro_f1: float | None = Field(default=None, ge=0, le=1)


class HistoryRecord(StrictModel):
    epochs: tuple[EpochRecord, ...] = Field(min_length=1, max_length=32_000)


class ValidationRecord(StrictModel):
    metric: Literal["decoded_segment_macro_f1/iou-0.50"] = "decoded_segment_macro_f1/iou-0.50"
    score: float = Field(ge=0, le=1)
    selected_seed: Seed
    selected_epoch: int = Field(ge=1, le=1000)


def validate_epoch_selection(
    architecture: str,
    policy: TrainingPolicy,
    validation: ValidationRecord,
    history: HistoryRecord,
    repeat_outcomes: tuple[RepeatOutcome, ...],
) -> None:
    outcomes = []
    for seed, entries in groupby(history.epochs, key=lambda entry: entry.seed):
        checkpoints = ValidationCheckpoints()
        reason = None
        last_epoch = 0
        for epoch, entry in enumerate(entries, 1):
            if reason is not None:
                raise ValueError(f"{architecture} history continues after its stopping condition")
            if entry.epoch != epoch or epoch > policy.epochs:
                raise ValueError(f"{architecture} history epochs differ from the configured search")
            scored = epoch in (1, policy.epochs) or epoch % policy.validation_interval == 0
            if scored != (entry.validation_segment_macro_f1 is not None):
                raise ValueError(
                    f"{architecture} validation history differs from the configured interval"
                )
            if entry.validation_segment_macro_f1 is not None:
                checkpoints.observe(epoch, entry.validation_segment_macro_f1)
            last_epoch = epoch
            reason = checkpoints.epoch_stop_reason(epoch, policy)
        if reason is None:
            raise ValueError(f"{architecture} history ends a seed before its stopping condition")
        outcomes.append(
            RepeatOutcome(
                seed=seed, step=last_epoch, best_step=checkpoints.best_step, reason=reason
            )
        )
    if tuple(row.seed for row in outcomes) != policy.seeds:
        raise ValueError(f"{architecture} history seeds differ from the configured search")
    if repeat_outcomes and repeat_outcomes != tuple(outcomes):
        raise ValueError(f"{architecture} repeat outcomes disagree with the validation history")
    best = max(
        (entry for entry in history.epochs if entry.validation_segment_macro_f1 is not None),
        key=lambda entry: cast(float, entry.validation_segment_macro_f1),
    )
    if (validation.selected_seed, validation.selected_epoch, validation.score) != (
        best.seed,
        best.epoch,
        best.validation_segment_macro_f1,
    ):
        raise ValueError(f"{architecture} selection disagrees with the validation history")
