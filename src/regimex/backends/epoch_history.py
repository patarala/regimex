"""Checkpoint records shared by epoch-trained models."""

from typing import Literal, cast

from pydantic import Field

from regimex.contracts import Seed, StrictModel
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
) -> None:
    seeds: list[int] = []
    previous_epoch = 0
    seed_best = -1.0
    stale = 0
    for entry in history.epochs:
        if not seeds or entry.seed != seeds[-1]:
            if seeds and previous_epoch != policy.epochs and stale < policy.patience:
                raise ValueError(
                    f"{architecture} history ends a seed before its stopping condition"
                )
            seeds.append(entry.seed)
            previous_epoch = 0
            seed_best = -1.0
            stale = 0
        elif stale >= policy.patience:
            raise ValueError(f"{architecture} history continues after its stopping condition")
        if entry.epoch != previous_epoch + 1 or entry.epoch > policy.epochs:
            raise ValueError(f"{architecture} history epochs differ from the configured search")
        scored = (
            entry.epoch == 1
            or entry.epoch % policy.validation_interval == 0
            or entry.epoch == policy.epochs
        )
        if scored != (entry.validation_segment_macro_f1 is not None):
            raise ValueError(
                f"{architecture} validation history differs from the configured interval"
            )
        if entry.validation_segment_macro_f1 is not None:
            if entry.validation_segment_macro_f1 > seed_best:
                seed_best = entry.validation_segment_macro_f1
                stale = 0
            else:
                stale += 1
        previous_epoch = entry.epoch
    if previous_epoch != policy.epochs and stale < policy.patience:
        raise ValueError(f"{architecture} history ends a seed before its stopping condition")
    if tuple(seeds) != policy.seeds:
        raise ValueError(f"{architecture} history seeds differ from the configured search")
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
