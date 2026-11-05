"""Validated update histories for the native fixed-update TCN artifact codec."""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from regimex.contracts import Seed, StrictModel
from regimex.model_registry import FixedUpdatePolicy


class UpdateRecord(StrictModel):
    seed: Seed
    update: Annotated[int, Field(ge=1, le=32_000)]
    pass_index: Annotated[int, Field(ge=1)]
    batch_in_pass: Annotated[int, Field(ge=1)]
    training_loss: Annotated[float, Field(ge=0)]
    trained_sequences: Annotated[int, Field(ge=1, le=128)]
    real_rows: Annotated[int, Field(ge=1)]
    padding_rows: Annotated[int, Field(ge=0)]
    eligible_rows: Annotated[int, Field(ge=1)]
    validation_segment_macro_f1: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def coherent_rows(self) -> UpdateRecord:
        if self.eligible_rows > self.real_rows or self.real_rows < self.trained_sequences:
            raise ValueError("TCN update rows disagree with complete training sequences")
        if (self.real_rows + self.padding_rows) % self.trained_sequences:
            raise ValueError("TCN padded rows must form a rectangular sequence batch")
        return self


class UpdateHistoryRecord(StrictModel):
    updates: Annotated[tuple[UpdateRecord, ...], Field(min_length=1, max_length=32_000)]


class UpdateValidationRecord(StrictModel):
    metric: Literal["decoded_segment_macro_f1/iou-0.50"] = "decoded_segment_macro_f1/iou-0.50"
    score: Annotated[float, Field(ge=0, le=1)]
    selected_seed: Seed
    selected_update: Annotated[int, Field(ge=1, le=32_000)]


def validate_update_selection(
    policy: FixedUpdatePolicy,
    sequence_count: int,
    validation: UpdateValidationRecord,
    history: UpdateHistoryRecord,
) -> None:
    """Require complete budgets, the declared schedule and first-win selection."""
    FixedUpdatePolicy.model_validate(policy.model_dump())
    if len(history.updates) != len(policy.seeds) * policy.updates:
        raise ValueError("TCN update history does not complete every seed's budget")
    batches = (sequence_count + policy.batch_size - 1) // policy.batch_size
    selected: UpdateRecord | None = None
    best_score = -1.0
    for seed_index, seed in enumerate(policy.seeds):
        entries = history.updates[seed_index * policy.updates : (seed_index + 1) * policy.updates]
        previous_pass, previous_batch = 0, 0
        for update, entry in enumerate(entries, 1):
            expected_batch_size = min(
                policy.batch_size, sequence_count - (entry.batch_in_pass - 1) * policy.batch_size
            )
            if (
                entry.seed != seed
                or entry.update != update
                or entry.batch_in_pass > batches
                or entry.trained_sequences != expected_batch_size
                or (entry.pass_index, entry.batch_in_pass) <= (previous_pass, previous_batch)
                or entry.pass_index > previous_pass + 1
            ):
                raise ValueError("TCN update history has invalid seed, pass or batch accounting")
            previous_pass, previous_batch = entry.pass_index, entry.batch_in_pass
            scored = entry.validation_segment_macro_f1 is not None
            if scored != (update in policy.validation_updates):
                raise ValueError("TCN update validation differs from its configured schedule")
            score = entry.validation_segment_macro_f1
            if score is not None and score > best_score:
                selected, best_score = entry, score
    if selected is None or (
        validation.selected_seed,
        validation.selected_update,
        validation.score,
    ) != (selected.seed, selected.update, selected.validation_segment_macro_f1):
        raise ValueError("TCN selected update disagrees with its validation history")
