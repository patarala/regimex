"""Validation checkpoint tracking shared by neural training loops and their codecs."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from regimex.model_bundles import RepeatOutcome, RepeatStopReason
from regimex.model_registry import TrainingPolicy


@dataclass(slots=True)
class ValidationCheckpoints:
    """Keep the earliest maximum and count checks since its last improvement."""

    best_step: int = 0
    best_score: float = -1.0
    checks_without_improvement: int = 0

    def observe(self, step: int, score: float) -> None:
        if score > self.best_score:
            self.best_step = step
            self.best_score = score
            self.checks_without_improvement = 0
        else:
            self.checks_without_improvement += 1

    def epoch_stop_reason(
        self, epoch: int, policy: TrainingPolicy
    ) -> Literal["epoch_limit", "early_stopping"] | None:
        if epoch == policy.epochs:
            return "epoch_limit"
        if self.checks_without_improvement >= policy.patience:
            return "early_stopping"
        return None

    def report(
        self,
        seed: int,
        step: int,
        row: dict[str, JsonValue],
        reason: RepeatStopReason | None,
        *,
        on_history: Callable[[dict[str, JsonValue]], None] | None,
        on_repeat_end: Callable[[RepeatOutcome], None] | None,
    ) -> None:
        if on_history is not None:
            progress = row.copy()
            if reason is not None:
                progress.update(stop_reason=reason, best_step=self.best_step)
            on_history(progress)
        if reason is not None and on_repeat_end is not None:
            on_repeat_end(
                RepeatOutcome(seed=seed, step=step, best_step=self.best_step, reason=reason)
            )
