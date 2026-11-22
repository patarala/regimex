"""Timing and workload measured for an administrative training run."""

from datetime import datetime
from typing import Annotated, Self

from pydantic import AwareDatetime, Field, JsonValue

from regimex.contracts import Nonnegative, StrictModel
from regimex.operations import TERMINAL_STATES, Operation


class TrainingFitStatistics(StrictModel):
    fit_seconds: Nonnegative
    train_rows: Annotated[int, Field(gt=0)]
    validation_rows: Annotated[int, Field(gt=0)]


class TrainingRunStatistics(StrictModel):
    as_of: AwareDatetime
    elapsed_seconds: Nonnegative
    queued_seconds: Nonnegative | None
    running_seconds: Nonnegative | None
    fit_seconds: Nonnegative | None
    train_rows: Annotated[int, Field(gt=0)] | None
    validation_rows: Annotated[int, Field(gt=0)] | None

    @classmethod
    def from_operation(
        cls, job: Operation, output: dict[str, JsonValue] | None, *, now: datetime
    ) -> Self:
        as_of = job.updated_at if job.state in TERMINAL_STATES else max(now, job.updated_at)
        elapsed = as_of - job.created_at
        queued = job.training_queue_time(as_of)
        saved = (output or {}).get("fit_statistics")
        fit = TrainingFitStatistics.model_validate(saved) if saved is not None else None
        return cls(
            as_of=as_of,
            elapsed_seconds=elapsed.total_seconds(),
            queued_seconds=queued.total_seconds() if queued is not None else None,
            running_seconds=(elapsed - queued).total_seconds() if queued is not None else None,
            fit_seconds=fit.fit_seconds if fit is not None else None,
            train_rows=fit.train_rows if fit is not None else None,
            validation_rows=fit.validation_rows if fit is not None else None,
        )
