"""Owner-visible records for commands run outside the worker queues."""

from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from regimex.contracts import StrictModel
from regimex.history_contracts import DisplayName

ActivityOutcome = Literal["succeeded", "failed", "interrupted"]
Note = Annotated[str, Field(max_length=500, pattern=r"^[^\x00]*$")]
ResultText = Annotated[str, Field(max_length=16_000, pattern=r"^[^\x00]*$")]
ExitCode = Annotated[int, Field(ge=-(2**31), le=2**31 - 1)]


class ActivityStart(StrictModel):
    activity_id: UUID
    title: DisplayName
    note: Note | None = None
    started_at: AwareDatetime | None = None


class ActivityFinish(StrictModel):
    status: ActivityOutcome
    exit_code: ExitCode | None = None
    finished_at: AwareDatetime | None = None
    result_text: ResultText | None = None

    @model_validator(mode="after")
    def valid_exit_code(self) -> Self:
        if self.status == "succeeded" and self.exit_code not in (None, 0):
            raise ValueError("Successful commands return exit code zero.")
        if self.status == "failed" and self.exit_code == 0:
            raise ValueError("A failed command cannot return exit code zero.")
        return self


class TerminalActivitySummary(StrictModel):
    activity_id: UUID
    title: DisplayName
    note: Note | None
    status: Literal["running"] | ActivityOutcome
    started_at: AwareDatetime
    finished_at: AwareDatetime | None
    exit_code: ExitCode | None


class TerminalActivity(TerminalActivitySummary):
    result_text: ResultText | None


class TerminalActivityPage(StrictModel):
    items: tuple[TerminalActivitySummary, ...]
    limit: int
    offset: int
