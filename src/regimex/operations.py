"""Operation state transitions; persistence must serialize them with a row lock or CAS."""

from datetime import datetime
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from regimex.contracts import Identifier, StrictModel

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
OperationKind = Literal[
    "generation", "training", "evaluation", "analysis", "reference", "publication"
]
OPERATION_VERSION: Literal["1.2"] = "1.2"
OperationState = Literal[
    "queued", "running", "cancel_requested", "cancelled", "succeeded", "failed"
]
TERMINAL_STATES = frozenset({"cancelled", "succeeded", "failed"})


class ResolvedModel(StrictModel):
    """Capture the concrete deployment at intake, before publication can change it."""

    model_version_id: Identifier
    bundle_sha256: Sha256
    inference_policy_sha256: Sha256


class OperationFailure(StrictModel):
    code: Literal[
        "invalid_data",
        "unsupported_configuration",
        "incompatible_artifact",
        "capacity_limit",
        "source_failure",
        "infrastructure_failure",
        "attempts_exhausted",
    ]
    message: Annotated[str, Field(min_length=1, max_length=1024)]
    retryable: bool = False

    @model_validator(mode="after")
    def retry_policy(self) -> Self:
        if self.retryable and self.code not in {"source_failure", "infrastructure_failure"}:
            raise ValueError("only transient source or infrastructure failures may be retried")
        return self


class Operation(StrictModel):
    version: Literal["1.2"] = OPERATION_VERSION
    operation_id: UUID
    owner_id: Identifier
    kind: OperationKind
    resource_id: Identifier
    request_sha256: Sha256
    idempotency_key: Identifier
    model: ResolvedModel | None = None
    state: OperationState = "queued"
    stage: Identifier = "queued"
    attempt: Annotated[int, Field(ge=0)] = 0
    max_attempts: Annotated[int, Field(ge=1, le=10)] = 3
    attempt_tokens: Annotated[tuple[UUID, ...], Field(max_length=10)] = ()
    attempt_token: UUID | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime
    heartbeat_at: AwareDatetime | None = None
    result_sha256: Sha256 | None = None
    failure: OperationFailure | None = None

    @model_validator(mode="after")
    def consistent_state(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("operation timestamps must be ordered")
        if self.attempt > self.max_attempts:
            raise ValueError("operation exceeds its attempt budget")
        if (
            len(self.attempt_tokens) != self.attempt
            or len(set(self.attempt_tokens)) != self.attempt
        ):
            raise ValueError("every attempt requires a distinct recorded token")
        if self.kind == "analysis" and self.model is None:
            raise ValueError("analysis requires a resolved immutable model")
        if self.state in {"running", "cancel_requested"} and (
            self.attempt < 1 or self.attempt_token is None or self.heartbeat_at is None
        ):
            raise ValueError("an active attempt requires a token and heartbeat")
        if self.state == "queued" and (
            self.attempt_token is not None or self.heartbeat_at is not None
        ):
            raise ValueError("a queued operation cannot retain an active attempt")
        if self.state == "queued" and self.attempt >= self.max_attempts:
            raise ValueError("an exhausted operation cannot remain queued")
        # A queued retry has no active attempt to retain when it is cancelled.
        cancelled_while_queued = (
            self.state == "cancelled" and self.attempt_token is None and self.heartbeat_at is None
        )
        if (
            self.state != "queued"
            and not cancelled_while_queued
            and self.attempt > 0
            and (self.attempt_token != self.attempt_tokens[-1] or self.heartbeat_at is None)
        ):
            raise ValueError("attempt state must retain the last recorded token and heartbeat")
        if self.attempt == 0 and (self.attempt_token is not None or self.heartbeat_at is not None):
            raise ValueError("an operation without attempts cannot retain a token or heartbeat")
        if self.heartbeat_at is not None and not (
            self.created_at <= self.heartbeat_at <= self.updated_at
        ):
            raise ValueError("heartbeat must fall within operation timestamps")
        if (self.state == "succeeded") != (self.result_sha256 is not None):
            raise ValueError("only a successful operation has a committed result")
        if (self.state == "failed") != (self.failure is not None):
            raise ValueError("only a failed operation has a terminal failure")
        if self.state in {"succeeded", "failed"} and self.attempt == 0:
            raise ValueError("a successful or failed operation requires a recorded attempt")
        return self


def _validate_time(operation: Operation, now: datetime) -> None:
    if now.tzinfo is None or now.utcoffset() is None or now < operation.updated_at:
        raise ValueError("transition time must be aware and monotonic")


def _change(operation: Operation, now: datetime, **changes: object) -> Operation:
    _validate_time(operation, now)
    return Operation.model_validate({**operation.model_dump(), **changes, "updated_at": now})


def _current(operation: Operation, token: UUID) -> None:
    if operation.state not in {"running", "cancel_requested"} or token != operation.attempt_token:
        raise ValueError("attempt is stale or no longer active")


def start_attempt(operation: Operation, token: UUID, now: datetime) -> Operation:
    if operation.state != "queued" or operation.attempt >= operation.max_attempts:
        raise ValueError("operation is not eligible to start")
    if token in operation.attempt_tokens:
        raise ValueError("attempt tokens cannot be reused")
    return _change(
        operation,
        now,
        state="running",
        stage="starting",
        attempt=operation.attempt + 1,
        attempt_tokens=(*operation.attempt_tokens, token),
        attempt_token=token,
        heartbeat_at=now,
    )


def heartbeat(operation: Operation, token: UUID, stage: str, now: datetime) -> Operation:
    _current(operation, token)
    return _change(operation, now, stage=stage, heartbeat_at=now)


def request_cancel(operation: Operation, now: datetime) -> Operation:
    _validate_time(operation, now)
    if operation.state in TERMINAL_STATES or operation.state == "cancel_requested":
        return operation
    state = "cancelled" if operation.state == "queued" else "cancel_requested"
    return _change(operation, now, state=state, stage=state)


def acknowledge_cancel(operation: Operation, token: UUID, now: datetime) -> Operation:
    _current(operation, token)
    if operation.state != "cancel_requested":
        raise ValueError("cancellation has not been requested")
    return _change(operation, now, state="cancelled", stage="cancelled")


def succeed(operation: Operation, token: UUID, result_sha256: str, now: datetime) -> Operation:
    _current(operation, token)
    if operation.state != "running":
        raise ValueError("a cancelled attempt cannot commit a result")
    return _change(operation, now, state="succeeded", stage="complete", result_sha256=result_sha256)


def fail(operation: Operation, token: UUID, failure: OperationFailure, now: datetime) -> Operation:
    _current(operation, token)
    if operation.state == "cancel_requested":
        return acknowledge_cancel(operation, token, now)
    if failure.retryable and operation.attempt < operation.max_attempts:
        return _change(
            operation,
            now,
            state="queued",
            stage="retry_pending",
            attempt_token=None,
            heartbeat_at=None,
        )
    return _change(operation, now, state="failed", stage="failed", failure=failure)


def recover_stale(operation: Operation, stale_before: datetime, now: datetime) -> Operation:
    """Fence a dead attempt before redelivery; never change a completed result."""
    _validate_time(operation, now)
    if stale_before.tzinfo is None or stale_before.utcoffset() is None or stale_before > now:
        raise ValueError("stale cutoff must be aware and no later than recovery")
    if operation.state not in {"running", "cancel_requested"}:
        raise ValueError("only an active attempt can be recovered")
    assert operation.heartbeat_at is not None
    if operation.heartbeat_at > stale_before:
        raise ValueError("attempt heartbeat is still current")
    assert operation.attempt_token is not None
    if operation.state == "cancel_requested":
        return acknowledge_cancel(operation, operation.attempt_token, now)
    failure = OperationFailure(
        code="attempts_exhausted", message="The worker stopped and the attempt budget is exhausted."
    )
    if operation.attempt >= operation.max_attempts:
        return fail(operation, operation.attempt_token, failure, now)
    return _change(
        operation,
        now,
        state="queued",
        stage="recovery_pending",
        attempt_token=None,
        heartbeat_at=None,
    )
