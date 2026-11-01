"""Commit fencing, cancellation ordering, bounded recovery and immutable model selection."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from regimex.operations import (
    Operation,
    OperationFailure,
    ResolvedModel,
    acknowledge_cancel,
    fail,
    heartbeat,
    recover_stale,
    request_cancel,
    start_attempt,
    succeed,
)

NOW = datetime(2026, 10, 21, tzinfo=UTC)


def queued(**changes: object) -> Operation:
    return Operation.model_validate(
        {
            "operation_id": uuid4(),
            "owner_id": "user",
            "kind": "analysis",
            "resource_id": "analysis",
            "request_sha256": "1" * 64,
            "idempotency_key": "request-1",
            "model": ResolvedModel(
                model_version_id="baseline-1",
                bundle_sha256="2" * 64,
                inference_policy_sha256="3" * 64,
            ),
            "created_at": NOW,
            "updated_at": NOW,
            **changes,
        }
    )


def test_cancel_before_completion_prevents_publication() -> None:
    token = uuid4()
    operation = start_attempt(queued(), token, NOW)
    operation = request_cancel(operation, NOW)
    with pytest.raises(ValueError, match="cannot commit"):
        succeed(operation, token, "a" * 64, NOW)
    operation = acknowledge_cancel(operation, token, NOW)
    assert operation.state == "cancelled"
    assert operation.result_sha256 is None
    with pytest.raises(ValueError, match="no longer active"):
        succeed(operation, token, "a" * 64, NOW)


def test_completion_before_cancellation_keeps_committed_result() -> None:
    token = uuid4()
    operation = succeed(start_attempt(queued(), token, NOW), token, "a" * 64, NOW)
    assert request_cancel(operation, NOW) == operation
    assert operation.model is not None
    assert operation.model.model_version_id == "baseline-1"
    assert operation.result_sha256 == "a" * 64


def test_recovery_fences_late_worker_even_after_new_attempt_starts() -> None:
    old_token, new_token = uuid4(), uuid4()
    operation = start_attempt(queued(), old_token, NOW)
    later = NOW + timedelta(minutes=5)
    operation = recover_stale(operation, NOW, later)
    with pytest.raises(ValueError, match="stale"):
        succeed(operation, old_token, "a" * 64, later)
    operation = start_attempt(operation, new_token, later)
    assert operation.attempt == 2
    with pytest.raises(ValueError, match="stale"):
        succeed(operation, old_token, "a" * 64, later)
    assert succeed(operation, new_token, "b" * 64, later).result_sha256 == "b" * 64


def test_worker_heartbeat_and_attempt_budget_bound_recovery() -> None:
    token = uuid4()
    later = NOW + timedelta(minutes=1)
    operation = heartbeat(
        start_attempt(queued(max_attempts=1), token, NOW), token, "loading", later
    )
    with pytest.raises(ValueError, match="still current"):
        recover_stale(operation, NOW, later)
    operation = recover_stale(operation, later, later)
    assert operation.state == "failed"
    assert operation.failure is not None
    assert operation.failure.code == "attempts_exhausted"


def test_recovery_of_cancelled_worker_does_not_restart_work() -> None:
    operation = request_cancel(start_attempt(queued(), uuid4(), NOW), NOW)
    assert recover_stale(operation, NOW, NOW).state == "cancelled"
    assert request_cancel(queued(), NOW).state == "cancelled"


@pytest.mark.parametrize("recovery", [False, True])
def test_cancellation_before_retry_keeps_prior_attempts_fenced(recovery: bool) -> None:
    token = uuid4()
    operation = start_attempt(queued(), token, NOW)
    if recovery:
        operation = recover_stale(operation, NOW, NOW)
    else:
        operation = fail(
            operation,
            token,
            OperationFailure(
                code="infrastructure_failure", message="Storage unavailable.", retryable=True
            ),
            NOW,
        )
    cancelled = request_cancel(operation, NOW)
    assert cancelled.state == "cancelled"
    assert cancelled.attempt_tokens == (token,)
    assert cancelled.attempt_token is None and cancelled.heartbeat_at is None
    assert request_cancel(cancelled, NOW) == cancelled
    with pytest.raises(ValueError, match="stale"):
        succeed(cancelled, token, "a" * 64, NOW)
    with pytest.raises(ValueError, match="not eligible"):
        start_attempt(cancelled, uuid4(), NOW)


def test_only_transient_failures_requeue_and_tokens_are_revoked() -> None:
    token = uuid4()
    operation = start_attempt(queued(), token, NOW)
    operation = fail(
        operation,
        token,
        OperationFailure(
            code="infrastructure_failure",
            message="Storage timeout.",
            retryable=True,
        ),
        NOW,
    )
    assert operation.state == "queued"
    with pytest.raises(ValueError, match="stale"):
        heartbeat(operation, token, "late", NOW)
    token = uuid4()
    operation = start_attempt(operation, token, NOW)
    operation = fail(
        operation, token, OperationFailure(code="invalid_data", message="Bad grid."), NOW
    )
    assert operation.state == "failed"
    with pytest.raises(ValidationError, match="only transient"):
        OperationFailure(code="invalid_data", message="Bad grid.", retryable=True)


def test_invalid_persisted_states_and_nonmonotonic_transitions_are_rejected() -> None:
    with pytest.raises(ValidationError, match="resolved immutable model"):
        queued(model=None)
    with pytest.raises(ValidationError, match="committed result"):
        queued(state="succeeded")
    with pytest.raises(ValidationError, match="terminal failure"):
        queued(failure=OperationFailure(code="invalid_data", message="Bad input."))
    with pytest.raises(ValueError, match="monotonic"):
        start_attempt(queued(), uuid4(), NOW - timedelta(seconds=1))
    with pytest.raises(ValueError, match="aware"):
        start_attempt(queued(), uuid4(), NOW.replace(tzinfo=None))


@pytest.mark.parametrize("recovery", [False, True])
def test_previously_used_token_cannot_unfence_a_worker(recovery: bool) -> None:
    token = uuid4()
    running = start_attempt(queued(), token, NOW)
    if recovery:
        retry = recover_stale(running, NOW, NOW)
    else:
        retry = fail(
            running,
            token,
            OperationFailure(code="source_failure", message="Temporary timeout.", retryable=True),
            NOW,
        )
    assert retry.attempt_tokens == (token,)
    with pytest.raises(ValueError, match="cannot be reused"):
        start_attempt(retry, token, NOW)
    new_token = uuid4()
    retry = start_attempt(retry, new_token, NOW)
    assert retry.attempt_tokens == (token, new_token)
    with pytest.raises(ValueError, match="stale"):
        succeed(retry, token, "a" * 64, NOW)


def test_terminal_states_require_attempt_evidence() -> None:
    with pytest.raises(ValidationError, match="requires a recorded attempt"):
        queued(state="succeeded", result_sha256="a" * 64)
    with pytest.raises(ValidationError, match="requires a recorded attempt"):
        queued(
            state="failed",
            failure=OperationFailure(code="invalid_data", message="Invalid observations."),
        )
    with pytest.raises(ValidationError, match="distinct recorded token"):
        queued(attempt=1)
    token = uuid4()
    with pytest.raises(ValidationError, match="distinct recorded token"):
        queued(attempt=2, attempt_tokens=(token, token))
    with pytest.raises(ValidationError, match="last recorded token"):
        queued(
            state="succeeded",
            result_sha256="a" * 64,
            attempt=1,
            attempt_tokens=(token,),
        )
    with pytest.raises(ValidationError, match="exhausted"):
        queued(attempt=1, max_attempts=1, attempt_tokens=(token,))


def test_idempotent_cancellation_still_rejects_invalid_time() -> None:
    cancelled = request_cancel(queued(), NOW)
    for invalid in (NOW.replace(tzinfo=None), NOW - timedelta(seconds=1)):
        with pytest.raises(ValueError, match="aware and monotonic"):
            request_cancel(cancelled, invalid)
    token = uuid4()
    completed = succeed(start_attempt(queued(), token, NOW), token, "a" * 64, NOW)
    with pytest.raises(ValueError, match="aware and monotonic"):
        request_cancel(completed, NOW - timedelta(seconds=1))


def test_recovery_rejects_naive_or_reversed_clock_before_comparing_cutoffs() -> None:
    running = start_attempt(queued(), uuid4(), NOW)
    with pytest.raises(ValueError, match="aware and monotonic"):
        recover_stale(running, NOW, NOW.replace(tzinfo=None))
    with pytest.raises(ValueError, match="aware and monotonic"):
        recover_stale(running, NOW, NOW - timedelta(seconds=1))
    with pytest.raises(ValueError, match="cutoff must be aware"):
        recover_stale(running, NOW.replace(tzinfo=None), NOW)
