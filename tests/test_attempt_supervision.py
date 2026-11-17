"""Operation leases remain live during blocking I/O and stop on lost ownership."""

import threading
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from test_operations import queued

from regimex.application.attempt_supervision import AttemptSupervisor
from regimex.computation_worker import ComputationCancelled
from regimex.operations import Operation, heartbeat, recover_stale, request_cancel, start_attempt
from regimex.resource_limits import TimeBudgetExceeded
from regimex.service_contracts import StaleAttempt


def test_blocking_work_does_not_lose_its_lease_and_refresh_stops_after_exit() -> None:
    token = uuid4()
    operation = start_attempt(queued(), token, datetime.now(UTC))
    first = operation.heartbeat_at
    refreshed = threading.Event()
    stages: list[str] = []

    def renew(stage: str) -> Operation:
        nonlocal operation
        operation = heartbeat(operation, token, stage, datetime.now(UTC))
        stages.append(stage)
        if len(stages) >= 3:
            refreshed.set()
        return operation

    with AttemptSupervisor(
        renew, heartbeat_seconds=0.01, time_budget_seconds=5, stage="loading"
    ) as supervisor:
        # Waiting represents an input/output call that cannot send heartbeats.
        assert refreshed.wait(5)
        supervisor.poll()
        assert first is not None
        with pytest.raises(ValueError, match="still current"):
            recover_stale(operation, first, datetime.now(UTC))
        supervisor.check("committing")
    count = len(stages)
    assert stages[-1] == "committing"
    assert not threading.Event().wait(0.04)
    assert len(stages) == count


def test_exit_waits_for_the_bounded_inflight_refresh_without_late_calls() -> None:
    operation = start_attempt(queued(), uuid4(), datetime.now(UTC))
    entered, release, exiting, exited = (threading.Event() for _ in range(4))
    calls = 0

    def renew(stage: str) -> Operation:
        nonlocal calls
        calls += 1
        if calls > 1:
            entered.set()
            assert release.wait(5)
        return operation

    def attempt() -> None:
        with AttemptSupervisor(renew, heartbeat_seconds=0.01, time_budget_seconds=5):
            assert entered.wait(5)
            exiting.set()
        exited.set()

    worker = threading.Thread(target=attempt)
    worker.start()
    try:
        assert exiting.wait(5)
        assert not exited.wait(0.05)
    finally:
        release.set()
        worker.join(timeout=5)
    assert exited.is_set() and not worker.is_alive()
    assert calls == 2


@pytest.mark.parametrize("reason", ["cancel", "stale", "deadline"])
def test_background_failure_is_propagated_at_the_next_checkpoint(reason: str) -> None:
    operation = start_attempt(queued(), uuid4(), datetime.now(UTC))
    assert operation.attempt_token is not None
    ready, fail = threading.Event(), threading.Event()

    def renew(stage: str) -> Operation:
        if fail.is_set():
            ready.set()
            if reason == "stale":
                raise StaleAttempt()
            return request_cancel(operation, datetime.now(UTC))
        return operation

    expected = {
        "cancel": ComputationCancelled,
        "stale": StaleAttempt,
        "deadline": TimeBudgetExceeded,
    }[reason]
    with AttemptSupervisor(renew, heartbeat_seconds=0.01, time_budget_seconds=5) as supervisor:
        if reason == "deadline":
            supervisor.deadline = 0
        else:
            fail.set()
            assert ready.wait(5)
            # Serialize with the background callback so its exception has been
            # captured before asserting the public checkpoint result.
            supervisor._thread.join(timeout=5)
        with pytest.raises(expected):
            supervisor.poll()
