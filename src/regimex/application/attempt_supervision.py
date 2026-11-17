"""Keep a claimed operation alive across input, computation and publication."""

import math
import threading
import time
from collections.abc import Callable
from types import TracebackType

from regimex.computation_worker import ComputationCancelled
from regimex.operations import Operation
from regimex.resource_limits import TimeBudgetExceeded


class AttemptSupervisor:
    """Renew the lease independently of blocking work and propagate lost ownership.

    Checkpoints run in the calling thread; the heartbeat thread never publishes
    results. Repository commits must still fence the attempt token atomically.
    """

    def __init__(
        self,
        heartbeat: Callable[[str], Operation],
        *,
        heartbeat_seconds: float,
        time_budget_seconds: float,
        stage: str = "loading",
    ) -> None:
        self._heartbeat = heartbeat
        self._interval = heartbeat_seconds
        self.deadline = time.monotonic() + time_budget_seconds
        self._stage = stage
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="regimex-attempt-heartbeat"
        )

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.check()
            except Exception as exc:
                self._error = exc
                return

    def poll(self) -> None:
        """Stop at the next I/O or computation checkpoint after cancellation/loss."""
        if self._error is not None:
            raise self._error
        if time.monotonic() >= self.deadline:
            raise TimeBudgetExceeded("The operation exceeded its wall-time budget.")

    def check(self, stage: str | None = None) -> None:
        """Refresh the lease immediately, serializing with background refreshes."""
        with self._lock:
            self.poll()
            if stage is not None:
                self._stage = stage
            operation = self._heartbeat(self._stage)
            if operation.state == "cancel_requested":
                raise ComputationCancelled()
            self.poll()

    @property
    def remaining_seconds(self) -> int:
        self.poll()
        return max(1, math.ceil(self.deadline - time.monotonic()))

    def __enter__(self) -> AttemptSupervisor:
        self.check()
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        # The repository callback has connection/statement timeouts. Finish its
        # in-flight call before releasing this attempt; do not leave one daemon
        # heartbeat behind for every subsequently processed job.
        self._thread.join()
