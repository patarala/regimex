"""Isolate computation from credentials, commits and serving workers."""

import argparse
import ctypes
import errno
import math
import os
import resource
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import JsonValue, TypeAdapter

from regimex.contracts import MAX_INPUT_BYTES
from regimex.infrastructure.object_store import ObjectCapacityError
from regimex.model_registry import RegistryError
from regimex.operations import OperationFailure
from regimex.resource_limits import ResourceBudgetExceeded
from regimex.storage import json_bytes

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


def _memory_mapping_failure(error: Exception) -> bool:
    """The dynamic loader wraps its mmap failure in ImportError without errno."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ImportError) and "failed to map segment from shared object" in str(
            current
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _failure_from_exception(error: Exception) -> OperationFailure:
    """Keep resource exhaustion separate from malformed data and unsafe details."""
    if (
        isinstance(error, (ResourceBudgetExceeded, ObjectCapacityError))
        or _memory_mapping_failure(error)
        or (
            isinstance(error, OSError)
            and error.errno in {errno.ENOMEM, errno.ENOSPC, errno.EDQUOT, errno.EFBIG}
        )
    ):
        return OperationFailure(
            code="capacity_limit",
            message="Computation exceeded its resource budget or available capacity.",
        )
    if isinstance(error, (RegistryError, ImportError)):
        return OperationFailure(
            code="unsupported_configuration",
            message="The requested backend is unavailable or incompatible.",
        )
    if isinstance(error, ValueError):
        return OperationFailure(
            code="invalid_data",
            message="The computation inputs or saved artifact failed validation.",
        )
    return OperationFailure(
        code="infrastructure_failure",
        message="The isolated computation could not complete.",
        retryable=isinstance(error, OSError),
    )


class ComputationFailure(RuntimeError):
    def __init__(self, failure: OperationFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


class ComputationCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ChildLimits:
    seconds: int
    memory_bytes: int
    scratch_bytes: int
    threads: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (
                self.seconds,
                self.memory_bytes,
                self.scratch_bytes,
                self.threads,
            )
        ):
            raise ValueError("child computation limits must be positive integers")


def _scratch_size(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            info = path.lstat()
        except FileNotFoundError:
            # The child publishes temporary outputs by atomic rename. Entries
            # can disappear during a live scan; the post-exit scan is complete.
            continue
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
        elif not stat.S_ISDIR(info.st_mode):
            raise ValueError("computation scratch may contain only regular files and directories")
    return total


def _stop_child(child: subprocess.Popen[bytes]) -> None:
    # The session leader can finish before its descendants. Keep the grace
    # period and forced cleanup scoped to the group created for this child.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait(timeout=5)
        return

    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        child.poll()
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            child.wait(timeout=5)
            return
        time.sleep(0.05)
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=5)


def supervise_child(
    payload: dict[str, JsonValue],
    root: Path,
    *,
    limits: ChildLimits,
    heartbeat: Callable[[], None],
    task: Literal["administration", "analysis"] = "administration",
) -> dict[str, JsonValue]:
    """Run one fresh child and terminate it before returning on any parent failure.

    Memory is an address-space limit, rather than a claim about container RSS.
    Scratch accounting includes materialized inputs and output files. The child
    inherits a small allowlist of environment settings and never receives service
    credentials; installed backends remain trusted Python code, not a sandbox.
    """
    if sys.platform != "linux":
        raise ComputationFailure(
            OperationFailure(
                code="unsupported_configuration", message="Isolated computation requires Linux."
            )
        )
    request_data = json_bytes(payload)
    if _scratch_size(root) + len(request_data) > limits.scratch_bytes:
        raise ComputationFailure(
            OperationFailure(
                code="capacity_limit",
                message="Materialized inputs and request metadata exceed the scratch budget.",
            )
        )
    request = root / "child-request.json"
    request.write_bytes(request_data)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "PATH",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "PYTHONPATH",
            "PYTHONUTF8",
        }
    }
    environment.update(
        {
            name: str(limits.threads)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
    )
    command = [
        sys.executable,
        "-m",
        "regimex.computation_worker",
        "--task",
        task,
        "--child",
        str(request),
        "--root",
        str(root),
        "--parent",
        str(os.getpid()),
        "--memory",
        str(limits.memory_bytes),
        "--seconds",
        str(limits.seconds),
        "--scratch",
        str(limits.scratch_bytes),
    ]
    started = time.monotonic()

    def check_limits() -> None:
        if time.monotonic() - started > limits.seconds:
            raise ComputationFailure(
                OperationFailure(
                    code="capacity_limit",
                    message="Computation exceeded its wall-time budget.",
                )
            )
        if _scratch_size(root) > limits.scratch_bytes:
            raise ComputationFailure(
                OperationFailure(
                    code="capacity_limit",
                    message="Computation exceeded its scratch budget.",
                )
            )

    with (root / "child-output.log").open("wb") as log:
        child = subprocess.Popen(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            while child.poll() is None:
                heartbeat()
                check_limits()
                time.sleep(0.1)
        finally:
            _stop_child(child)
        heartbeat()
        # Descendants can finish writing after their leader exits. Stop the
        # whole group before the final budget scan and output validation.
        # Per-file RLIMIT_FSIZE does not bound the aggregate scratch size.
        check_limits()
        if child.returncode != 0:
            failure_path = root / "child-failure.json"
            if failure_path.is_file() and not failure_path.is_symlink():
                with failure_path.open("rb") as stream:
                    failure = OperationFailure.model_validate_json(stream.read(8193))
                raise ComputationFailure(failure)
            code: Literal["capacity_limit", "infrastructure_failure"] = (
                "capacity_limit"
                if child.returncode
                in {
                    -signal.SIGKILL,
                    -signal.SIGXCPU,
                    -signal.SIGXFSZ,
                    72,
                }
                else "infrastructure_failure"
            )
            raise ComputationFailure(
                OperationFailure(
                    code=code,
                    message="The isolated computation process terminated before completion.",
                )
            )
        result_path = root / "child-result.json"
        if result_path.is_symlink() or not result_path.is_file():
            raise ValueError("computation did not write a regular result manifest")
        with result_path.open("rb") as stream:
            data = stream.read(MAX_INPUT_BYTES + 1)
        if len(data) > MAX_INPUT_BYTES:
            raise ValueError("computation result manifest exceeds 16 MiB")
        return _JSON_OBJECT.validate_json(data)


def _child_main(
    request: Path,
    root: Path,
    parent: int,
    memory: int,
    seconds: int,
    scratch: int,
    task: Literal["administration", "analysis"] = "administration",
) -> int:
    # Set the death signal before importing any scientific implementation, and
    # close the race where the parent dies just before prctl is installed.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent:
        return 70
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        cpu_seconds = math.ceil(seconds) * int(os.environ.get("OMP_NUM_THREADS", "1"))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (scratch, scratch))
        ready = root / "child-process.tmp"
        ready.write_bytes(json_bytes({"pid": os.getpid(), "parent": parent}))
        ready.rename(root / "child-process.json")
        if task == "analysis":
            from regimex.application.analysis_computation import compute
        else:
            from regimex.application.admin_computation import compute

        data = _JSON_OBJECT.validate_json(request.read_bytes())
        result = compute(data, root)
        (root / "child-result.json").write_bytes(json_bytes(result))
        return 0
    except MemoryError:
        os._exit(72)
    except Exception as exc:
        failure = _failure_from_exception(exc)
        if task == "analysis" and failure.code == "invalid_data":
            failure = OperationFailure(
                code="incompatible_artifact",
                message="The committed input or model failed integrity and compatibility checks.",
            )
        (root / "child-failure.json").write_bytes(json_bytes(failure.model_dump(mode="json")))
        return 65


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--memory", type=int, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--scratch", type=int, required=True)
    parser.add_argument("--task", choices=("administration", "analysis"), default="administration")
    args = parser.parse_args()
    return _child_main(
        args.child, args.root, args.parent, args.memory, args.seconds, args.scratch, args.task
    )


if __name__ == "__main__":
    raise SystemExit(main())
