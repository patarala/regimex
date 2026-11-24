"""Copy the object portion of an offline archive within the recovery deadline."""

import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import nullcontext, suppress
from functools import partial
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from regimex.application.service import materialize_object
from regimex.contracts import StrictModel
from regimex.infrastructure.object_store import (
    ObjectCapacityError,
    ObjectConflictError,
    ObjectInfo,
    ObjectIntegrityError,
    ObjectStore,
)
from regimex.infrastructure.s3_config import S3Config
from regimex.infrastructure.s3_object_store import S3ObjectStore

Transfer = Literal["backup", "check_empty", "restore"]
_VERSIONS = TypeAdapter(tuple[Annotated[str, Field(min_length=1, max_length=256)], ...])
_TIMEOUT = "Offline recovery exceeded its wall-time budget."


class _TransferRequest(StrictModel):
    operation: Transfer
    config: S3Config
    scope_id: str
    max_object_bytes: int
    max_total_bytes: int
    objects: tuple[ObjectInfo, ...]
    archive: Path
    scratch: Path | None
    deadline: float
    parent_pid: int


def check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError(_TIMEOUT)


def _copy(
    operation: Transfer,
    store: ObjectStore,
    objects: tuple[ObjectInfo, ...],
    archive: Path,
    scratch: Path | None,
    checkpoint: Callable[[], None],
) -> dict[str, ObjectInfo]:
    replacements = {}
    checkpoint()
    if operation == "check_empty":
        store.assert_empty_store()
        checkpoint()
        return {}
    for info in objects:
        checkpoint()
        path = archive / "objects" / info.sha256
        if operation == "backup":
            assert scratch is not None
            copy = scratch / "payload"
            # Distinct logical keys can share an archive file. Verify each generation.
            materialize_object(store, info, copy, checkpoint=checkpoint)
            copy.chmod(0o600)
            with copy.open("rb") as stream:
                os.fsync(stream.fileno())
            if not path.exists():
                os.link(copy, path)
            copy.unlink()
        else:
            restored = store.put(info.key, path, expected_sha256=info.sha256, checkpoint=checkpoint)
            if (restored.key, restored.sha256, restored.size_bytes) != (
                info.key,
                info.sha256,
                info.size_bytes,
            ):
                raise ValueError("The target object adapter changed object content identity.")
            if store.stat(restored.key, checkpoint=checkpoint) != restored:
                raise ValueError("The restored object differs from its verified generation.")
            replacements[info.key] = restored
    checkpoint()
    return replacements


def transfer_objects(
    operation: Transfer,
    store: ObjectStore,
    objects: tuple[ObjectInfo, ...],
    archive: Path,
    *,
    deadline: float,
) -> dict[str, ObjectInfo]:
    check_deadline(deadline)
    if not objects and operation != "check_empty":
        return {}
    # The coordinator owns the backup staging directory even if the SDK process dies.
    staging = (
        tempfile.TemporaryDirectory(dir=archive, prefix=".copy-")
        if operation == "backup"
        else nullcontext(None)
    )
    with staging as directory:
        scratch = Path(directory) if directory is not None else None
        if not isinstance(store, S3ObjectStore):
            return _copy(
                operation, store, objects, archive, scratch, partial(check_deadline, deadline)
            )
        request = _TransferRequest(
            operation=operation,
            config=store.config,
            scope_id=store.scope_id,
            max_object_bytes=store.max_object_bytes,
            max_total_bytes=store.max_total_bytes,
            objects=objects,
            archive=archive,
            scratch=scratch,
            deadline=deadline,
            parent_pid=os.getpid(),
        )
        return _run_s3(request)


def _run_s3(request: _TransferRequest) -> dict[str, ObjectInfo]:
    # Isolate credential lookup and SDK calls so the coordinator can stop a
    # preflight or a transfer even while the network is waiting for its first byte.
    payload = request.model_dump_json().encode()
    check_deadline(request.deadline)
    with subprocess.Popen(
        [sys.executable, "-P", "-m", "regimex.application.recovery_objects"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    ) as process:
        try:
            output, _ = process.communicate(
                payload, timeout=max(0, request.deadline - time.monotonic())
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(_TIMEOUT) from None
        finally:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    check_deadline(request.deadline)
    if process.returncode == -signal.SIGALRM:
        raise TimeoutError(_TIMEOUT)
    if process.returncode in _ERRORS:
        raise _ERRORS[process.returncode](output.decode())
    if process.returncode:
        raise OSError("The isolated S3 recovery transfer stopped before completion.")
    # Only new version IDs cross back. Keys, digests and lengths remain pinned by
    # the parent. The worker has verified the restored bytes before returning them.
    versions = _VERSIONS.validate_json(output)
    expected = request.objects if request.operation == "restore" else ()
    return {
        info.key: info.model_copy(update={"version_id": version})
        for info, version in zip(expected, versions, strict=True)
    }


_ERRORS: dict[int, type[Exception]] = {
    40: ObjectConflictError,
    41: ObjectIntegrityError,
    42: ObjectCapacityError,
    43: FileNotFoundError,
    44: PermissionError,
    45: TimeoutError,
    46: ValueError,
    47: OSError,
}


def _worker() -> int:
    from regimex.process_lifetime import bind_to_parent

    request = _TransferRequest.model_validate_json(sys.stdin.buffer.read())
    if not bind_to_parent(request.parent_pid):
        return 70
    try:
        check_deadline(request.deadline)
        if sys.platform != "linux":
            # Linux ties the worker to its coordinator through prctl. On other
            # POSIX hosts a timer also bounds an orphaned transfer.
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            signal.setitimer(signal.ITIMER_REAL, max(1, request.deadline - time.monotonic() + 1))
        store = S3ObjectStore(
            request.config,
            # Offline recovery holds the maintenance lock, quiesces every writer and
            # restores into a fresh prefix before admitting database connections.
            _offline_recovery=True,
            max_object_bytes=request.max_object_bytes,
            max_total_bytes=request.max_total_bytes,
        )
        if store.scope_id != request.scope_id:
            raise ObjectIntegrityError("S3 coordination directory identity changed")
        restored = _copy(
            request.operation,
            store,
            request.objects,
            request.archive,
            request.scratch,
            lambda: check_deadline(request.deadline),
        )
        versions = tuple(info.version_id for info in restored.values())
        sys.stdout.buffer.write(_VERSIONS.dump_json(versions))
        return 0
    except (OSError, ValueError) as exc:
        for code, error_type in _ERRORS.items():
            if isinstance(exc, error_type):
                sys.stdout.buffer.write(str(exc).encode())
                return code
        raise


if __name__ == "__main__":
    sys.exit(_worker())
