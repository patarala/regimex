"""Bounded operational probes, distinct from scientific worker capability."""

import fcntl
import json
import os
import socket
import ssl
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from http.client import HTTPResponse
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import uuid4

import psycopg
from procrastinate.connector import BaseConnector
from procrastinate.manager import JobManager
from pydantic import AwareDatetime, Field

from regimex.contracts import StrictModel
from regimex.infrastructure.database import DatabaseRepository
from regimex.service_settings import ServiceSettings
from regimex.storage import json_bytes

QueueName = Literal["analysis", "admin", "maintenance"]


class WorkerIdentity(StrictModel):
    worker_id: Annotated[int, Field(ge=1)]
    queue: QueueName
    pid: Annotated[int, Field(ge=1)]
    boot_id: Annotated[str, Field(min_length=1, max_length=128)]
    start_ticks: Annotated[str, Field(pattern=r"^[0-9]{1,32}$")]
    created_at: AwareDatetime


def _process_identity(pid: int) -> tuple[str, str]:
    # Linux procfs start ticks plus the boot identity protect against PID reuse.
    # A PID alone cannot identify the worker that registered a database heartbeat.
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    process = Path(f"/proc/{pid}/stat").read_text()
    fields = process[process.rindex(")") + 2 :].split()
    if fields[0] == "Z":
        raise ValueError("worker process is a zombie")
    return boot, fields[19]


def worker_identity_path(settings: ServiceSettings, queue: QueueName) -> Path:
    return settings.work_root / f"worker-{queue}.json"


def write_worker_identity(settings: ServiceSettings, queue: QueueName, worker_id: int) -> None:
    boot, ticks = _process_identity(os.getpid())
    identity = WorkerIdentity(
        worker_id=worker_id,
        queue=queue,
        pid=os.getpid(),
        boot_id=boot,
        start_ticks=ticks,
        created_at=datetime.now(UTC),
    )
    target = worker_identity_path(settings, queue)
    temporary = target.with_name(target.name + ".incoming-" + uuid4().hex)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(json_bytes(identity.model_dump(mode="json")))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def read_worker_identity(settings: ServiceSettings, queue: QueueName) -> WorkerIdentity:
    descriptor = os.open(
        worker_identity_path(settings, queue), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    )
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("worker identity must be a private regular file")
        payload = stream.read(4097)
    if len(payload) > 4096:
        raise ValueError("worker identity exceeds its byte limit")
    identity = WorkerIdentity.model_validate_json(payload)
    if identity.queue != queue or _process_identity(identity.pid) != (
        identity.boot_id,
        identity.start_ticks,
    ):
        raise ValueError("worker identity does not describe the current process")
    return identity


def worker_healthy(settings: ServiceSettings, queue: QueueName) -> bool:
    """Verify this container's exact live process and queue-library heartbeat row.

    This deliberately does not claim installed models can execute. Model metadata
    readiness and actual saved-model analysis remain separate checks.
    """
    try:
        identity = read_worker_identity(settings, queue)
        repository = DatabaseRepository(
            settings.database_dsn.get_secret_value(), schema=settings.database_schema
        )
        if not repository.readiness():
            return False
        with repository.connection() as connection:
            row = connection.execute(
                "SELECT last_heartbeat >= now() - %s * interval '1 second' "
                "FROM procrastinate_workers WHERE id=%s",
                (settings.stale_seconds, identity.worker_id),
            ).fetchone()
        return row is not None and row[0] is True
    except OSError, ValueError, psycopg.Error:
        return False


@contextmanager
def worker_health_lease(settings: ServiceSettings, queue: QueueName) -> Iterator[None]:
    """A work directory identifies one live instance of each worker queue."""
    descriptor = os.open(
        settings.work_root / f"worker-{queue}.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("worker health lock must be a private regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                "this queue already has a worker using the configured work directory"
            ) from None
        worker_identity_path(settings, queue).unlink(missing_ok=True)
        yield
    finally:
        os.close(descriptor)


class TrackedJobManager(JobManager):
    """Bind the documented queue manager registration to our local health identity."""

    def __init__(
        self, *, connector: BaseConnector, settings: ServiceSettings, queue: QueueName
    ) -> None:
        super().__init__(connector=connector)
        self.settings = settings
        self.queue = queue

    async def register_worker(self) -> int:
        worker_id = await super().register_worker()
        try:
            write_worker_identity(self.settings, self.queue, worker_id)
        except BaseException:
            await super().unregister_worker(worker_id)
            raise
        return worker_id

    async def unregister_worker(self, worker_id: int) -> None:
        await super().unregister_worker(worker_id)
        target = worker_identity_path(self.settings, self.queue)
        try:
            identity = read_worker_identity(self.settings, self.queue)
        except OSError, ValueError:
            return
        if identity.worker_id == worker_id:
            target.unlink(missing_ok=True)


def api_healthy(
    settings: ServiceSettings,
    *,
    component: Literal["ready", "models"],
    port: int,
    ca_file: Path | None = None,
) -> bool:
    """Probe the local listener while verifying its configured TLS name and CA."""
    origin = urlsplit(settings.auth_policy.origin)
    host = origin.hostname
    assert host is not None
    if origin.scheme != "https" and ca_file is not None:
        raise ValueError("a CA file requires an HTTPS public origin")
    try:
        context = ssl.create_default_context(cafile=None if ca_file is None else str(ca_file))
        with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
            connection = (
                context.wrap_socket(raw, server_hostname=host) if origin.scheme == "https" else raw
            )
            try:
                connection.sendall(
                    (
                        f"GET /health/{component} HTTP/1.1\r\nHost: {origin.netloc}\r\n"
                        "Connection: close\r\n\r\n"
                    ).encode("ascii")
                )
                with HTTPResponse(connection) as response:
                    response.begin()
                    payload = response.read(4097)
                    if response.status != 200 or len(payload) > 4096:
                        return False
                data = json.loads(payload)
                return isinstance(data, dict) and (
                    data.get("database") is True
                    if component == "ready"
                    else data.get("deployment_configured") is True
                )
            finally:
                if connection is not raw:
                    connection.close()
    except OSError, ValueError:
        return False
