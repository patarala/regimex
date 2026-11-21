"""Relay measured training history from the compute child to its claimed attempt."""

import logging
import os
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from threading import Event, Thread
from typing import BinaryIO
from uuid import UUID

from pydantic import JsonValue, TypeAdapter

from regimex.service_contracts import ServiceError, StaleAttempt
from regimex.storage import json_bytes

PROGRESS_FILENAME = "training-progress.jsonl"
_ROW = TypeAdapter(dict[str, JsonValue])
_POLL_SECONDS = 1.0
_RETRY_SECONDS = 5.0
_BATCH_SIZE = 256
_MAX_LINE_BYTES = 4096
_LOG = logging.getLogger("regimex.training")


@contextmanager
def record_training_progress(root: Path) -> Iterator[Callable[[dict[str, JsonValue]], None]]:
    stream: BinaryIO | None = None
    disabled = False

    def record(row: dict[str, JsonValue]) -> None:
        nonlocal stream, disabled
        if disabled:
            return
        try:
            if stream is None:
                stream = (root / PROGRESS_FILENAME).open("xb", buffering=0)
            data = json_bytes(row)
            if len(data) > _MAX_LINE_BYTES or stream.write(data) != len(data):
                disabled = True
        except OSError, ValueError, TypeError:
            # The model and its final history can still be saved if live reporting fails.
            disabled = True

    try:
        yield record
    finally:
        if stream is not None:
            with suppress(OSError):
                stream.close()


def _read_batch(path: Path, offset: int) -> tuple[list[dict[str, JsonValue]], int]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except FileNotFoundError:
        return [], offset
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size < offset:
            raise ValueError("Training progress must be an append-only regular file.")
        stream.seek(offset)
        rows = []
        for _ in range(_BATCH_SIZE):
            line = stream.readline(_MAX_LINE_BYTES + 1)
            if len(line) > _MAX_LINE_BYTES:
                raise ValueError("Training progress row exceeds its size limit.")
            if not line.endswith(b"\n"):
                break
            rows.append(_ROW.validate_json(line, strict=True))
            offset += len(line)
        return rows, offset


@contextmanager
def relay_training_progress(
    root: Path,
    run_id: UUID,
    append: Callable[[int, list[dict[str, JsonValue]]], None],
) -> Iterator[None]:
    import psycopg

    # Keep the supervisor's cancellation, deadline and scratch checks independent
    # of database I/O. The append transaction is bounded and fenced by the worker.
    stop = Event()

    def relay() -> None:
        path = root / PROGRESS_FILENAME
        offset, end, next_cursor = 0, 0, 1
        rows: list[dict[str, JsonValue]] = []
        delay = _POLL_SECONDS
        while not stop.wait(delay):
            try:
                if not rows:
                    rows, end = _read_batch(path, offset)
                if rows:
                    append(next_cursor, rows)
                    # Keep the exact batch until its transaction is acknowledged.
                    next_cursor += len(rows)
                    offset = end
                delay = 0 if len(rows) == _BATCH_SIZE else _POLL_SECONDS
                rows = []
            except psycopg.OperationalError, psycopg.InterfaceError:
                if delay != _RETRY_SECONDS:
                    _LOG.warning("Live training metrics temporarily unavailable for run %s", run_id)
                delay = _RETRY_SECONDS
            except StaleAttempt:
                return
            except OSError, ValueError, psycopg.Error, ServiceError:
                _LOG.warning("Stopped relaying live training metrics for run %s", run_id)
                return

    thread = Thread(target=relay, name="regimex-training-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
