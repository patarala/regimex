import os
from pathlib import Path
from threading import Event
from uuid import uuid4

import pytest
from pydantic import JsonValue

from regimex.application import training_progress as progress


def test_reader_advances_only_past_complete_bounded_rows(tmp_path: Path) -> None:
    path = tmp_path / progress.PROGRESS_FILENAME
    assert progress._read_batch(path, 0) == ([], 0)
    first = b'{"epoch":1,"training_loss":0}\n'
    path.write_bytes(first + b'{"epoch"')
    rows, offset = progress._read_batch(path, 0)
    assert rows == [{"epoch": 1, "training_loss": 0}] and offset == len(first)
    with path.open("ab") as stream:
        stream.write(b":2}\n")
    assert progress._read_batch(path, offset) == ([{"epoch": 2}], path.stat().st_size)

    path.write_bytes(b"x" * (progress._MAX_LINE_BYTES + 1))
    with pytest.raises(ValueError, match="size limit"):
        progress._read_batch(path, 0)
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular file"):
        progress._read_batch(path, 0)


def test_relay_retries_the_same_rows_before_advancing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    psycopg = pytest.importorskip("psycopg")
    monkeypatch.setattr(progress, "_POLL_SECONDS", 0.01)
    monkeypatch.setattr(progress, "_RETRY_SECONDS", 0.01)
    path = tmp_path / progress.PROGRESS_FILENAME
    path.write_bytes(b'{"epoch":1}\n')
    calls: list[tuple[int, list[dict[str, JsonValue]]]] = []
    done = Event()

    def append(first: int, rows: list[dict[str, JsonValue]]) -> None:
        calls.append((first, rows))
        if len(calls) == 1:
            with path.open("ab") as stream:
                stream.write(b'{"epoch":2}\n')
            raise psycopg.OperationalError("lost acknowledgment")
        if first == 2:
            done.set()

    with progress.relay_training_progress(tmp_path, uuid4(), append):
        assert done.wait(5)
    assert calls == [(1, [{"epoch": 1}]), (1, [{"epoch": 1}]), (2, [{"epoch": 2}])]
