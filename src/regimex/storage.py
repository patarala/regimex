"""Atomic local output and checksummed artifact manifests."""

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def write_json(path: Path, value: object) -> None:
    path.write_bytes(json_bytes(value))


def digest(path: Path, *, checkpoint: Callable[[], None] | None = None) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            if checkpoint is not None:
                checkpoint()
            block = stream.read(1024 * 1024)
            if not block:
                break
            checksum.update(block)
    return checksum.hexdigest()


@contextmanager
def output_directory(target: Path) -> Iterator[Path]:
    if target.is_symlink() or (target.exists() and (not target.is_dir() or any(target.iterdir()))):
        raise ValueError(f"output location already exists and is not empty: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        yield temporary
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
