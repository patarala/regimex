"""Atomic local output and checksummed artifact manifests."""

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


def open_binary_input(
    path: str | Path, *, follow_symlinks: bool = False, dir_fd: int | None = None
) -> BinaryIO:
    """Open without waiting on FIFOs so the caller can inspect the file type."""
    extra_flags = os.O_NONBLOCK if follow_symlinks else os.O_NONBLOCK | os.O_NOFOLLOW
    # Let open own the descriptor even if it rejects the input while constructing the stream.
    return open(
        path, "rb", opener=lambda name, flags: os.open(name, flags | extra_flags, dir_fd=dir_fd)
    )


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def content_hash(value: object) -> str:
    """SHA-256 of the repository's canonical JSON encoding, including its newline."""
    return hashlib.sha256(json_bytes(value)).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_bytes(json_bytes(value))


def write_new_file(path: Path, data: bytes, *, sync_contents: bool = False) -> None:
    """Publish the complete file only if the destination is still free."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete_on_close=False) as temporary:
        temporary.write(data)
        if sync_contents:
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary.close()
        os.link(temporary.name, path)


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
