"""Checked immutable objects on a local filesystem, with no application ownership.

The root must be dedicated to this backend. A process-shared lock serializes
store operations, including conditional deletion and committed-byte accounting.
Filesystem access outside this adapter is not an authorized way to mutate stored
objects. Callers must authorize deletion and establish that an object is unused.
"""

import fcntl
import hashlib
import os
import re
import stat
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Protocol, Self

from pydantic import Field, model_validator

from regimex.contracts import StrictModel
from regimex.infrastructure.object_inventory import IncompleteUpload, StoredVersion
from regimex.storage import json_bytes

_INTERNAL_PREFIX = ".regimex-"
_LOCK_NAME = _INTERNAL_PREFIX + "lock"
_CHUNK_BYTES = 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_VERSION_ATTRIBUTE = "user.regimex.object-version"
_TEMPORARY_NAME = re.compile(r"\.regimex-temporary-[0-9a-f]{32}\Z")
_BINDING_NAME = ".regimex-s3-binding.json"
_LOCK_TIMEOUT_SECONDS = 5.0


class ObjectConflictError(ValueError):
    """An immutable key or an exclusive destination already has other content."""


class ObjectIntegrityError(ValueError):
    """An object or upload source differs from its recorded bytes or revision."""


class ObjectCapacityError(ValueError):
    """The configured per-object or committed-payload capacity would be exceeded."""


def _key_parts(key: str) -> tuple[str, ...]:
    if (
        not isinstance(key, str)
        or not key
        or len(key.encode("utf-8")) > 1024
        or "\\" in key
        or ":" in key
        or any(ord(character) < 32 or ord(character) == 127 for character in key)
    ):
        raise ValueError("object key must be a bounded relative POSIX path")
    parts = tuple(key.split("/"))
    if any(part in {"", ".", ".."} or part.startswith(_INTERNAL_PREFIX) for part in parts):
        raise ValueError("object key contains a reserved or unsafe path component")
    return parts


def _check_digest(value: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError("expected SHA-256 must be 64 lowercase hexadecimal characters")


class ObjectInfo(StrictModel):
    key: str
    size_bytes: Annotated[int, Field(ge=0)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    version_id: Annotated[str, Field(min_length=1, max_length=256)]

    @model_validator(mode="after")
    def safe_key(self) -> Self:
        _key_parts(self.key)
        return self


class ObjectStore(Protocol):
    @property
    def scope_id(self) -> str: ...

    def put(self, key: str, source: Path, *, expected_sha256: str) -> ObjectInfo: ...

    def stat(self, key: str) -> ObjectInfo: ...

    def materialize(
        self,
        key: str,
        destination: Path,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
    ) -> ObjectInfo: ...

    def delete(self, expected: ObjectInfo) -> None: ...

    def inventory(self) -> tuple[StoredVersion, ...]: ...

    def stat_version(self, key: str, version_id: str) -> ObjectInfo: ...

    def delete_version(self, expected: ObjectInfo) -> None: ...

    def list_multipart_uploads(self) -> tuple[IncompleteUpload, ...]: ...

    def abort_multipart(
        self, expected: IncompleteUpload, *, initiated_before: datetime
    ) -> None: ...

    def assert_empty_store(self) -> None: ...


def _absolute_parts(path: Path) -> tuple[str, ...]:
    parts = path.absolute().parts
    if parts[0] != "/" or any(part == ".." for part in parts):
        raise ValueError("local paths must not contain parent traversal")
    return parts[1:]


@contextmanager
def _child_directory(parent: int, parts: tuple[str, ...], *, create: bool) -> Iterator[int]:
    descriptor = os.dup(parent)
    try:
        for part in parts:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _local_parent(path: Path, *, create: bool) -> Iterator[tuple[int, str]]:
    parts = _absolute_parts(path)
    if not parts:
        raise ValueError("local object paths must name a regular file")
    root = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        with _child_directory(root, parts[:-1], create=create) as parent:
            yield parent, parts[-1]
    finally:
        os.close(root)


@contextmanager
def _regular_file(parent: int, name: str) -> Iterator[int]:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("object inputs must be regular files without symbolic links")
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _temporary_file(parent: int) -> Iterator[tuple[int, str]]:
    name = _INTERNAL_PREFIX + "temporary-" + uuid.uuid4().hex
    descriptor = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent)
    try:
        yield descriptor, name
    finally:
        os.close(descriptor)
        os.unlink(name, dir_fd=parent)


def _revision(metadata: os.stat_result) -> str:
    return ":".join(
        str(value)
        for value in (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
    )


def _stream(source: int, destination: int | None, *, limit: int) -> tuple[int, str]:
    before = os.fstat(source)
    if before.st_size > limit:
        raise ObjectCapacityError("object exceeds the configured per-object byte limit")
    checksum = hashlib.sha256()
    size = 0
    while block := os.read(source, min(_CHUNK_BYTES, limit - size + 1)):
        size += len(block)
        if size > limit:
            raise ObjectCapacityError("object exceeds the configured per-object byte limit")
        checksum.update(block)
        if destination is not None:
            pending = memoryview(block)
            while pending:
                written = os.write(destination, pending)
                if written <= 0:
                    raise OSError("object copy made no progress")
                pending = pending[written:]
    if size != before.st_size or _revision(before) != _revision(os.fstat(source)):
        raise ObjectIntegrityError("object source changed while it was being read")
    return size, checksum.hexdigest()


def _stored_version(descriptor: int) -> str:
    try:
        generation = os.getxattr(descriptor, _VERSION_ATTRIBUTE).decode("ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise ObjectIntegrityError("stored object has no readable immutable version") from exc
    if re.fullmatch(r"[0-9a-f]{32}", generation) is None:
        raise ObjectIntegrityError("stored object has an invalid immutable version")
    return generation + ":" + _revision(os.fstat(descriptor))


def _upload_identity(metadata: os.stat_result) -> str:
    value = {"revision": _revision(metadata), "links": metadata.st_nlink}
    return "local-temporary-v1:" + hashlib.sha256(json_bytes(value)).hexdigest()


def _upload_parts(key: str) -> tuple[str, ...]:
    parts = tuple(key.split("/"))
    if len(key.encode("utf-8")) > 1024 or _TEMPORARY_NAME.fullmatch(parts[-1]) is None:
        raise ValueError("local incomplete upload requires an exact reserved temporary name")
    _key_parts("/".join((*parts[:-1], "upload")))
    return parts


def _incomplete_uploads(root: int) -> tuple[IncompleteUpload, ...]:
    """List abandoned local staging files through the incomplete-upload DTO."""
    results: list[IncompleteUpload] = []

    def visit(directory: int, parents: tuple[str, ...]) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                if not parents and entry.name in {_LOCK_NAME, _BINDING_NAME}:
                    with _regular_file(directory, entry.name):
                        pass
                    continue
                if entry.is_symlink():
                    raise ValueError("upload inventory cannot traverse symbolic links")
                key = "/".join((*parents, entry.name))
                if entry.name.startswith(_INTERNAL_PREFIX):
                    _upload_parts(key)
                    with _regular_file(directory, entry.name) as descriptor:
                        metadata = os.fstat(descriptor)
                        if len(results) >= 100_000:
                            raise ObjectCapacityError(
                                "local upload inventory exceeds its entry limit"
                            )
                        results.append(
                            IncompleteUpload(
                                key=key,
                                upload_id=_upload_identity(metadata),
                                initiated_at=datetime.fromtimestamp(
                                    max(metadata.st_mtime, metadata.st_ctime), UTC
                                ),
                            )
                        )
                elif entry.is_dir(follow_symlinks=False):
                    _key_parts(key)
                    with _child_directory(directory, (entry.name,), create=False) as child:
                        visit(child, (*parents, entry.name))
                elif not entry.is_file(follow_symlinks=False):
                    raise ValueError("upload inventory requires regular files and directories")

    visit(root, ())
    return tuple(sorted(results, key=lambda item: (item.key, item.upload_id)))


def _abort_local_upload(
    root: int, expected: IncompleteUpload, *, initiated_before: datetime
) -> None:
    expected = IncompleteUpload.model_validate_json(expected.model_dump_json())
    parts = _upload_parts(expected.key)
    now = datetime.now(UTC)
    if initiated_before.tzinfo is None or initiated_before > now:
        raise ValueError("incomplete upload cutoff must be an aware timestamp in the past")
    if expected.initiated_at >= initiated_before:
        raise ObjectConflictError("the local upload is newer than the authorized cutoff")
    with (
        _child_directory(root, parts[:-1], create=False) as parent,
        _regular_file(parent, parts[-1]) as descriptor,
    ):
        metadata = os.fstat(descriptor)
        observed = datetime.fromtimestamp(max(metadata.st_mtime, metadata.st_ctime), UTC)
        if (
            _upload_identity(metadata) != expected.upload_id
            or observed != expected.initiated_at
            or observed >= initiated_before
            or metadata.st_nlink != 1
        ):
            raise ObjectConflictError(
                "conditional local upload cleanup received stale or linked metadata"
            )
        os.unlink(parts[-1], dir_fd=parent)
        os.fsync(parent)


def _info(key: str, descriptor: int, *, limit: int) -> ObjectInfo:
    size, checksum = _stream(descriptor, None, limit=limit)
    return ObjectInfo(
        key=key,
        size_bytes=size,
        sha256=checksum,
        version_id=_stored_version(descriptor),
    )


def _committed_bytes(root: int) -> int:
    total = 0
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.name.startswith(_INTERNAL_PREFIX):
                continue
            if entry.is_symlink():
                raise ValueError("object store cannot contain symbolic links")
            if entry.is_dir(follow_symlinks=False):
                with _child_directory(root, (entry.name,), create=False) as child:
                    total += _committed_bytes(child)
            elif entry.is_file(follow_symlinks=False):
                total += entry.stat(follow_symlinks=False).st_size
            else:
                raise ValueError("object store must contain only regular files and directories")
    return total


class LocalObjectStore:
    """Immutable local objects; byte limits count payloads, excluding temporary copies.

    Commits fsync the file and containing directory. The opaque revision supports
    conditional deletion across compliant local-store instances; it is not a
    portable content digest or an application reference/authorization check.
    The filesystem must support user extended attributes, file/directory fsync,
    advisory locks and hard links; unsupported operations fail explicitly.
    """

    def __init__(self, root: Path, *, max_object_bytes: int, max_total_bytes: int) -> None:
        for limit in (max_object_bytes, max_total_bytes):
            if type(limit) is not int or limit < 0:
                raise ValueError("object-store byte limits must be nonnegative integers")
        if max_object_bytes > max_total_bytes:
            raise ValueError("per-object capacity must not exceed total committed-byte capacity")
        self.root = root.absolute()
        if self.root == Path("/"):
            raise ValueError(
                "object store requires a dedicated directory below the filesystem root"
            )
        self.max_object_bytes = max_object_bytes
        self.max_total_bytes = max_total_bytes
        _absolute_parts(self.root)
        with self._locked_root(create=True):
            pass

    @property
    def scope_id(self) -> str:
        location = {"backend": "local", "root": str(self.root)}
        return "local:" + hashlib.sha256(json_bytes(location)).hexdigest()

    @contextmanager
    def _locked_root(self, *, create: bool = False) -> Iterator[int]:
        anchor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        try:
            with _child_directory(anchor, _absolute_parts(self.root), create=create) as root:
                lock = os.open(
                    _LOCK_NAME,
                    os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                    0o600,
                    dir_fd=root,
                )
                try:
                    if not stat.S_ISREG(os.fstat(lock).st_mode):
                        raise ValueError("object-store lock must be a regular file")
                    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
                    while True:
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            if time.monotonic() >= deadline:
                                raise TimeoutError(
                                    "local object-store coordination timed out"
                                ) from None
                            time.sleep(0.02)
                    yield root
                finally:
                    os.close(lock)
        finally:
            os.close(anchor)

    def put(self, key: str, source: Path, *, expected_sha256: str) -> ObjectInfo:
        parts = _key_parts(key)
        _check_digest(expected_sha256)
        with (
            self._locked_root() as root,
            _child_directory(root, parts[:-1], create=True) as parent,
        ):
            with _temporary_file(parent) as (temporary, temporary_name):
                with (
                    _local_parent(source, create=False) as (source_parent, source_name),
                    _regular_file(source_parent, source_name) as source_file,
                ):
                    size, checksum = _stream(source_file, temporary, limit=self.max_object_bytes)
                if checksum != expected_sha256:
                    raise ObjectIntegrityError("upload source SHA-256 mismatch")
                try:
                    with _regular_file(parent, parts[-1]) as existing:
                        info = _info(key, existing, limit=self.max_object_bytes)
                except FileNotFoundError:
                    info = None
                if info is not None:
                    if (info.sha256, info.size_bytes) != (checksum, size):
                        raise ObjectConflictError("immutable object key already has other content")
                    return info
                if _committed_bytes(root) + size > self.max_total_bytes:
                    raise ObjectCapacityError("object store exceeds committed-payload capacity")
                os.setxattr(temporary, _VERSION_ATTRIBUTE, uuid.uuid4().hex.encode("ascii"))
                os.fchmod(temporary, 0o444)
                os.fsync(temporary)
                os.link(
                    temporary_name,
                    parts[-1],
                    src_dir_fd=parent,
                    dst_dir_fd=parent,
                    follow_symlinks=False,
                )
            # Unlinking the temporary hard link changes ctime; record the final revision.
            os.fsync(parent)
            with _regular_file(parent, parts[-1]) as committed:
                return _info(key, committed, limit=self.max_object_bytes)

    def stat(self, key: str) -> ObjectInfo:
        parts = _key_parts(key)
        with (
            self._locked_root() as root,
            _child_directory(root, parts[:-1], create=False) as parent,
            _regular_file(parent, parts[-1]) as source,
        ):
            return _info(key, source, limit=self.max_object_bytes)

    def materialize(
        self,
        key: str,
        destination: Path,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
    ) -> ObjectInfo:
        parts = _key_parts(key)
        _check_digest(expected_sha256)
        if (
            type(expected_size_bytes) is not int
            or not 0 <= expected_size_bytes <= self.max_object_bytes
        ):
            raise ValueError("expected object size exceeds the configured byte limit")
        _absolute_parts(destination)
        if destination.absolute().is_relative_to(self.root):
            raise ValueError("materialization destination must be outside the object-store root")
        with (
            self._locked_root() as root,
            _child_directory(root, parts[:-1], create=False) as parent,
            _regular_file(parent, parts[-1]) as source,
        ):
            version = _stored_version(source)
            with _local_parent(destination, create=True) as (target_parent, target_name):
                try:
                    os.stat(target_name, dir_fd=target_parent, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise ObjectConflictError("materialization destination already exists")
                with _temporary_file(target_parent) as (temporary, temporary_name):
                    size, checksum = _stream(source, temporary, limit=self.max_object_bytes)
                    if (size, checksum) != (expected_size_bytes, expected_sha256):
                        raise ObjectIntegrityError(
                            "stored object differs from expected size or SHA-256"
                        )
                    if _stored_version(source) != version:
                        raise ObjectIntegrityError(
                            "stored object version changed during materialization"
                        )
                    os.fsync(temporary)
                    try:
                        os.link(
                            temporary_name,
                            target_name,
                            src_dir_fd=target_parent,
                            dst_dir_fd=target_parent,
                            follow_symlinks=False,
                        )
                    except FileExistsError as exc:
                        raise ObjectConflictError(
                            "materialization destination already exists"
                        ) from exc
                os.fsync(target_parent)
            return ObjectInfo(
                key=key,
                size_bytes=size,
                sha256=checksum,
                version_id=version,
            )

    def delete(self, expected: ObjectInfo) -> None:
        expected = ObjectInfo.model_validate_json(expected.model_dump_json())
        parts = _key_parts(expected.key)
        with (
            self._locked_root() as root,
            _child_directory(root, parts[:-1], create=False) as parent,
            _regular_file(parent, parts[-1]) as source,
        ):
            if _info(expected.key, source, limit=self.max_object_bytes) != expected:
                raise ObjectConflictError("conditional deletion received stale object information")
            os.unlink(parts[-1], dir_fd=parent)
            os.fsync(parent)

    def inventory(self) -> tuple[StoredVersion, ...]:
        """List bounded immutable generations, preserving unverified private files."""
        results: list[StoredVersion] = []

        def visit(directory: int, parents: tuple[str, ...]) -> None:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.name.startswith(_INTERNAL_PREFIX):
                        continue
                    if entry.is_symlink():
                        raise ValueError("object inventory cannot traverse symbolic links")
                    parts = (*parents, entry.name)
                    key = "/".join(parts)
                    _key_parts(key)
                    if entry.is_dir(follow_symlinks=False):
                        with _child_directory(directory, (entry.name,), create=False) as child:
                            visit(child, parts)
                    elif entry.is_file(follow_symlinks=False):
                        if len(results) >= 100_000:
                            raise ObjectCapacityError("local inventory exceeds its entry limit")
                        with _regular_file(directory, entry.name) as descriptor:
                            metadata = os.fstat(descriptor)
                            results.append(
                                StoredVersion(
                                    key=key,
                                    version_id=_stored_version(descriptor),
                                    size_bytes=metadata.st_size,
                                    last_modified=datetime.fromtimestamp(
                                        max(metadata.st_mtime, metadata.st_ctime), UTC
                                    ),
                                )
                            )
                    else:
                        raise ValueError("object inventory requires regular payload files")

        with self._locked_root() as root:
            visit(root, ())
        return tuple(sorted(results, key=lambda item: (item.key, item.version_id)))

    def stat_version(self, key: str, version_id: str) -> ObjectInfo:
        info = self.stat(key)
        if info.version_id != version_id:
            raise ObjectConflictError("the requested local object generation no longer exists")
        return info

    def delete_version(self, expected: ObjectInfo) -> None:
        self.delete(expected)

    def list_multipart_uploads(self) -> tuple[IncompleteUpload, ...]:
        with self._locked_root() as root:
            return _incomplete_uploads(root)

    def abort_multipart(self, expected: IncompleteUpload, *, initiated_before: datetime) -> None:
        with self._locked_root() as root:
            _abort_local_upload(root, expected, initiated_before=initiated_before)

    def assert_empty_store(self) -> None:
        """Reject payloads and private upload remnants before offline restoration."""

        def visit(directory: int, *, root: bool) -> None:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if root and entry.name == _LOCK_NAME:
                        continue
                    if entry.is_symlink():
                        raise ValueError("restore targets cannot contain symbolic links")
                    if entry.name.startswith(_INTERNAL_PREFIX):
                        raise ObjectConflictError("restore target contains private upload remnants")
                    if entry.is_dir(follow_symlinks=False):
                        with _child_directory(directory, (entry.name,), create=False) as child:
                            visit(child, root=False)
                    elif entry.is_file(follow_symlinks=False):
                        raise ObjectConflictError("restore requires an empty object store")
                    else:
                        raise ValueError("restore targets require regular directories")

        with self._locked_root() as root:
            visit(root, root=True)
