"""Immutable S3 objects, initially verified against native SeaweedFS 4.48.

Versioning must remain enabled. PostgreSQL-coordinated nodes share a durable
reservation before issuing each tagged conditional PUT. A standalone adapter
uses the local coordination directory and a current provider inventory.
All writers require exclusive use of the configured bucket prefix.

No method establishes application ownership or permission to delete. The caller
must authorize ordinary deletion, orphan-version deletion and multipart cleanup
against PostgreSQL references and the appropriate grace policy before calling.
"""

import base64
import fcntl
import hashlib
import io
import json
import os
import stat
import time
from collections.abc import Buffer, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, BinaryIO, Protocol, cast

from regimex.infrastructure.object_store import (
    ObjectCapacityError,
    ObjectConflictError,
    ObjectInfo,
    ObjectIntegrityError,
    _abort_local_upload,
    _absolute_parts,
    _anonymous_temporary_file,
    _check_digest,
    _child_directory,
    _incomplete_uploads,
    _key_parts,
    _local_parent,
    _regular_file,
    _stream,
    _temporary_file,
)
from regimex.infrastructure.s3_config import S3Config, S3MultipartUpload, S3ObjectVersion

if TYPE_CHECKING:
    from regimex.infrastructure.maintenance import Connection, ServicesStoppedAttestation
    from regimex.infrastructure.s3_coordination import PendingWrite, S3Coordinator

_CHUNK_BYTES = 1024 * 1024
_MAX_OBJECT_BYTES = 512 * 1024**2
_MAX_LIST_PAGES = 1000
_RECONCILE_BATCH_SIZE = 1000
_LOCK_NAME = ".regimex-lock"
_BINDING_NAME = ".regimex-s3-binding.json"
_LOCK_TIMEOUT_SECONDS = 5


class _Events(Protocol):
    def register(self, event_name: str, handler: Callable[..., None]) -> None: ...


class _ClientMetadata(Protocol):
    events: _Events


class _Client(Protocol):
    meta: _ClientMetadata

    def get_bucket_versioning(self, **values: object) -> dict[str, object]: ...
    def head_object(self, **values: object) -> dict[str, object]: ...
    def get_object(self, **values: object) -> dict[str, object]: ...
    def put_object(self, **values: object) -> dict[str, object]: ...
    def delete_object(self, **values: object) -> dict[str, object]: ...
    def list_object_versions(self, **values: object) -> dict[str, object]: ...
    def list_multipart_uploads(self, **values: object) -> dict[str, object]: ...
    def abort_multipart_upload(self, **values: object) -> dict[str, object]: ...


class _Session(Protocol):
    def client(self, service_name: str, **values: object) -> _Client: ...


class _Boto(Protocol):
    def Session(self, *, profile_name: str | None) -> _Session: ...


class _BotocoreConfig(Protocol):
    def Config(self, **values: object) -> object: ...


class _ResponseReader(Protocol):
    def read1(self, amount: int, *, decode_content: bool = False) -> bytes: ...


class _ResponseBody(Protocol):
    def __enter__(self) -> _ResponseReader: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class _Sdk:
    client: _Client
    client_error: type[Exception]
    core_error: type[Exception]
    http_error: type[Exception]
    pid: int


class _PreconditionFailed(ObjectConflictError):
    pass


class _ConditionalConflict(ObjectConflictError):
    pass


@dataclass(frozen=True)
class WriteReconciliation:
    verified: int
    released_missing: int
    pending: int
    next_after: str | None


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ObjectIntegrityError("S3 returned invalid response metadata")
    return cast(dict[str, object], value)


def _single_put_attempt(request: object, **_: object) -> None:
    # Region redirects and custom retry handlers also create a new request.
    context = _mapping(getattr(request, "context", None))
    if _mapping(context.get("retries")).get("attempt") != 1:
        raise OSError("S3 write token cannot be transmitted twice")


def _rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ObjectIntegrityError("S3 returned an invalid inventory")
    return [_mapping(row) for row in value]


def _string(value: object, *, bound: int = 2048, empty: bool = False) -> str:
    if (
        not isinstance(value, str)
        or (not value and not empty)
        or len(value) > bound
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ObjectIntegrityError("S3 returned an invalid object identifier")
    return value


def _version(value: object) -> str:
    version = _string(value, bound=256)
    if version == "null":
        raise ObjectIntegrityError("S3 object has no immutable version")
    return version


def _size(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ObjectIntegrityError("S3 returned an invalid payload size")
    return value


def _time(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ObjectIntegrityError("S3 returned an invalid inventory timestamp")
    return value


def _truncated(value: object) -> bool:
    if type(value) is not bool:
        raise ObjectIntegrityError("S3 returned invalid pagination metadata")
    return value


def _write_all(descriptor: int, block: bytes) -> None:
    pending = memoryview(block)
    while pending:
        written = os.write(descriptor, pending)
        if written <= 0:
            raise OSError("object copy made no progress")
        pending = pending[written:]


class _CheckedUpload(io.BufferedReader):
    """Check operation ownership as the SDK consumes or retries upload bytes."""

    def __init__(self, descriptor: int, checkpoint: Callable[[], None] | None) -> None:
        super().__init__(io.FileIO(os.dup(descriptor), "rb"))
        self._checkpoint = checkpoint
        self.failure: Exception | None = None

    def check(self) -> None:
        if self.failure is not None:
            raise self.failure
        if self._checkpoint is not None:
            try:
                self._checkpoint()
            except Exception as exc:
                # HTTP libraries can wrap errors raised while reading a request
                # body. Keep the original cancellation/deadline/fencing error.
                self.failure = exc
                raise

    def read(self, size: int | None = -1) -> bytes:
        self.check()
        return super().read(size)

    def readinto(self, buffer: Buffer) -> int:
        self.check()
        return super().readinto(buffer)


class S3ObjectStore:
    """Version-pinned immutable payloads with explicit writer coordination."""

    def __init__(
        self,
        config: S3Config,
        *,
        max_object_bytes: int,
        max_total_bytes: int,
        coordinator: S3Coordinator | None = None,
        _offline_recovery: bool = False,
    ) -> None:
        if not _offline_recovery and (config.coordination == "postgres") != (
            coordinator is not None
        ):
            raise ValueError("PostgreSQL S3 coordination requires the service database")
        self.coordinator = coordinator
        for limit in (max_object_bytes, max_total_bytes):
            if type(limit) is not int or limit < 0:
                raise ValueError("object-store byte limits must be nonnegative integers")
        if max_object_bytes > max_total_bytes:
            raise ValueError("per-object capacity must not exceed total committed-byte capacity")
        if max_object_bytes > _MAX_OBJECT_BYTES:
            raise ValueError("S3 single-part objects are limited to 512 MiB")
        self.config = S3Config.model_validate_json(config.model_dump_json())
        self.root = self.config.lock_root.absolute()
        self.max_object_bytes = max_object_bytes
        self.max_total_bytes = max_total_bytes
        self._sdk_value: _Sdk | None = None
        self._root_identity: tuple[int, int] | None = None
        with self._locked_root(create=True):
            pass

    @property
    def scope_id(self) -> str:
        """Stable nonsecret identity for plans bound to this exact provider namespace."""
        scope: dict[str, object] = {
            "backend": "s3",
            "endpoint": self.config.endpoint_origin,
            "bucket": self.config.bucket,
            "prefix": self.config.prefix,
        }
        if self.config.coordination == "postgres":
            scope["coordination"] = "postgres"
        else:
            scope.update(
                region=self.config.region,
                lock_root=str(self.root),
                lock_root_identity=self._root_identity,
            )
        identity = json.dumps(scope, sort_keys=True, separators=(",", ":")).encode()
        return "s3:" + hashlib.sha256(identity).hexdigest()

    def _full_key(self, key: str) -> str:
        _key_parts(key)
        combined = self.config.prefix + "/" + key
        if len(combined.encode("utf-8")) > 1024:
            raise ValueError("object key and S3 prefix exceed 1024 UTF-8 bytes")
        return combined

    def _relative_key(self, value: object) -> str:
        key = _string(value, bound=1024)
        prefix = self.config.prefix + "/"
        if not key.startswith(prefix):
            raise ObjectIntegrityError("S3 inventory escaped the configured prefix")
        relative = key[len(prefix) :]
        try:
            self._full_key(relative)
        except ValueError as exc:
            raise ObjectIntegrityError("S3 inventory contains an unsupported object key") from exc
        return relative

    @contextmanager
    def _locked_root(
        self,
        *,
        create: bool = False,
        shared: bool = False,
        checkpoint: Callable[[], None] | None = None,
    ) -> Iterator[int]:
        anchor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        try:
            with _child_directory(anchor, _absolute_parts(self.root), create=create) as root:
                metadata = os.fstat(root)
                identity = (metadata.st_dev, metadata.st_ino)
                if self._root_identity is None:
                    self._root_identity = identity
                elif self._root_identity != identity:
                    raise ObjectIntegrityError("S3 coordination directory identity changed")
                lock = os.open(
                    _LOCK_NAME,
                    os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                    0o600,
                    dir_fd=root,
                )
                try:
                    if not stat.S_ISREG(os.fstat(lock).st_mode):
                        raise ValueError("S3 coordination lock must be a regular file")
                    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
                    while True:
                        if checkpoint is not None:
                            checkpoint()
                        try:
                            mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
                            fcntl.flock(lock, mode | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise TimeoutError(
                                    "S3 coordination lock wait exceeded 5 seconds"
                                ) from None
                            time.sleep(min(0.05, remaining))
                    self._bind(root, create=not shared)
                    yield root
                finally:
                    os.close(lock)
        finally:
            os.close(anchor)

    @contextmanager
    def _access(
        self,
        *,
        shared: bool = False,
        key: str | None = None,
        checkpoint: Callable[[], None] | None = None,
        initialize: bool = False,
    ) -> Iterator[tuple[int, Connection | None]]:
        with self._locked_root(shared=shared, checkpoint=checkpoint) as root:
            if self.coordinator is None:
                yield root, None
                return
            limits = self.max_object_bytes, self.max_total_bytes, self.config.max_inventory_entries
            with self.coordinator.session(
                self.scope_id,
                limits,
                shared=shared,
                key=key,
                checkpoint=checkpoint,
                verify_namespace=self._ready if initialize or key is not None else None,
            ) as connection:
                yield root, connection

    def _bind(self, root: int, *, create: bool) -> None:
        expected = json.dumps(
            {
                "endpoint": self.config.endpoint_origin,
                "region": self.config.region,
                "bucket": self.config.bucket,
                "prefix": self.config.prefix,
                "max_object_bytes": self.max_object_bytes,
                "max_total_bytes": self.max_total_bytes,
                **({"coordination": "postgres"} if self.config.coordination == "postgres" else {}),
            },
            sort_keys=True,
        ).encode()
        try:
            with _regular_file(root, _BINDING_NAME) as descriptor:
                actual = os.read(descriptor, len(expected) + 1)
        except FileNotFoundError:
            if not create:
                raise ObjectIntegrityError("S3 coordination binding disappeared") from None
            with _temporary_file(root) as (temporary, name):
                _write_all(temporary, expected)
                os.fsync(temporary)
                os.link(
                    name, _BINDING_NAME, src_dir_fd=root, dst_dir_fd=root, follow_symlinks=False
                )
            os.fsync(root)
        else:
            if actual != expected:
                raise ValueError("S3 coordination directory is bound to different storage settings")

    def _sdk(self) -> _Sdk:
        if self._sdk_value is not None and self._sdk_value.pid == os.getpid():
            return self._sdk_value
        try:
            boto = cast(_Boto, import_module("boto3"))
            configuration = cast(_BotocoreConfig, import_module("botocore.config"))
            errors = import_module("botocore.exceptions")
            http_errors = import_module("urllib3.exceptions")
        except ImportError as exc:
            raise OSError("install the S3 extra before selecting S3 storage") from exc
        if self.config.ca_bundle is not None:
            with (
                _local_parent(self.config.ca_bundle, create=False) as (parent, name),
                _regular_file(parent, name),
            ):
                pass
        core_error = cast(type[Exception], errors.BotoCoreError)
        client_error = cast(type[Exception], errors.ClientError)
        try:
            session = boto.Session(profile_name=self.config.credential_profile)
            client = session.client(
                "s3",
                endpoint_url=self.config.endpoint_origin,
                region_name=self.config.region,
                verify=str(self.config.ca_bundle) if self.config.ca_bundle else True,
                config=configuration.Config(
                    signature_version="s3v4",
                    connect_timeout=self.config.connect_timeout_seconds,
                    read_timeout=self.config.read_timeout_seconds,
                    retries={"mode": "standard", "total_max_attempts": self.config.max_attempts},
                    s3={"addressing_style": "path"},
                ),
            )
            if self.config.coordination == "postgres":
                client.meta.events.register("request-created.s3.PutObject", _single_put_attempt)
        except core_error:
            raise OSError("S3 client configuration or credentials are unavailable") from None
        self._sdk_value = _Sdk(
            client,
            client_error,
            core_error,
            cast(type[Exception], http_errors.HTTPError),
            os.getpid(),
        )
        return self._sdk_value

    def _request(self, operation: Callable[[], dict[str, object]]) -> dict[str, object]:
        sdk = self._sdk()
        try:
            return operation()
        except sdk.client_error as exc:
            response = _mapping(getattr(exc, "response", {}))
            error = _mapping(response.get("Error", {}))
            code = _string(error.get("Code"), bound=128)
            if code in {"NoSuchKey", "NoSuchVersion", "NoSuchUpload", "NotFound", "404"}:
                raise FileNotFoundError("S3 object or upload was not found") from None
            if code in {"PreconditionFailed", "412"}:
                raise _PreconditionFailed(
                    "S3 conditional operation rejected stale information"
                ) from None
            if code in {"ConditionalRequestConflict", "409"}:
                raise _ConditionalConflict(
                    "S3 conditional operation conflicted with another write"
                ) from None
            if code in {"BadDigest", "InvalidDigest"}:
                raise ObjectIntegrityError("S3 rejected the payload checksum") from None
            if code in {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch", "403"}:
                raise PermissionError("S3 operation is not authorized") from None
            raise OSError("S3 operation failed") from None
        except sdk.core_error:
            raise OSError("S3 transport or credentials are unavailable") from None

    def _ready(self) -> _Client:
        client = self._sdk().client
        versioning = self._request(lambda: client.get_bucket_versioning(Bucket=self.config.bucket))
        if versioning.get("Status") != "Enabled":
            raise ObjectIntegrityError("configured S3 bucket must have versioning enabled")
        return client

    def check(self) -> None:
        """Probe configuration/versioning explicitly; construction performs no network I/O."""
        with self._access(shared=True):
            self._ready()

    def capacity_report(self) -> dict[str, int]:
        with self._access() as (_, db):
            versions = self._inventory(self._ready())
            counts = dict(
                retained_versions=len(versions),
                retained_payload_bytes=sum(v.size_bytes for v in versions),
            )
            if self.coordinator is not None:
                assert db is not None
                size, entries = self.coordinator.pending_usage(db, self.scope_id)
                counts.update(pending_writes=entries, reserved_payload_bytes=size)
            return counts

    def assert_empty_store(self) -> None:
        """Require a fresh namespace for an authorized, quiesced restore target."""
        with self._access() as (root, db):
            if self.coordinator is not None:
                assert db is not None
                if self.coordinator.pending_usage(db, self.scope_id)[1]:
                    raise ObjectConflictError("Restore requires a prefix without pending S3 writes")
            with os.scandir(root) as entries:
                if any(entry.name not in {_LOCK_NAME, _BINDING_NAME} for entry in entries):
                    raise ObjectConflictError("restore requires an empty S3 coordination directory")
            client = self._ready()
            if self._inventory(client) or self._multipart_uploads(client):
                raise ObjectConflictError("restore requires an empty S3 object namespace")

    def _head(self, client: _Client, key: str, version_id: str | None = None) -> dict[str, object]:
        values: dict[str, object] = {"Bucket": self.config.bucket, "Key": self._full_key(key)}
        if version_id is not None:
            values["VersionId"] = _version(version_id)
        response = self._request(lambda: client.head_object(**values))
        if _size(response.get("ContentLength")) > self.max_object_bytes:
            raise ObjectCapacityError("object exceeds the configured per-object byte limit")
        actual_version = _version(response.get("VersionId"))
        if version_id is not None and actual_version != version_id:
            raise ObjectIntegrityError("S3 returned another version than requested")
        return response

    def _read(
        self,
        client: _Client,
        key: str,
        *,
        destination: int | None = None,
        version_id: str | None = None,
        checkpoint: Callable[[], None] | None = None,
    ) -> ObjectInfo:
        if checkpoint is not None:
            checkpoint()
        head = self._head(client, key, version_id)
        version = _version(head.get("VersionId"))
        expected_size = _size(head.get("ContentLength"))
        response = self._request(
            lambda: client.get_object(
                Bucket=self.config.bucket, Key=self._full_key(key), VersionId=version
            )
        )
        body = cast(_ResponseBody, response.get("Body"))
        try:
            # read1() returns after one network read. Waiting for a whole block
            # could delay the cancellation check on a slow but active connection.
            with body as reader:
                if not callable(getattr(reader, "read1", None)):
                    raise ObjectIntegrityError("S3 transport does not support bounded stream reads")
                if (
                    _version(response.get("VersionId")) != version
                    or _size(response.get("ContentLength")) != expected_size
                ):
                    raise ObjectIntegrityError("S3 response differs from the pinned object version")
                size = 0
                checksum = hashlib.sha256()
                while True:
                    if checkpoint is not None:
                        checkpoint()
                    block = reader.read1(
                        min(_CHUNK_BYTES, self.max_object_bytes - size + 1), decode_content=False
                    )
                    if checkpoint is not None:
                        checkpoint()
                    if not block:
                        break
                    if not isinstance(block, bytes):
                        raise ObjectIntegrityError("S3 returned a nonbinary object payload")
                    size += len(block)
                    if size > self.max_object_bytes:
                        raise ObjectCapacityError(
                            "object exceeds the configured per-object byte limit"
                        )
                    if size > expected_size:
                        raise ObjectIntegrityError("S3 payload exceeds its recorded size")
                    checksum.update(block)
                    if destination is not None:
                        _write_all(destination, block)
                if size != expected_size:
                    raise ObjectIntegrityError("S3 payload differs from its recorded size")
        except self._sdk().core_error, self._sdk().http_error:
            raise OSError("S3 payload transfer was interrupted") from None
        return ObjectInfo(key=key, size_bytes=size, sha256=checksum.hexdigest(), version_id=version)

    def put(
        self,
        key: str,
        source: Path,
        *,
        expected_sha256: str,
        checkpoint: Callable[[], None] | None = None,
    ) -> ObjectInfo:
        self._full_key(key)
        _check_digest(expected_sha256)
        if source.absolute().is_relative_to(self.root):
            raise ValueError("upload source must be outside the S3 coordination directory")
        with (
            self._access(
                shared=self.coordinator is not None,
                key=key,
                checkpoint=checkpoint,
            ) as (root, connection),
            _anonymous_temporary_file(root) as temporary,
        ):
            client = self._ready()
            with (
                _local_parent(source, create=False) as (parent, name),
                _regular_file(parent, name) as descriptor,
            ):
                size, digest = _stream(
                    descriptor, temporary, limit=self.max_object_bytes, checkpoint=checkpoint
                )
            if digest != expected_sha256:
                raise ObjectIntegrityError("upload source SHA-256 mismatch")
            if self.coordinator is not None:
                assert connection is not None
                return self._put_reserved(
                    connection, client, key, temporary, size, digest, checkpoint
                )
            try:
                existing = self._read(client, key, checkpoint=checkpoint)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                return self._same_payload(existing, size, digest)
            versions = self._inventory(client, checkpoint=checkpoint)
            if len(versions) >= self.config.max_inventory_entries:
                raise ObjectCapacityError("S3 inventory has reached its configured entry limit")
            if sum(item.size_bytes for item in versions) + size > self.max_total_bytes:
                raise ObjectCapacityError("object store exceeds committed-payload capacity")
            os.fsync(temporary)
            with _CheckedUpload(temporary, checkpoint) as upload:
                try:
                    return self._put_stable(
                        client, key, upload, size, digest, checkpoint=checkpoint
                    )
                except Exception:
                    if upload.failure is not None:
                        raise upload.failure from None
                    raise

    def _verify_pending(
        self,
        client: _Client,
        write: PendingWrite,
        info: ObjectInfo,
    ) -> None:
        head = self._request(
            lambda: client.head_object(
                Bucket=self.config.bucket,
                Key=self._full_key(write.key),
                VersionId=info.version_id,
            )
        )
        if (
            _version(head.get("VersionId")) != info.version_id
            or _mapping(head.get("Metadata")).get("regimex-write-id") != str(write.token)
            or (info.sha256, info.size_bytes) != (write.digest, write.size)
        ):
            raise ObjectIntegrityError("S3 version differs from the pending write token or payload")

    def _put_reserved(
        self,
        db: Connection,
        client: _Client,
        key: str,
        temporary: int,
        size: int,
        digest: str,
        checkpoint: Callable[[], None] | None,
    ) -> ObjectInfo:
        coordinator = self.coordinator
        assert coordinator is not None
        pending = coordinator.pending(db, self.scope_id, key)
        try:
            existing = self._read(client, key, checkpoint=checkpoint)
        except FileNotFoundError:
            existing = None
        if pending is not None:
            if existing is None:
                raise ObjectConflictError(
                    "An earlier S3 write is unconfirmed; reconcile it after the provider drains"
                )
            self._verify_pending(client, pending, existing)
            coordinator.finish(db, self.scope_id, pending)
        if existing is not None:
            return self._same_payload(existing, size, digest)

        def usage() -> tuple[int, int]:
            versions = self._inventory(client, checkpoint=checkpoint)
            return sum(item.size_bytes for item in versions), len(versions)

        os.fsync(temporary)
        with _CheckedUpload(temporary, checkpoint) as upload:
            upload.seek(0)
            upload.check()
            write = coordinator.reserve(
                db,
                self.scope_id,
                key,
                digest,
                size,
                usage,
                byte_limit=self.max_total_bytes,
                entry_limit=self.config.max_inventory_entries,
                checkpoint=checkpoint,
            )
            # Commit the admission before sending bytes. Never resend this token: once
            # its version exists another node can conclusively recover a lost response.
            try:
                response = self._send_put(
                    client,
                    key,
                    upload,
                    size,
                    digest,
                    metadata={"regimex-write-id": str(write.token)},
                )
            except Exception:
                if upload.failure is not None:
                    raise upload.failure from None
                raise
            info = self._read(
                client, key, version_id=_version(response.get("VersionId")), checkpoint=checkpoint
            )
            self._verify_pending(client, write, info)
            coordinator.finish(db, self.scope_id, write)
            return info

    def _send_put(
        self,
        client: _Client,
        key: str,
        upload: BinaryIO,
        size: int,
        digest: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, object]:
        return self._request(
            lambda: client.put_object(
                Bucket=self.config.bucket,
                Key=self._full_key(key),
                Body=upload,
                ContentLength=size,
                ContentType="application/octet-stream",
                ChecksumSHA256=base64.b64encode(bytes.fromhex(digest)).decode("ascii"),
                IfNoneMatch="*",
                **({"Metadata": metadata} if metadata is not None else {}),
            )
        )

    @staticmethod
    def _same_payload(info: ObjectInfo, size: int, digest: str) -> ObjectInfo:
        if (info.size_bytes, info.sha256) != (size, digest):
            raise ObjectConflictError("immutable object key already has other content")
        return info

    def _put_stable(
        self,
        client: _Client,
        key: str,
        upload: BinaryIO,
        size: int,
        digest: str,
        *,
        checkpoint: Callable[[], None] | None = None,
    ) -> ObjectInfo:
        # Offline restores to a distributed prefix have no database reservation.
        # Sending once per key avoids a late retry recreating a deleted version.
        attempts = 1 if self.config.coordination == "postgres" else self.config.max_attempts
        for attempt in range(attempts):
            if checkpoint is not None:
                checkpoint()
            upload.seek(0)
            try:
                response = self._send_put(client, key, upload, size, digest)
            except OSError, _PreconditionFailed, _ConditionalConflict:
                try:
                    existing = self._read(client, key, checkpoint=checkpoint)
                except FileNotFoundError:
                    if attempt + 1 == attempts:
                        raise OSError("S3 conditional upload did not complete") from None
                    continue
                except OSError:
                    raise OSError("S3 upload outcome could not be verified") from None
                return self._same_payload(existing, size, digest)
            else:
                version = _version(response.get("VersionId"))
                info = self._read(client, key, version_id=version, checkpoint=checkpoint)
                if (info.size_bytes, info.sha256) != (size, digest):
                    raise ObjectIntegrityError("uploaded S3 version failed integrity verification")
                return info
        raise AssertionError("unreachable bounded S3 upload loop")

    def stat(self, key: str, *, checkpoint: Callable[[], None] | None = None) -> ObjectInfo:
        self._full_key(key)
        with self._access(shared=True, checkpoint=checkpoint):
            return self._read(self._ready(), key, checkpoint=checkpoint)

    def stat_version(self, key: str, version_id: str) -> ObjectInfo:
        self._full_key(key)
        _version(version_id)
        with self._access(shared=True):
            return self._read(self._ready(), key, version_id=version_id)

    def materialize(
        self,
        key: str,
        destination: Path,
        *,
        expected_sha256: str,
        expected_size_bytes: int,
        checkpoint: Callable[[], None] | None = None,
    ) -> ObjectInfo:
        self._full_key(key)
        _check_digest(expected_sha256)
        if (
            type(expected_size_bytes) is not int
            or not 0 <= expected_size_bytes <= self.max_object_bytes
        ):
            raise ValueError("expected object size exceeds the configured byte limit")
        _absolute_parts(destination)
        if destination.absolute().is_relative_to(self.root):
            raise ValueError(
                "materialization destination must be outside the S3 coordination directory"
            )
        with (
            self._access(shared=True, checkpoint=checkpoint),
            _local_parent(destination, create=True) as (parent, name),
        ):
            try:
                os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise ObjectConflictError("materialization destination already exists")
            with _temporary_file(parent) as (temporary, temporary_name):
                info = self._read(self._ready(), key, destination=temporary, checkpoint=checkpoint)
                if (info.size_bytes, info.sha256) != (expected_size_bytes, expected_sha256):
                    raise ObjectIntegrityError(
                        "stored object differs from expected size or SHA-256"
                    )
                os.fsync(temporary)
                if checkpoint is not None:
                    checkpoint()
                try:
                    os.link(
                        temporary_name,
                        name,
                        src_dir_fd=parent,
                        dst_dir_fd=parent,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    raise ObjectConflictError(
                        "materialization destination already exists"
                    ) from None
            os.fsync(parent)
            return info

    def delete(self, expected: ObjectInfo) -> None:
        expected = ObjectInfo.model_validate_json(expected.model_dump_json())
        self._full_key(expected.key)
        with self._access(initialize=True) as (_, db):
            self._require_settled(db, expected.key)
            client = self._ready()
            if self._read(client, expected.key) != expected:
                raise ObjectConflictError("conditional deletion received stale object information")
            self._delete_version(client, expected)

    def delete_version(self, expected: ObjectInfo) -> None:
        """Delete a verified orphan version after caller authorization/reference checks."""
        expected = ObjectInfo.model_validate_json(expected.model_dump_json())
        self._full_key(expected.key)
        with self._access(initialize=True) as (_, db):
            self._require_settled(db, expected.key)
            client = self._ready()
            if self._read(client, expected.key, version_id=expected.version_id) != expected:
                raise ObjectConflictError("version deletion received stale object information")
            self._delete_version(client, expected)

    def reconcile_pending_writes(
        self,
        attestation: ServicesStoppedAttestation,
        *,
        provider_drained: bool,
        after_key: str | None = None,
    ) -> WriteReconciliation:
        """Settle up to 1000 reservations in a quiesced deployment, without deleting objects."""
        from regimex.infrastructure.maintenance import (
            ServicesStoppedAttestation,
            maintenance_lock,
            quiescent_transaction,
        )

        attestation = ServicesStoppedAttestation.model_validate_json(attestation.model_dump_json())
        if attestation.stopped_at > datetime.now(UTC):
            raise ValueError("The stopped-services attestation cannot come from the future")
        coordinator = self.coordinator
        if coordinator is None:
            raise ValueError("Write reconciliation requires PostgreSQL S3 coordination")
        if after_key is not None:
            self._full_key(after_key)
        schema = coordinator.database.schema
        with self._access() as (_, db):
            assert db is not None
            maintenance_lock(db, schema)
            verified = missing = 0
            with quiescent_transaction(db, schema=schema):
                client = self._ready()
                keys = db.execute(
                    "SELECT object_key FROM regimex_s3_pending_writes WHERE scope_id=%s "
                    "AND object_key>%s ORDER BY object_key LIMIT %s",
                    (self.scope_id, after_key or "", _RECONCILE_BATCH_SIZE + 1),
                ).fetchall()
                page = keys[:_RECONCILE_BATCH_SIZE]
                for (raw_key,) in page:
                    key = cast(str, raw_key)
                    write = coordinator.pending(db, self.scope_id, key)
                    assert write is not None
                    try:
                        info = self._read(client, key)
                    except FileNotFoundError:
                        if not provider_drained:
                            continue
                        missing += 1
                    else:
                        self._verify_pending(client, write, info)
                        verified += 1
                    coordinator.finish(db, self.scope_id, write)
                remaining = db.execute(
                    "SELECT count(*) FROM regimex_s3_pending_writes WHERE scope_id=%s",
                    (self.scope_id,),
                ).fetchone()
                assert remaining is not None
            next_after = cast(str, page[-1][0]) if len(keys) > len(page) else None
            return WriteReconciliation(verified, missing, cast(int, remaining[0]), next_after)

    def _require_settled(self, db: Connection | None, key: str) -> None:
        if self.coordinator is not None:
            assert db is not None
            if self.coordinator.pending(db, self.scope_id, key) is not None:
                raise ObjectConflictError("Reconcile the pending S3 write before deleting its key")

    def _delete_version(self, client: _Client, expected: ObjectInfo) -> None:
        self._request(
            lambda: client.delete_object(
                Bucket=self.config.bucket,
                Key=self._full_key(expected.key),
                VersionId=expected.version_id,
            )
        )

    def inventory(self) -> tuple[S3ObjectVersion, ...]:
        """Inventory retained versions/markers; metadata does not attest their SHA256."""
        with self._access():
            return self._inventory(self._ready())

    def _inventory(
        self, client: _Client, *, checkpoint: Callable[[], None] | None = None
    ) -> tuple[S3ObjectVersion, ...]:
        arguments: dict[str, object] = {
            "Bucket": self.config.bucket,
            "Prefix": self.config.prefix + "/",
            "MaxKeys": 1000,
        }
        result: list[S3ObjectVersion] = []
        seen: set[tuple[str, str]] = set()
        previous_marker: tuple[str, str] | None = None
        for _ in range(_MAX_LIST_PAGES):
            if checkpoint is not None:
                checkpoint()
            response = self._request(lambda: client.list_object_versions(**arguments))
            for marker, field in ((False, "Versions"), (True, "DeleteMarkers")):
                for row in _rows(response.get(field, [])):
                    key = self._relative_key(row.get("Key"))
                    version = _version(row.get("VersionId"))
                    identity = (key, version)
                    if identity in seen:
                        raise ObjectIntegrityError("S3 inventory repeated an object version")
                    seen.add(identity)
                    result.append(
                        S3ObjectVersion(
                            key=key,
                            version_id=version,
                            size_bytes=0 if marker else _size(row.get("Size")),
                            last_modified=_time(row.get("LastModified")),
                            is_delete_marker=marker,
                        )
                    )
                    if len(result) > self.config.max_inventory_entries:
                        raise ObjectCapacityError("S3 inventory exceeds its configured entry limit")
            if not _truncated(response.get("IsTruncated", False)):
                return tuple(sorted(result, key=lambda item: (item.key, item.version_id)))
            next_marker = (
                _string(response.get("NextKeyMarker"), bound=1024),
                _string(response.get("NextVersionIdMarker"), bound=256, empty=True),
            )
            if next_marker == previous_marker:
                raise ObjectIntegrityError("S3 inventory pagination made no progress")
            previous_marker = next_marker
            arguments.update(KeyMarker=next_marker[0], VersionIdMarker=next_marker[1])
        raise ObjectCapacityError("S3 inventory exceeds the bounded page limit")

    def list_multipart_uploads(self) -> tuple[S3MultipartUpload, ...]:
        with self._access() as (root, _):
            local = _incomplete_uploads(root) if self.coordinator is None else ()
            remote = self._multipart_uploads(self._ready())
            if len(local) + len(remote) > self.config.max_inventory_entries:
                raise ObjectCapacityError("S3 upload inventory exceeds its configured entry limit")
            return tuple(sorted((*local, *remote), key=lambda item: (item.key, item.upload_id)))

    def _multipart_uploads(self, client: _Client) -> tuple[S3MultipartUpload, ...]:
        arguments: dict[str, object] = {
            "Bucket": self.config.bucket,
            "Prefix": self.config.prefix + "/",
            "MaxUploads": 1000,
        }
        result: list[S3MultipartUpload] = []
        seen: set[tuple[str, str]] = set()
        previous_marker: tuple[str, str] | None = None
        for _ in range(_MAX_LIST_PAGES):
            response = self._request(lambda: client.list_multipart_uploads(**arguments))
            for row in _rows(response.get("Uploads", [])):
                key = self._relative_key(row.get("Key"))
                upload = _string(row.get("UploadId"))
                identity = (key, upload)
                if identity in seen:
                    raise ObjectIntegrityError("S3 inventory repeated a multipart upload")
                seen.add(identity)
                result.append(
                    S3MultipartUpload(
                        key=key, upload_id=upload, initiated_at=_time(row.get("Initiated"))
                    )
                )
                if len(result) > self.config.max_inventory_entries:
                    raise ObjectCapacityError(
                        "S3 multipart inventory exceeds its configured entry limit"
                    )
            if not _truncated(response.get("IsTruncated", False)):
                return tuple(sorted(result, key=lambda item: (item.key, item.upload_id)))
            next_marker = (
                _string(response.get("NextKeyMarker"), bound=1024),
                _string(response.get("NextUploadIdMarker"), empty=True),
            )
            if next_marker == previous_marker:
                raise ObjectIntegrityError("S3 multipart pagination made no progress")
            previous_marker = next_marker
            arguments.update(KeyMarker=next_marker[0], UploadIdMarker=next_marker[1])
        raise ObjectCapacityError("S3 multipart inventory exceeds the bounded page limit")

    def abort_multipart(self, expected: S3MultipartUpload, *, initiated_before: datetime) -> None:
        """Abort an old authorized upload; no ownership/grace decision is inferred."""
        expected = S3MultipartUpload.model_validate_json(expected.model_dump_json())
        if expected.upload_id.startswith("local-temporary-v1:"):
            if self.coordinator is not None:
                raise ObjectConflictError("Cluster cleanup cannot address node-local staging files")
            with self._access() as (root, _):
                _abort_local_upload(root, expected, initiated_before=initiated_before)
            return
        self._full_key(expected.key)
        if (
            initiated_before.tzinfo is None
            or initiated_before.utcoffset() is None
            or initiated_before > datetime.now(UTC)
        ):
            raise ValueError("multipart cutoff must be an aware timestamp in the past")
        if expected.initiated_at >= initiated_before:
            raise ObjectConflictError("multipart upload is newer than the authorized cutoff")
        with self._access(initialize=True):
            client = self._ready()
            found = next(
                (
                    item
                    for item in self._multipart_uploads(client)
                    if (item.key, item.upload_id) == (expected.key, expected.upload_id)
                ),
                None,
            )
            if found is None:
                raise FileNotFoundError("multipart upload was not found")
            if found != expected:
                raise ObjectConflictError("multipart cleanup received stale upload information")
            self._request(
                lambda: client.abort_multipart_upload(
                    Bucket=self.config.bucket,
                    Key=self._full_key(expected.key),
                    UploadId=expected.upload_id,
                )
            )
