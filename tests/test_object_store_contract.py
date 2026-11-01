"""The same immutable object behavior against local and actual S3 storage."""

import hashlib
import os
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier
from typing import Protocol

import pytest

from regimex.infrastructure.object_store import (
    LocalObjectStore,
    ObjectCapacityError,
    ObjectConflictError,
    ObjectInfo,
    ObjectIntegrityError,
    ObjectStore,
)
from regimex.infrastructure.s3_config import S3Config
from regimex.infrastructure.s3_object_store import S3ObjectStore, _Client, _rows


class RawClient(_Client, Protocol):
    def create_multipart_upload(self, **values: object) -> dict[str, object]: ...
    def upload_part(self, **values: object) -> dict[str, object]: ...
    def put_bucket_versioning(self, **values: object) -> dict[str, object]: ...


@dataclass(frozen=True)
class Backend:
    name: str
    root: Path
    create: Callable[[int, int], ObjectStore]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source(root: Path, data: bytes = b"payload", name: str = "source") -> Path:
    path = root / name
    path.write_bytes(data)
    return path


def s3_configuration(root: Path) -> S3Config:
    endpoint = os.environ.get("REGIMEX_TEST_S3_ENDPOINT")
    bucket = os.environ.get("REGIMEX_TEST_S3_BUCKET")
    if not endpoint or not bucket:
        pytest.skip("set a disposable REGIMEX_TEST_S3_ENDPOINT and REGIMEX_TEST_S3_BUCKET")
    pytest.importorskip("boto3")
    ca_bundle = os.environ.get("REGIMEX_TEST_S3_CA_BUNDLE")
    return S3Config(
        endpoint_url=endpoint,
        region=os.environ.get("REGIMEX_TEST_S3_REGION", "us-east-1"),
        bucket=bucket,
        prefix="regimex-test/" + uuid.uuid4().hex,
        lock_root=root,
        allow_loopback_http=True,
        ca_bundle=Path(ca_bundle) if ca_bundle else None,
        max_attempts=2,
    )


def cleanup_owned_test_prefix(objects: S3ObjectStore) -> None:
    """Tests own this fresh namespace; never use this helper for application GC."""
    client = objects._sdk().client
    arguments = {"Bucket": objects.config.bucket, "Prefix": objects.config.prefix + "/"}
    for upload in _rows(client.list_multipart_uploads(**arguments).get("Uploads", [])):
        client.abort_multipart_upload(
            Bucket=objects.config.bucket, Key=upload["Key"], UploadId=upload["UploadId"]
        )
    # Test-created objects are small; remove exact versions and owned delete markers.
    for _ in range(100):
        response = client.list_object_versions(**arguments, MaxKeys=1000)
        rows = _rows(response.get("Versions", [])) + _rows(response.get("DeleteMarkers", []))
        if not rows:
            return
        for row in rows:
            client.delete_object(
                Bucket=objects.config.bucket, Key=row["Key"], VersionId=row["VersionId"]
            )
    raise AssertionError("test cleanup exceeded its bounded namespace size")


@pytest.fixture(params=("local", "s3"))
def backend(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Backend]:
    root = tmp_path / "objects"
    if request.param == "local":
        yield Backend(
            "local",
            root,
            lambda object_limit, total_limit: LocalObjectStore(
                root, max_object_bytes=object_limit, max_total_bytes=total_limit
            ),
        )
        return
    config = s3_configuration(root)
    created: list[S3ObjectStore] = []

    def create(object_limit: int, total_limit: int) -> ObjectStore:
        objects = S3ObjectStore(config, max_object_bytes=object_limit, max_total_bytes=total_limit)
        created.append(objects)
        return objects

    yield Backend("s3", root, create)
    if created:
        cleanup_owned_test_prefix(created[0])


def test_immutable_lifecycle(backend: Backend, tmp_path: Path) -> None:
    objects = backend.create(1024, 4096)
    upload = source(tmp_path)
    info = objects.put("nested/raw", upload, expected_sha256=digest(b"payload"))
    assert ObjectInfo.model_validate_json(info.model_dump_json()) == info
    assert info.size_bytes == 7 and info.sha256 == digest(b"payload")
    assert objects.stat(info.key) == info
    assert objects.put(info.key, upload, expected_sha256=info.sha256) == info
    upload.write_bytes(b"source changed after upload")
    destination = tmp_path / "materialized" / "raw"
    assert (
        objects.materialize(
            info.key, destination, expected_sha256=info.sha256, expected_size_bytes=info.size_bytes
        )
        == info
    )
    assert destination.read_bytes() == b"payload"
    destination.write_bytes(b"independent download mutation")
    assert objects.stat(info.key) == info
    objects.delete(info)
    with pytest.raises(FileNotFoundError):
        objects.stat(info.key)
    assert not tuple(tmp_path.rglob(".regimex-temporary-*"))


def test_empty_payload_with_zero_capacity(backend: Backend, tmp_path: Path) -> None:
    objects = backend.create(0, 0)
    info = objects.put("empty", source(tmp_path, b""), expected_sha256=digest(b""))
    assert info.size_bytes == 0
    objects.delete(info)


def test_conflict_and_wrong_source_keep_original(backend: Backend, tmp_path: Path) -> None:
    objects = backend.create(1024, 4096)
    upload = source(tmp_path)
    info = objects.put("fixed", upload, expected_sha256=digest(b"payload"))
    upload.write_bytes(b"other")
    with pytest.raises(ObjectConflictError):
        objects.put(info.key, upload, expected_sha256=digest(b"other"))
    with pytest.raises(ObjectIntegrityError):
        objects.put(info.key, upload, expected_sha256=info.sha256)
    assert objects.stat(info.key) == info


@pytest.mark.parametrize("wrong", ("digest", "size"))
def test_failed_materialization_exposes_no_destination(
    backend: Backend, tmp_path: Path, wrong: str
) -> None:
    objects = backend.create(1024, 4096)
    info = objects.put("verified", source(tmp_path), expected_sha256=digest(b"payload"))
    destination = tmp_path / "not-committed"
    with pytest.raises(ObjectIntegrityError):
        objects.materialize(
            info.key,
            destination,
            expected_sha256="0" * 64 if wrong == "digest" else info.sha256,
            expected_size_bytes=info.size_bytes + 1 if wrong == "size" else info.size_bytes,
        )
    assert not destination.exists()
    assert not tuple(tmp_path.rglob(".regimex-temporary-*"))


def test_materialization_never_overwrites_destination(backend: Backend, tmp_path: Path) -> None:
    objects = backend.create(1024, 4096)
    info = objects.put("verified", source(tmp_path), expected_sha256=digest(b"payload"))
    destination = source(tmp_path, b"keep", "existing")
    linked = tmp_path / "link"
    linked.symlink_to(destination)
    for target in (destination, linked, backend.root / "download"):
        with pytest.raises(ValueError):
            objects.materialize(
                info.key, target, expected_sha256=info.sha256, expected_size_bytes=info.size_bytes
            )
    assert destination.read_bytes() == b"keep"
    assert not (backend.root / "download").exists()


@pytest.mark.parametrize("location", ("source", "source-parent", "destination-parent", "lock"))
def test_local_symlinks_are_rejected(backend: Backend, tmp_path: Path, location: str) -> None:
    objects = backend.create(1024, 4096)
    actual = tmp_path / "actual"
    actual.mkdir()
    upload = source(actual)
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    if location in ("source", "source-parent"):
        if location == "source":
            target = tmp_path / "source-link"
            target.symlink_to(upload)
        else:
            target = linked / upload.name
        with pytest.raises(OSError):
            objects.put("bad", target, expected_sha256=digest(b"payload"))
    elif location == "lock":
        lock = backend.root / ".regimex-lock"
        lock.unlink()
        lock.symlink_to(upload)
        with pytest.raises(OSError):
            objects.stat("bad")
        lock.unlink()
    else:
        info = objects.put("verified", upload, expected_sha256=digest(b"payload"))
        with pytest.raises(OSError):
            objects.materialize(
                info.key,
                linked / "bad",
                expected_sha256=info.sha256,
                expected_size_bytes=info.size_bytes,
            )
        assert not (actual / "bad").exists()


def test_special_file_source_does_not_block(backend: Backend, tmp_path: Path) -> None:
    objects = backend.create(1024, 4096)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular files"):
        objects.put("bad", fifo, expected_sha256=digest(b""))


@pytest.mark.parametrize(
    "key",
    (
        "",
        "/absolute",
        "../bad",
        "a//b",
        "a/./b",
        "a\\b",
        "a:b",
        "a\x00b",
        "a\nb",
        "a\x1fb",
        "a\x7fb",
        ".regimex-lock",
        "a/.regimex-temporary-x",
    ),
)
def test_unsafe_keys_are_rejected(backend: Backend, tmp_path: Path, key: str) -> None:
    objects = backend.create(1024, 4096)
    with pytest.raises(ValueError):
        objects.put(key, source(tmp_path), expected_sha256=digest(b"payload"))
    with pytest.raises(ValueError):
        objects.stat(key)


def test_limits_preserve_previous_commits(backend: Backend, tmp_path: Path) -> None:
    objects = backend.create(4, 6)
    upload = source(tmp_path, b"four")
    info = objects.put("one", upload, expected_sha256=digest(b"four"))
    assert objects.put(info.key, upload, expected_sha256=info.sha256) == info
    with pytest.raises(ObjectCapacityError):
        objects.put("two", upload, expected_sha256=info.sha256)
    upload.write_bytes(b"five!")
    with pytest.raises(ObjectCapacityError):
        objects.put("large", upload, expected_sha256=digest(b"five!"))
    assert objects.stat(info.key) == info


def test_concurrent_instances_share_quota(backend: Backend, tmp_path: Path) -> None:
    objects = (backend.create(4, 4), backend.create(4, 4))
    upload = source(tmp_path, b"four")
    barrier = Barrier(2)

    def run(index: int) -> ObjectInfo | ObjectCapacityError:
        barrier.wait(timeout=10)
        try:
            return objects[index].put(str(index), upload, expected_sha256=digest(b"four"))
        except ObjectCapacityError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(run, (0, 1)))
    assert sum(isinstance(result, ObjectInfo) for result in results) == 1
    assert sum(isinstance(result, ObjectCapacityError) for result in results) == 1


def test_deleted_and_recreated_generation_rejects_stale_delete(
    backend: Backend, tmp_path: Path
) -> None:
    objects = backend.create(1024, 4096)
    upload = source(tmp_path)
    original = objects.put("fixed", upload, expected_sha256=digest(b"payload"))
    objects.delete(original)
    recreated = objects.put(original.key, upload, expected_sha256=original.sha256)
    assert recreated.version_id != original.version_id
    with pytest.raises(ObjectConflictError):
        objects.delete(original)
    assert objects.stat(recreated.key) == recreated


@pytest.mark.parametrize("object_limit,total_limit", ((-1, 4), (1, -1), (True, 4), (5, 4)))
def test_invalid_capacities_fail(backend: Backend, object_limit: int, total_limit: int) -> None:
    with pytest.raises(ValueError):
        backend.create(object_limit, total_limit)
