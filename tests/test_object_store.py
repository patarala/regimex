"""Local object integrity, exclusive commits and conditional deletion."""

import hashlib
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from regimex.infrastructure import object_store
from regimex.infrastructure.object_store import (
    LocalObjectStore,
    ObjectCapacityError,
    ObjectConflictError,
    ObjectInfo,
    ObjectIntegrityError,
    ObjectStore,
)


def checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def store(root: Path, *, object_limit: int = 1024, total_limit: int = 4096) -> LocalObjectStore:
    return LocalObjectStore(root, max_object_bytes=object_limit, max_total_bytes=total_limit)


def source(root: Path, name: str = "source.bin", data: bytes = b"payload") -> Path:
    path = root / name
    path.write_bytes(data)
    return path


def assert_no_temporary_files(root: Path) -> None:
    assert not tuple(root.rglob(".regimex-temporary-*"))


def test_immutable_put_idempotency_materialization_and_conditional_delete(tmp_path: Path) -> None:
    objects: ObjectStore = store(tmp_path / "objects")
    upload = source(tmp_path)
    info = objects.put("snapshots/raw.json", upload, expected_sha256=checksum(b"payload"))
    assert info.size_bytes == 7
    assert info.sha256 == checksum(b"payload")
    assert ObjectInfo.model_validate_json(info.model_dump_json()) == info
    assert objects.stat(info.key) == info
    assert objects.put(info.key, upload, expected_sha256=info.sha256) == info
    upload.write_bytes(b"changed-source")
    destination = tmp_path / "materialized" / "raw.json"
    assert (
        objects.materialize(
            info.key, destination, expected_sha256=info.sha256, expected_size_bytes=info.size_bytes
        )
        == info
    )
    assert destination.read_bytes() == b"payload"
    destination.write_bytes(b"changed-download")
    assert objects.stat(info.key) == info
    objects.delete(info)
    with pytest.raises(FileNotFoundError):
        objects.stat(info.key)
    assert_no_temporary_files(tmp_path)


def test_zero_byte_objects_are_valid_with_zero_capacity(tmp_path: Path) -> None:
    objects = store(tmp_path / "objects", object_limit=0, total_limit=0)
    upload = source(tmp_path, data=b"")
    info = objects.put("empty", upload, expected_sha256=checksum(b""))
    assert info.size_bytes == 0
    destination = tmp_path / "empty-copy"
    objects.materialize(info.key, destination, expected_sha256=info.sha256, expected_size_bytes=0)
    assert destination.read_bytes() == b""
    objects.delete(info)


def test_conflicting_upload_keeps_original_bytes_and_revision(tmp_path: Path) -> None:
    objects = store(tmp_path / "objects")
    original = source(tmp_path, data=b"original")
    replacement = source(tmp_path, "replacement.bin", b"replacement")
    info = objects.put("fixed", original, expected_sha256=checksum(b"original"))
    with pytest.raises(ObjectConflictError):
        objects.put("fixed", replacement, expected_sha256=checksum(b"replacement"))
    assert objects.stat("fixed") == info
    assert (objects.root / "fixed").read_bytes() == b"original"
    assert_no_temporary_files(tmp_path)


def test_idempotent_upload_still_validates_the_supplied_source(tmp_path: Path) -> None:
    objects = store(tmp_path / "objects")
    upload = source(tmp_path)
    info = objects.put("fixed", upload, expected_sha256=checksum(b"payload"))
    upload.write_bytes(b"incorrect")
    with pytest.raises(ObjectIntegrityError, match="upload source SHA-256 mismatch"):
        objects.put(info.key, upload, expected_sha256=info.sha256)
    assert objects.stat(info.key) == info
    assert_no_temporary_files(tmp_path)


@pytest.mark.parametrize("identical", [True, False])
def test_racing_commits_are_idempotent_or_conflict_without_overwrite(
    tmp_path: Path, identical: bool
) -> None:
    root = tmp_path / "objects"
    first = source(tmp_path, "first", b"first")
    second = source(tmp_path, "second", b"first" if identical else b"second")
    stores = (store(root), store(root))
    start = Barrier(2)

    def upload(index: int) -> ObjectInfo | ObjectConflictError:
        start.wait()
        path = (first, second)[index]
        try:
            return stores[index].put(
                "shared/key", path, expected_sha256=checksum(path.read_bytes())
            )
        except ObjectConflictError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(upload, (0, 1)))
    successes = tuple(result for result in results if isinstance(result, ObjectInfo))
    assert len(successes) == (2 if identical else 1)
    assert all(info == stores[0].stat("shared/key") for info in successes)
    if not identical:
        assert sum(isinstance(result, ObjectConflictError) for result in results) == 1
    assert_no_temporary_files(tmp_path)


def test_deletion_rejects_stale_revision_even_after_identical_recreation(tmp_path: Path) -> None:
    objects = store(tmp_path / "objects")
    upload = source(tmp_path)
    first = objects.put("recreated", upload, expected_sha256=checksum(b"payload"))
    objects.delete(first)
    second = objects.put(first.key, upload, expected_sha256=first.sha256)
    assert first.sha256 == second.sha256
    assert first.size_bytes == second.size_bytes
    assert first.version_id != second.version_id
    with pytest.raises(ObjectConflictError, match="stale object information"):
        objects.delete(first)
    assert objects.stat(second.key) == second


def test_deleted_object_is_not_a_placeholder_success(tmp_path: Path) -> None:
    objects = store(tmp_path / "objects")
    info = objects.put("once", source(tmp_path), expected_sha256=checksum(b"payload"))
    objects.delete(info)
    with pytest.raises(FileNotFoundError):
        objects.delete(info)


@pytest.mark.parametrize("operation", ["put", "materialize"])
def test_interrupted_copy_never_commits_or_leaves_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    objects = store(tmp_path / "objects")
    payload = b"abcdefgh"
    upload = source(tmp_path, data=payload)
    info = objects.put("existing", upload, expected_sha256=checksum(payload))
    real_read = os.read
    reads = 0

    def interrupted_read(descriptor: int, size: int) -> bytes:
        nonlocal reads
        reads += 1
        if reads == 2:
            raise OSError("simulated interrupted copy")
        return real_read(descriptor, size)

    monkeypatch.setattr(object_store, "_CHUNK_BYTES", 4)
    monkeypatch.setattr(os, "read", interrupted_read)
    with pytest.raises(OSError, match="interrupted copy"):
        if operation == "put":
            objects.put("uncommitted", upload, expected_sha256=checksum(payload))
        else:
            objects.materialize(
                info.key,
                tmp_path / "uncommitted",
                expected_sha256=info.sha256,
                expected_size_bytes=info.size_bytes,
            )
    assert not (objects.root / "uncommitted").exists()
    assert not (tmp_path / "uncommitted").exists()
    assert_no_temporary_files(tmp_path)


def test_source_mutation_cannot_publish_bytes_other_than_the_declared_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    objects = store(tmp_path / "objects")
    payload = b"original"
    upload = source(tmp_path, data=payload)
    real_read = os.read
    changed = False

    def changing_read(descriptor: int, size: int) -> bytes:
        nonlocal changed
        block = real_read(descriptor, size)
        if not changed:
            changed = True
            upload.write_bytes(b"replaced")
        return block

    monkeypatch.setattr(os, "read", changing_read)
    try:
        info = objects.put("snapshot", upload, expected_sha256=checksum(payload))
    except ObjectIntegrityError:
        assert not (objects.root / "snapshot").exists()
    else:
        # Capturing all original bytes before mutation is a valid immutable snapshot.
        assert info.sha256 == checksum(payload)
        assert (objects.root / info.key).read_bytes() == payload
    assert_no_temporary_files(tmp_path)


def test_source_mutation_that_mixes_chunks_fails_before_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    objects = store(tmp_path / "objects")
    upload = source(tmp_path, data=b"abcdefgh")
    real_read = os.read
    changed = False

    def changing_read(descriptor: int, size: int) -> bytes:
        nonlocal changed
        block = real_read(descriptor, size)
        if not changed:
            changed = True
            upload.write_bytes(b"xxxxxxxx")
        return block

    monkeypatch.setattr(object_store, "_CHUNK_BYTES", 4)
    monkeypatch.setattr(os, "read", changing_read)
    with pytest.raises(ObjectIntegrityError):
        objects.put("mixed", upload, expected_sha256=checksum(b"abcdefgh"))
    assert not (objects.root / "mixed").exists()
    assert_no_temporary_files(tmp_path)


def test_missing_immutable_version_prevents_materialization(tmp_path: Path) -> None:
    objects = store(tmp_path / "objects")
    info = objects.put("test", source(tmp_path), expected_sha256=checksum(b"payload"))
    path = objects.root / info.key
    path.chmod(0o600)
    os.removexattr(path, "user.regimex.object-version")
    destination = tmp_path / "uncommitted"
    with pytest.raises(ObjectIntegrityError, match="immutable version"):
        objects.materialize(
            info.key, destination, expected_sha256=info.sha256, expected_size_bytes=info.size_bytes
        )
    assert not destination.exists()
    assert_no_temporary_files(tmp_path)


def test_unsupported_extended_attributes_fail_without_committing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    objects = store(tmp_path / "objects")
    upload = source(tmp_path)

    def unsupported(descriptor: int, name: str, value: bytes) -> None:
        raise OSError("extended attributes unsupported")

    monkeypatch.setattr(os, "setxattr", unsupported)
    with pytest.raises(OSError, match="attributes unsupported"):
        objects.put("uncommitted", upload, expected_sha256=checksum(b"payload"))
    assert not (objects.root / "uncommitted").exists()
    assert_no_temporary_files(tmp_path)


@pytest.mark.parametrize("wrong", ["digest", "size", "stored-bytes"])
def test_download_rejects_mismatched_identity_before_exposing_destination(
    tmp_path: Path, wrong: str
) -> None:
    objects = store(tmp_path / "objects")
    info = objects.put("verified", source(tmp_path), expected_sha256=checksum(b"payload"))
    if wrong == "stored-bytes":
        path = objects.root / info.key
        path.chmod(0o600)
        path.write_bytes(b"corrupt")
    destination = tmp_path / "downloaded"
    with pytest.raises(ObjectIntegrityError, match="expected size or SHA-256"):
        objects.materialize(
            info.key,
            destination,
            expected_sha256="a" * 64 if wrong == "digest" else info.sha256,
            expected_size_bytes=info.size_bytes + 1 if wrong == "size" else info.size_bytes,
        )
    assert not destination.exists()
    assert_no_temporary_files(tmp_path)
    if wrong == "stored-bytes":
        with pytest.raises(ObjectConflictError):
            objects.delete(info)


def test_upload_digest_mismatch_never_commits(tmp_path: Path) -> None:
    objects = store(tmp_path / "objects")
    with pytest.raises(ObjectIntegrityError):
        objects.put("uncommitted", source(tmp_path), expected_sha256="a" * 64)
    assert not (objects.root / "uncommitted").exists()
    assert_no_temporary_files(tmp_path)


@pytest.mark.parametrize(
    "key",
    [
        "",
        "/absolute",
        "a/../b",
        "a//b",
        "./a",
        "a/",
        "a\\b",
        "a:b",
        "a\x00b",
        "a\nb",
        "a\x01b",
        "a\x7fb",
        ".regimex-lock",
        "a/.regimex-temporary-x",
    ],
)
def test_unsafe_keys_fail_before_touching_objects(tmp_path: Path, key: str) -> None:
    objects = store(tmp_path / "objects")
    with pytest.raises(ValueError):
        objects.put(key, source(tmp_path), expected_sha256=checksum(b"payload"))
    with pytest.raises(ValueError):
        objects.stat(key)


def test_nested_store_symlink_cannot_write_or_read_outside_root(tmp_path: Path) -> None:
    objects = store(tmp_path / "objects")
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = source(outside, "file", b"unchanged")
    (objects.root / "nested").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        objects.put("nested/file", source(tmp_path), expected_sha256=checksum(b"payload"))
    with pytest.raises(OSError):
        objects.stat("nested/file")
    assert victim.read_bytes() == b"unchanged"
    assert_no_temporary_files(tmp_path)


def test_symlink_object_and_symlink_lock_are_rejected(tmp_path: Path) -> None:
    objects = store(tmp_path / "objects")
    outside = source(tmp_path)
    (objects.root / "linked").symlink_to(outside)
    with pytest.raises(OSError):
        objects.stat("linked")
    with pytest.raises(OSError):
        objects.put("linked", outside, expected_sha256=checksum(b"payload"))
    lock = objects.root / ".regimex-lock"
    lock.unlink()
    lock.symlink_to(outside)
    with pytest.raises(OSError):
        objects.stat("linked")
    assert outside.read_bytes() == b"payload"


@pytest.mark.parametrize("location", ["root", "source", "source-parent", "destination-parent"])
def test_symlinks_in_local_path_components_are_rejected(tmp_path: Path, location: str) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    if location == "root":
        with pytest.raises(OSError):
            store(linked / "objects")
        return
    objects = store(tmp_path / "objects")
    upload = source(actual)
    if location == "source":
        link = tmp_path / "source-link"
        link.symlink_to(upload)
        with pytest.raises(OSError):
            objects.put("test", link, expected_sha256=checksum(b"payload"))
    elif location == "source-parent":
        with pytest.raises(OSError):
            objects.put("test", linked / upload.name, expected_sha256=checksum(b"payload"))
    else:
        info = objects.put("test", upload, expected_sha256=checksum(b"payload"))
        with pytest.raises(OSError):
            objects.materialize(
                info.key,
                linked / "download",
                expected_sha256=info.sha256,
                expected_size_bytes=info.size_bytes,
            )
        assert not (actual / "download").exists()
    assert_no_temporary_files(tmp_path)


def test_special_sources_are_rejected_without_blocking(tmp_path: Path) -> None:
    objects = store(tmp_path / "objects")
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular files"):
        objects.put("fifo", fifo, expected_sha256=checksum(b""))
    assert_no_temporary_files(tmp_path)


def test_materialization_destination_is_exclusive_and_must_be_outside_store(tmp_path: Path) -> None:
    objects = store(tmp_path / "objects")
    info = objects.put("test", source(tmp_path), expected_sha256=checksum(b"payload"))
    existing = source(tmp_path, "existing", b"do-not-overwrite")
    for destination in (existing, objects.root / "new"):
        with pytest.raises(ValueError):
            objects.materialize(
                info.key,
                destination,
                expected_sha256=info.sha256,
                expected_size_bytes=info.size_bytes,
            )
    linked = tmp_path / "existing-link"
    linked.symlink_to(existing)
    with pytest.raises(ObjectConflictError):
        objects.materialize(
            info.key, linked, expected_sha256=info.sha256, expected_size_bytes=info.size_bytes
        )
    assert existing.read_bytes() == b"do-not-overwrite"
    assert not (objects.root / "new").exists()
    assert_no_temporary_files(tmp_path)


def test_object_and_total_capacity_limits_preserve_prior_commits(tmp_path: Path) -> None:
    objects = store(tmp_path / "objects", object_limit=4, total_limit=6)
    upload = source(tmp_path, data=b"four")
    info = objects.put("first", upload, expected_sha256=checksum(b"four"))
    assert objects.put(info.key, upload, expected_sha256=info.sha256) == info
    with pytest.raises(ObjectCapacityError, match="committed-payload capacity"):
        objects.put("second", upload, expected_sha256=info.sha256)
    upload.write_bytes(b"five!")
    with pytest.raises(ObjectCapacityError, match="per-object byte limit"):
        objects.put("large", upload, expected_sha256=checksum(b"five!"))
    assert objects.stat(info.key) == info
    assert not (objects.root / "second").exists()
    assert not (objects.root / "large").exists()
    assert_no_temporary_files(tmp_path)


def test_racing_uploads_cannot_exceed_total_capacity(tmp_path: Path) -> None:
    root = tmp_path / "objects"
    stores = tuple(store(root, object_limit=4, total_limit=4) for _ in range(2))
    upload = source(tmp_path, data=b"four")
    start = Barrier(2)

    def put(index: int) -> ObjectInfo | ObjectCapacityError:
        start.wait()
        try:
            return stores[index].put(str(index), upload, expected_sha256=checksum(b"four"))
        except ObjectCapacityError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(put, (0, 1)))
    assert sum(isinstance(result, ObjectInfo) for result in results) == 1
    assert sum(isinstance(result, ObjectCapacityError) for result in results) == 1
    assert_no_temporary_files(tmp_path)


def test_commits_sync_payload_and_containing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    objects = store(tmp_path / "objects")
    synced: list[tuple[int, int]] = []
    real_fsync = os.fsync

    def record_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        synced.append((metadata.st_mode, metadata.st_size))
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", record_fsync)
    objects.put("nested/test", source(tmp_path), expected_sha256=checksum(b"payload"))
    assert any(stat.S_ISREG(mode) and size == 7 for mode, size in synced)
    assert any(stat.S_ISDIR(mode) for mode, _ in synced)


@pytest.mark.parametrize("object_limit,total_limit", [(-1, 4), (1, -1), (True, 4), (5, 4)])
def test_invalid_capacity_configuration_fails(
    tmp_path: Path, object_limit: int, total_limit: int
) -> None:
    with pytest.raises(ValueError):
        store(tmp_path / "objects", object_limit=object_limit, total_limit=total_limit)
