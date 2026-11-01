"""Actual provider invariants and lightweight S3 deployment validation."""

import fcntl
import multiprocessing
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import BinaryIO, cast

import pytest
from test_object_store_contract import (
    RawClient,
    cleanup_owned_test_prefix,
    digest,
    s3_configuration,
    source,
)

from regimex.infrastructure.object_store import (
    ObjectCapacityError,
    ObjectConflictError,
    ObjectInfo,
    ObjectIntegrityError,
)
from regimex.infrastructure.s3_config import S3Config
from regimex.infrastructure.s3_object_store import S3ObjectStore, _Client, _ResponseBody


@dataclass(frozen=True)
class Context:
    objects: S3ObjectStore
    client: RawClient

    @property
    def bucket(self) -> str:
        return self.objects.config.bucket

    def key(self, relative: str) -> str:
        return self.objects.config.prefix + "/" + relative


@pytest.fixture
def actual_s3(tmp_path: Path) -> Iterator[Context]:
    config = s3_configuration(tmp_path / "coordination")
    objects = S3ObjectStore(config, max_object_bytes=1024, max_total_bytes=8192)
    objects.check()
    yield Context(objects, cast(RawClient, objects._sdk().client))
    cleanup_owned_test_prefix(objects)


def plain_config(root: Path, **values: object) -> S3Config:
    configuration: dict[str, object] = {
        "endpoint_url": "https://storage.example.invalid",
        "region": "us-east-1",
        "bucket": "private-bucket",
        "prefix": "regimex",
        "lock_root": root,
    }
    configuration.update(values)
    return S3Config.model_validate(configuration)


@pytest.mark.parametrize(
    "endpoint",
    (
        "http://192.0.2.1:8333",
        "http://localhost:8333",
        "https://user:secret@example.invalid",
        "https://example.invalid/path",
        "https://example.invalid?token=secret",
        "https://example.invalid#fragment",
        "file:///tmp/storage",
        "https://example.invalid:0",
    ),
)
def test_unsafe_or_inert_endpoint_settings_rejected(tmp_path: Path, endpoint: str) -> None:
    with pytest.raises(ValueError):
        plain_config(tmp_path, endpoint_url=endpoint, allow_loopback_http=True)


def test_loopback_http_requires_explicit_option(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plain_config(tmp_path, endpoint_url="http://127.0.0.1:8333")
    config = plain_config(tmp_path, endpoint_url="http://127.0.0.1:8333", allow_loopback_http=True)
    assert config.endpoint_origin == "http://127.0.0.1:8333"
    with pytest.raises(ValueError):
        plain_config(
            tmp_path,
            endpoint_url=config.endpoint_url,
            allow_loopback_http=True,
            ca_bundle=tmp_path / "ca",
        )


@pytest.mark.parametrize(
    "endpoint",
    (
        "https://operator:private-value@example.invalid",
        "https://operator:private-value",
        "https://[private-value]",
        "https://example.invalid?token=private-value",
    ),
)
def test_configuration_errors_hide_secret_bearing_endpoint(tmp_path: Path, endpoint: str) -> None:
    with pytest.raises(ValueError) as caught:
        plain_config(tmp_path, endpoint_url=endpoint)
    assert "private-value" not in str(caught.value)
    assert "private-value" not in repr(caught.value)
    assert endpoint not in str(caught.value)


def test_scope_and_coordination_binding_are_nonsecret_and_stable(tmp_path: Path) -> None:
    first = S3ObjectStore(
        plain_config(tmp_path / "one", endpoint_url="https://STORAGE.example.invalid:443/"),
        max_object_bytes=4,
        max_total_bytes=8,
    )
    second = S3ObjectStore(plain_config(tmp_path / "one"), max_object_bytes=4, max_total_bytes=8)
    assert first.scope_id == second.scope_id
    other_root = S3ObjectStore(
        plain_config(tmp_path / "two"), max_object_bytes=4, max_total_bytes=8
    )
    assert other_root.scope_id != first.scope_id
    different = S3ObjectStore(
        plain_config(tmp_path / "three", prefix="other"), max_object_bytes=4, max_total_bytes=8
    )
    assert different.scope_id != first.scope_id
    with pytest.raises(ValueError, match="different storage settings"):
        S3ObjectStore(plain_config(tmp_path / "one"), max_object_bytes=4, max_total_bytes=9)


def test_replacing_coordination_root_changes_scope_and_invalidates_old_instance(
    tmp_path: Path,
) -> None:
    config = plain_config(tmp_path / "coordination")
    previous = S3ObjectStore(config, max_object_bytes=4, max_total_bytes=8)
    previous.root.rename(tmp_path / "previous-root")
    replacement = S3ObjectStore(config, max_object_bytes=4, max_total_bytes=8)
    assert previous.scope_id != replacement.scope_id
    with pytest.raises(ObjectIntegrityError, match="directory identity changed"):
        previous.check()


def test_coordination_lock_wait_is_bounded_without_network(tmp_path: Path) -> None:
    objects = S3ObjectStore(
        plain_config(tmp_path / "coordination"), max_object_bytes=4, max_total_bytes=8
    )
    with (objects.root / ".regimex-lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="coordination lock wait"):
            objects.stat("never-read")
        assert 4.5 <= time.monotonic() - started < 10


def test_constructor_is_offline_and_imports_no_sdk_or_science(tmp_path: Path) -> None:
    program = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "from regimex.infrastructure.s3_config import S3Config",
            "from regimex.infrastructure.s3_object_store import S3ObjectStore",
            "config = S3Config(",
            "    endpoint_url='https://unresolvable.invalid', region='us-east-1',",
            "    bucket='private-bucket', prefix='regimex',",
            f"    lock_root=Path({str(tmp_path / 'offline')!r}))",
            "S3ObjectStore(config, max_object_bytes=4, max_total_bytes=8)",
            "assert not any(name in sys.modules for name in",
            "               ('boto3', 'botocore', 'numpy', 'sklearn', 'torch'))",
        )
    )
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)


def test_stat_streams_actual_bytes_instead_of_metadata(actual_s3: Context) -> None:
    actual_s3.client.put_object(
        Bucket=actual_s3.bucket,
        Key=actual_s3.key("forged"),
        Body=b"actual",
        Metadata={"sha256": "0" * 64},
    )
    info = actual_s3.objects.stat("forged")
    assert info.sha256 == digest(b"actual") and info.size_bytes == 6


def test_restore_empty_target_check_includes_versions_and_multipart(actual_s3: Context) -> None:
    objects = actual_s3.objects
    objects.assert_empty_store()
    actual_s3.client.put_object(Bucket=actual_s3.bucket, Key=actual_s3.key("prior"), Body=b"prior")
    actual_s3.client.delete_object(Bucket=actual_s3.bucket, Key=actual_s3.key("prior"))
    with pytest.raises(ObjectConflictError, match="empty"):
        objects.assert_empty_store()
    cleanup_owned_test_prefix(objects)
    objects.assert_empty_store()
    upload = actual_s3.client.create_multipart_upload(
        Bucket=actual_s3.bucket, Key=actual_s3.key("pending")
    )
    with pytest.raises(ObjectConflictError, match="empty"):
        objects.assert_empty_store()
    actual_s3.client.abort_multipart_upload(
        Bucket=actual_s3.bucket,
        Key=actual_s3.key("pending"),
        UploadId=upload["UploadId"],
    )
    objects.assert_empty_store()


@pytest.mark.parametrize(
    "name", (".regimex-unknown", ".regimex-temporary-" + "1" * 32, "unexpected")
)
def test_restore_empty_target_check_rejects_local_coordination_leftovers(
    actual_s3: Context, name: str
) -> None:
    (actual_s3.objects.root / name).write_bytes(b"not an empty restore target")
    with pytest.raises(ObjectConflictError, match="empty S3 coordination"):
        actual_s3.objects.assert_empty_store()


def test_bucket_versioning_suspension_fails_closed(actual_s3: Context, tmp_path: Path) -> None:
    actual_s3.client.put_bucket_versioning(
        Bucket=actual_s3.bucket, VersioningConfiguration={"Status": "Suspended"}
    )
    try:
        with pytest.raises(ObjectIntegrityError, match="versioning enabled"):
            actual_s3.objects.put(
                "uncommitted", source(tmp_path), expected_sha256=digest(b"payload")
            )
    finally:
        actual_s3.client.put_bucket_versioning(
            Bucket=actual_s3.bucket, VersioningConfiguration={"Status": "Enabled"}
        )
    assert actual_s3.objects.inventory() == ()


def test_retained_old_versions_consume_quota(tmp_path: Path) -> None:
    config = s3_configuration(tmp_path / "coordination")
    objects = S3ObjectStore(config, max_object_bytes=4, max_total_bytes=8)
    objects.check()
    client = objects._sdk().client
    try:
        original = objects.put("same", source(tmp_path, b"four"), expected_sha256=digest(b"four"))
        newer = client.put_object(Bucket=config.bucket, Key=config.prefix + "/same", Body=b"four")
        assert newer["VersionId"] != original.version_id
        with pytest.raises(ObjectCapacityError):
            objects.put("next", source(tmp_path, b"four"), expected_sha256=digest(b"four"))
        assert sum(item.size_bytes for item in objects.inventory()) == 8
        assert objects.stat_version(original.key, original.version_id) == original
        with pytest.raises(ObjectConflictError):
            objects.delete(original)
        objects.delete_version(original)
        accepted = objects.put("next", source(tmp_path, b"four"), expected_sha256=digest(b"four"))
        assert accepted.size_bytes == 4
    finally:
        cleanup_owned_test_prefix(objects)


def test_newer_generation_interleaved_with_exact_delete_survives(
    actual_s3: Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    objects, client = actual_s3.objects, actual_s3.client
    info = objects.put("same", source(tmp_path), expected_sha256=digest(b"payload"))
    original_delete = client.delete_object
    created: list[str] = []

    def interleave(**values: object) -> dict[str, object]:
        response = client.put_object(
            Bucket=actual_s3.bucket, Key=actual_s3.key("same"), Body=b"payload"
        )
        version = response["VersionId"]
        assert isinstance(version, str)
        created.append(version)
        return original_delete(**values)

    with monkeypatch.context() as patch:
        patch.setattr(client, "delete_object", interleave)
        objects.delete(info)
    assert len(created) == 1
    assert objects.stat(info.key).version_id == created[0]
    assert objects.stat(info.key).sha256 == info.sha256


@pytest.mark.parametrize("after_commit", (False, True))
def test_ambiguous_upload_reconciles_without_new_version(
    actual_s3: Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after_commit: bool
) -> None:
    client = actual_s3.client
    original_put = client.put_object
    calls = 0

    def interrupted(**values: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            if after_commit:
                original_put(**values)
            raise TimeoutError("private transport details")
        return original_put(**values)

    with monkeypatch.context() as patch:
        patch.setattr(client, "put_object", interrupted)
        info = actual_s3.objects.put(
            "recovered", source(tmp_path), expected_sha256=digest(b"payload")
        )
    versions = actual_s3.objects.inventory()
    assert len(versions) == 1 and versions[0].version_id == info.version_id
    assert calls == (1 if after_commit else 2)


@pytest.mark.parametrize("phase", ("read", "close"))
def test_stream_sdk_failure_is_redacted_and_partial_destination_removed(
    actual_s3: Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    objects, client = actual_s3.objects, actual_s3.client
    info = objects.put("verified", source(tmp_path), expected_sha256=digest(b"payload"))
    original_get = client.get_object
    error = cast(
        Exception,
        import_module("botocore.exceptions").ReadTimeoutError(
            endpoint_url="https://operator:private-value@example.invalid"
        ),
    )
    closed: list[bool] = []

    class InterruptedBody:
        def __init__(self, body: _ResponseBody) -> None:
            self.body = body
            self.reads = 0

        def read(self, amount: int) -> bytes:
            self.reads += 1
            if phase == "read" and self.reads > 1:
                raise error
            return self.body.read(min(amount, 2) if phase == "read" else amount)

        def close(self) -> None:
            self.body.close()
            closed.append(True)
            if phase == "close":
                raise error

    def interrupted_get(**values: object) -> dict[str, object]:
        response = original_get(**values)
        response["Body"] = InterruptedBody(cast(_ResponseBody, response["Body"]))
        return response

    destination = tmp_path / "partial-download"
    with monkeypatch.context() as patch:
        patch.setattr(client, "get_object", interrupted_get)
        with pytest.raises(OSError, match="interrupted") as caught:
            objects.materialize(
                info.key,
                destination,
                expected_sha256=info.sha256,
                expected_size_bytes=info.size_bytes,
            )
    assert "private-value" not in str(caught.value)
    assert not destination.exists()
    assert not tuple(tmp_path.rglob(".regimex-temporary-*"))
    assert closed == [True]


def test_multipart_cleanup_respects_scope_cutoff_and_identity(actual_s3: Context) -> None:
    client, objects = actual_s3.client, actual_s3.objects
    response = client.create_multipart_upload(
        Bucket=actual_s3.bucket, Key=actual_s3.key("abandoned")
    )
    upload_id = response["UploadId"]
    client.upload_part(
        Bucket=actual_s3.bucket,
        Key=actual_s3.key("abandoned"),
        UploadId=upload_id,
        PartNumber=1,
        Body=b"partial",
    )
    uploads = objects.list_multipart_uploads()
    assert len(uploads) == 1 and uploads[0].key == "abandoned"
    expected = uploads[0]
    with pytest.raises(ObjectConflictError, match="newer"):
        objects.abort_multipart(expected, initiated_before=expected.initiated_at)
    assert objects.list_multipart_uploads() == uploads
    with pytest.raises(ObjectConflictError, match="stale"):
        objects.abort_multipart(
            expected.model_copy(
                update={"initiated_at": expected.initiated_at - timedelta(seconds=1)}
            ),
            initiated_before=datetime.now(UTC),
        )
    assert objects.list_multipart_uploads() == uploads
    objects.abort_multipart(expected, initiated_before=datetime.now(UTC))
    assert objects.list_multipart_uploads() == ()
    assert objects.inventory() == ()


def crash_spooling_process(config_json: str, upload: str, ready: str) -> None:
    class PausedStore(S3ObjectStore):
        def _put_stable(
            self, client: _Client, key: str, source: BinaryIO, size: int, checksum: str
        ) -> ObjectInfo:
            Path(ready).touch()
            while True:
                time.sleep(1)

    objects = PausedStore(
        S3Config.model_validate_json(config_json), max_object_bytes=1024, max_total_bytes=8192
    )
    objects.put("never-published", Path(upload), expected_sha256=digest(b"payload"))


def test_sigkill_spooling_leaves_only_pinned_local_upload_for_grace_cleanup(
    actual_s3: Context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    objects = actual_s3.objects
    upload = source(tmp_path)
    committed = objects.put("keep", upload, expected_sha256=digest(b"payload"))
    ready = tmp_path / "child-spooling"
    process = multiprocessing.get_context("fork").Process(
        target=crash_spooling_process,
        args=(objects.config.model_dump_json(), str(upload), str(ready)),
    )
    try:
        process.start()
        deadline = time.monotonic() + 20
        while not ready.exists():
            assert process.is_alive() and time.monotonic() < deadline
            time.sleep(0.01)
        process.kill()
        process.join(10)
        assert process.exitcode == -9
    finally:
        if process.is_alive():
            process.kill()
            process.join(10)
    uploads = objects.list_multipart_uploads()
    assert len(uploads) == 1
    expected = uploads[0]
    assert expected.upload_id.startswith("local-temporary-v1:")
    assert expected.key.startswith(".regimex-temporary-")
    assert (objects.root / expected.key).read_bytes() == b"payload"
    with pytest.raises(FileNotFoundError):
        objects.stat("never-published")
    with pytest.raises(ObjectConflictError, match="newer"):
        objects.abort_multipart(expected, initiated_before=expected.initiated_at)
    with pytest.raises(ObjectConflictError, match="empty S3 coordination"):
        objects.assert_empty_store()
    time.sleep(1.1)
    cutoff = datetime.now(UTC) - timedelta(seconds=1)

    def no_remote_call() -> _Client:
        raise AssertionError("local spool cleanup must not call the remote provider")

    with monkeypatch.context() as patch:
        patch.setattr(objects, "_ready", no_remote_call)
        objects.abort_multipart(expected, initiated_before=cutoff)
    assert objects.list_multipart_uploads() == ()
    assert objects.stat(committed.key) == committed
    assert {item.key for item in objects.inventory()} == {committed.key}


def quota_process(config_json: str, upload: str, result_file: str, start_file: str) -> None:
    objects = S3ObjectStore(
        S3Config.model_validate_json(config_json), max_object_bytes=4, max_total_bytes=4
    )
    Path(result_file + ".ready").touch()
    deadline = time.monotonic() + 20
    while not Path(start_file).exists():
        if time.monotonic() > deadline:
            raise AssertionError("parent did not release concurrent quota test")
        time.sleep(0.01)
    try:
        objects.put(Path(result_file).name, Path(upload), expected_sha256=digest(b"four"))
    except ObjectCapacityError:
        outcome = "capacity"
    else:
        outcome = "accepted"
    Path(result_file).write_text(outcome)


def test_processes_share_one_retained_byte_quota(tmp_path: Path) -> None:
    config = s3_configuration(tmp_path / "coordination")
    objects = S3ObjectStore(config, max_object_bytes=4, max_total_bytes=4)
    objects.check()
    upload = source(tmp_path, b"four")
    start = tmp_path / "start"
    outputs = (tmp_path / "one", tmp_path / "two")
    # The supported coordination backend is POSIX; children receive no SDK object.
    context = multiprocessing.get_context("fork")
    processes = tuple(
        context.Process(
            target=quota_process,
            args=(config.model_dump_json(), str(upload), str(output), str(start)),
        )
        for output in outputs
    )
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 20
        while not all(Path(str(output) + ".ready").exists() for output in outputs):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        start.touch()
        for process in processes:
            process.join(30)
            assert process.exitcode == 0
        assert sorted(output.read_text() for output in outputs) == ["accepted", "capacity"]
        assert sum(item.size_bytes for item in objects.inventory()) == 4
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(10)
        cleanup_owned_test_prefix(objects)


def test_inventory_pagination_and_prefix_isolation(actual_s3: Context) -> None:
    client = actual_s3.client
    for index in range(1002):
        client.put_object(Bucket=actual_s3.bucket, Key=actual_s3.key(f"many/{index:04d}"), Body=b"")
    inventory = actual_s3.objects.inventory()
    assert len(inventory) == 1002
    assert {item.key for item in inventory} == {f"many/{index:04d}" for index in range(1002)}
    assert all(item.size_bytes == 0 and not item.is_delete_marker for item in inventory)
    assert all(item.version_id != "null" for item in inventory)
