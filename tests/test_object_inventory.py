"""Local maintenance inventory preserves version and empty-target safeguards."""

import hashlib
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from regimex.infrastructure.object_store import LocalObjectStore, ObjectConflictError


def test_local_inventory_tracks_exact_generations_and_has_stable_location_scope(
    tmp_path: Path,
) -> None:
    store = LocalObjectStore(tmp_path / "objects", max_object_bytes=1024, max_total_bytes=4096)
    store.assert_empty_store()
    source = tmp_path / "source"
    source.write_bytes(b"payload")
    before = datetime.now(UTC)
    info = store.put(
        "nested/object", source, expected_sha256=hashlib.sha256(b"payload").hexdigest()
    )
    inventory = store.inventory()
    assert len(inventory) == 1
    assert (inventory[0].key, inventory[0].version_id, inventory[0].size_bytes) == (
        info.key,
        info.version_id,
        info.size_bytes,
    )
    assert before - timedelta(seconds=1) <= inventory[0].last_modified <= datetime.now(UTC)
    assert store.stat_version(info.key, info.version_id) == info
    assert store.list_multipart_uploads() == ()
    again = LocalObjectStore(tmp_path / "objects", max_object_bytes=2048, max_total_bytes=8192)
    assert again.scope_id == store.scope_id
    other = LocalObjectStore(tmp_path / "other", max_object_bytes=1024, max_total_bytes=4096)
    assert other.scope_id != store.scope_id
    with pytest.raises(ObjectConflictError):
        store.assert_empty_store()
    store.delete_version(info)
    new = store.put(info.key, source, expected_sha256=info.sha256)
    assert new.version_id != info.version_id
    with pytest.raises(ObjectConflictError):
        store.stat_version(info.key, info.version_id)
    with pytest.raises(ObjectConflictError):
        store.delete_version(info)
    assert store.stat(info.key) == new


def test_unverified_private_upload_remnants_are_preserved_and_block_restore(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path / "objects", max_object_bytes=1024, max_total_bytes=4096)
    remnant = store.root / ".regimex-temporary-unverified"
    remnant.write_bytes(b"partial")
    assert store.inventory() == ()
    with pytest.raises(ValueError, match="exact reserved"):
        store.list_multipart_uploads()
    with pytest.raises(ObjectConflictError, match="remnants"):
        store.assert_empty_store()
    assert remnant.read_bytes() == b"partial"


def test_inventory_and_restore_refuse_symbolic_links(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path / "objects", max_object_bytes=1024, max_total_bytes=4096)
    outside = tmp_path / "outside"
    outside.write_bytes(b"untouched")
    (store.root / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic links"):
        store.inventory()
    with pytest.raises(ValueError, match="symbolic links"):
        store.assert_empty_store()
    assert outside.read_bytes() == b"untouched"


def test_local_incomplete_cleanup_rejects_changed_identity_and_authoritative_alias(
    tmp_path: Path,
) -> None:
    store = LocalObjectStore(tmp_path / "objects", max_object_bytes=1024, max_total_bytes=4096)
    temporary = store.root / (".regimex-temporary-" + uuid4().hex)
    temporary.write_bytes(b"abandoned staging")
    expected = store.list_multipart_uploads()[0]
    temporary.write_bytes(b"changed staging")
    with pytest.raises(ObjectConflictError, match="stale or linked"):
        store.abort_multipart(expected, initiated_before=datetime.now(UTC))
    assert temporary.read_bytes() == b"changed staging"
    source = tmp_path / "source"
    source.write_bytes(b"authoritative")
    info = store.put(
        "authoritative", source, expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest()
    )
    linked = store.root / (".regimex-temporary-" + uuid4().hex)
    os.link(store.root / info.key, linked)
    current = store.stat(info.key)
    linked_upload = next(item for item in store.list_multipart_uploads() if item.key == linked.name)
    with pytest.raises(ObjectConflictError, match="stale or linked"):
        store.abort_multipart(linked_upload, initiated_before=datetime.now(UTC))
    assert store.stat(info.key) == current
    assert linked.read_bytes() == source.read_bytes()


def test_local_coordination_has_a_bounded_wait_for_another_writer(tmp_path: Path) -> None:
    first = LocalObjectStore(tmp_path / "objects", max_object_bytes=1024, max_total_bytes=4096)
    second = LocalObjectStore(tmp_path / "objects", max_object_bytes=1024, max_total_bytes=4096)
    with first._locked_root():
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="coordination timed out"):
            second.inventory()
        assert 4.5 <= time.monotonic() - started < 10.0
