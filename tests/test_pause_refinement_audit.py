"""The native pause audit must never consult validation or confirmation snapshots."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_data_catalog import snapshot

from regimex.data_catalog import SnapshotMember, SourceSnapshot, create_snapshot_member


def test_audit_reads_only_training_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    audit = importlib.import_module("audit_pause_refinement")
    monkeypatch.setattr(audit, "annotate_pauses", audit.annotate)
    source = snapshot(1)
    members = (
        create_snapshot_member(snapshot(2), "validation"),
        create_snapshot_member(source, "train"),
        create_snapshot_member(snapshot(3), "test"),
    )
    reads: list[SnapshotMember] = []

    def read(member: SnapshotMember) -> SourceSnapshot:
        assert member.partition == "train"
        reads.append(member)
        return source

    corpus = SimpleNamespace(
        dataset=SimpleNamespace(members=members, content_sha256="fixture"), snapshot=read
    )
    summary = audit.audit(corpus)
    assert reads == [members[1]]
    assert summary["experiments"] == 1
    assert summary["sequences"] == 1
    assert sum(summary["steps_by_status"].values()) == source.export.time_grid.step_count
    assert summary["inserted_short"] == summary["changed_sequences"] == 0
