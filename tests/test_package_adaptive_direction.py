"""Check the catalog that identifies the trained and evaluated adaptive forest."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_catalog_preserves_actual_fit_and_full_selection_memberships(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from test_data_catalog import snapshot

    from regimex.data_catalog import Partition, create_snapshot_member, seal_dataset

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    packaging = import_module("package_adaptive_direction")
    original_order: tuple[Partition, ...] = ("test", "validation", "train")

    def corpus(start: int) -> SimpleNamespace:
        members = tuple(
            create_snapshot_member(snapshot(i), partition)
            for i, partition in enumerate(original_order, start)
        )
        return SimpleNamespace(dataset=seal_dataset(f"dataset-{start}", members))

    train, selection, final = (corpus(i) for i in (0, 3, 6))
    merged = packaging.catalog(train, selection, final)
    assert [m.split_group.seed for m in merged.members] == [2, 5, 4, 3, 8, 7, 6]
    assert [m.partition for m in merged.members] == ["train"] + ["validation"] * 3 + ["test"] * 3
    source = tmp_path / "mismatched.py"
    source.write_text("raise AssertionError('unverified source ran')")
    with pytest.raises(ValueError, match="fitted source"):
        packaging.frozen_implementation(source, "0" * 64)
