"""Short-interval sampling must preserve the experiment mix and the held-out split."""

import importlib
from pathlib import Path
from types import ModuleType

import pytest
from test_data_catalog import snapshot

from regimex.data_catalog import SnapshotMember, create_snapshot_member


@pytest.fixture
def sampler(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("plateau_sampling")


def test_short_interval_sampling_replaces_only_negatives_in_the_same_stratum(
    sampler: ModuleType,
) -> None:
    members = tuple(
        create_snapshot_member(snapshot(i, instrument_count=2 if i >= 7 else 1), "train")
        for i in range(1, 10)
    )
    uniform = (*members[:4], *members[6:8])
    counts = {i: int(i in {2, 5, 6}) for i in range(1, 10)}
    selected, strata = sampler._swap_within_strata(uniform, members, counts)
    assert [member.split_group.seed for member in selected] == [5, 2, 3, 4, 7, 8]
    assert [(s["swaps"], s["positive_after"]) for s in strata] == [(1, 2), (0, 0)]
    assert sampler._swap_within_strata(uniform, tuple(reversed(members)), counts) == (
        selected,
        strata,
    )


@pytest.mark.parametrize("invalid", ["validation", "test", "duplicate", "foreign"])
def test_enrichment_rejects_invalid_members_before_reading_observations(
    invalid: str, sampler: ModuleType
) -> None:
    member = create_snapshot_member(snapshot(1), "train")
    pool = [member]
    uniform = [member]
    if invalid in {"validation", "test"}:
        pool.append(member.model_copy(update={"partition": invalid}))
    elif invalid == "duplicate":
        pool.append(member)
    else:
        uniform = [create_snapshot_member(snapshot(2), "train")]

    def unexpected_read(member: SnapshotMember) -> None:
        pytest.fail(f"Invalid cohort accessed snapshot {member.snapshot_id}")

    with pytest.raises(ValueError, match=r"[Tt]raining"):
        sampler.enrich_short_sideways(uniform, pool, unexpected_read)


def test_enrichment_verifies_the_snapshot_before_relabeling(sampler: ModuleType) -> None:
    member = create_snapshot_member(snapshot(1), "train")
    with pytest.raises(ValueError, match="registered identity"):
        sampler.enrich_short_sideways([member], [member], lambda _: snapshot(2))
