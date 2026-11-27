"""Protect the selection and qualification rules of the adaptive interval comparison."""

import importlib
from pathlib import Path
from types import ModuleType

import pytest

from regimex.storage import write_json


@pytest.fixture
def trial(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("compare_adaptive_intervals")


def test_control_wins_ties_and_preserves_unopened_holdout(
    trial: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = {name: {"interval_macro_f1": 0.75} for name in trial.CANDIDATES}
    assert trial.choose(selection) == trial.CONTROL
    monkeypatch.setattr(trial, "read_freeze", lambda _: {"winner": trial.CONTROL})
    with pytest.raises(ValueError, match="remains reserved"):
        trial.score(tmp_path, tmp_path)
    assert not (tmp_path / "opened.json").exists()


def test_only_winner_is_requested_on_the_reserved_final(
    trial: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_json(tmp_path / "freeze.json", {"winner": "pieces"})
    monkeypatch.setattr(trial, "read_freeze", lambda _: {"winner": "pieces"})
    monkeypatch.setattr(trial, "checked_reservation", lambda _: {})
    seen = []

    def opened(path: Path, receipt: object, pool: str) -> str:
        assert pool == "holdout" and (path / "opened.json").exists()
        return pool

    def measure(path: Path, data: str, names: tuple[str, ...], bundle: Path) -> object:
        assert data == "holdout" and names == (trial.CONTROL, "pieces")
        seen.append(data)
        raise RuntimeError("stop after checking the prediction request")

    monkeypatch.setattr(trial, "open_pool", opened)
    monkeypatch.setattr(trial, "measure", measure)
    with pytest.raises(RuntimeError, match="prediction request"):
        trial.score(tmp_path, tmp_path)
    with pytest.raises(FileExistsError):
        trial.score(tmp_path, tmp_path)
    assert seen == ["holdout"]


def test_positive_pooled_gain_cannot_mask_short_sideways_or_length_regression(
    trial: ModuleType,
) -> None:
    baseline = {
        "short_sideways_recall": 0.5,
        "point_macro_f1": 0.90,
        "interval_per_class": {"sideways": {"f1": 0.8}},
        "by_length": {"1024": {"interval_macro_f1": 0.75}},
    }
    paired = {"interval_macro_f1": {"interval_95pct": [0.01, 0.09]}}
    candidate = {**baseline, "short_sideways_recall": 0.45}
    assert not trial.qualification(baseline, candidate, paired)["short_sideways_recall"]
    candidate["short_sideways_recall"] = None
    assert not trial.qualification(baseline, candidate, paired)["short_sideways_recall"]
    candidate = {**baseline, "by_length": {"1024": {"interval_macro_f1": 0.70}}}
    assert not trial.qualification(baseline, candidate, paired)["length_f1"]
    assert not trial.qualification(baseline, {**baseline, "point_macro_f1": None}, paired)[
        "point_f1"
    ]
    paired["interval_macro_f1"]["interval_95pct"][0] = 0.0
    assert not trial.qualification(baseline, baseline, paired)["positive_interval_gain"]


def test_new_pools_are_balanced_and_use_distinct_generator_seeds(trial: ModuleType) -> None:
    import numpy as np

    seen: set[int] = set()
    for name, count in (("validation", 20), ("holdout", 40)):
        specs = trial.specs(name)
        assert [s.plan.step_count for s in specs] == [256, 512, 1024]
        for spec in specs:
            assert spec.plan.experiment_count == count
            seeds = {
                int(np.random.SeedSequence([spec.plan.seed, i]).generate_state(1)[0])
                for i in range(count)
            }
            assert len(seeds) == count and seeds.isdisjoint(seen)
            seen.update(seeds)


def test_exposure_inventory_includes_both_native_pools_and_known_truth_groups(
    trial: ModuleType, tmp_path: Path
) -> None:
    from regimex.benchmark_contracts import dataset_identity_keys
    from regimex.data_catalog import Partition, SourceSnapshot, create_snapshot_member, seal_dataset
    from regimex.market_export import sample_snapshot
    from regimex.market_identity import market_seed_group
    from regimex.storage import content_hash, digest

    previous, transfer = tmp_path / "old", tmp_path / "transfer"
    transfer.mkdir()
    prior = tmp_path / "known.json"
    prior_key = "a" * 64
    write_json(prior, {"excluded_seeds": [13], "identity_keys_sha256": [prior_key]})
    records = {}
    keys = {prior_key}
    partitions: tuple[Partition, ...] = ("train", "validation", "test")
    for pool, start in (("validation", 23), ("holdout", 53)):
        (previous / pool).mkdir(parents=True)
        members = tuple(
            create_snapshot_member(
                SourceSnapshot(
                    source_namespace=pool,
                    split_group=market_seed_group(seed),
                    export=sample_snapshot(seed=seed, steps=32, instruments=1),
                ),
                partition,
            )
            for seed, partition in enumerate(partitions, start)
        )
        dataset = seal_dataset(pool, members)
        write_json(previous / pool / "dataset.json", dataset.model_dump(mode="json"))
        keys.update(content_hash(key) for key in dataset_identity_keys(members))
        records[pool] = {
            "dataset_sha256": dataset.content_sha256,
            "external_inventory_sha256": digest(prior),
        }
    write_json(previous / "reservation.json", {"pools": records})
    path = transfer / "plan.json"
    write_json(
        path,
        {
            "native_reservation_sha256": digest(previous / "reservation.json"),
            "prior_sha256": digest(prior),
            "protocol": {"groups": [73, 83]},
        },
    )
    assert trial.prior_exposure(previous, transfer, prior, ()) == {
        "excluded_seeds": [13, 23, 24, 25, 53, 54, 55, 73, 83],
        "identity_keys_sha256": sorted(keys),
    }
    prior.write_bytes(b"{}")
    with pytest.raises(ValueError, match="different histories"):
        trial.prior_exposure(previous, transfer, prior, ())
