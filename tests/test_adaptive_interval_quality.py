"""Protect the selection and qualification rules of the adaptive interval comparison."""

import importlib
import json
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
    trial: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from regimex.storage import digest

    previous, transfer = tmp_path / "old", tmp_path / "transfer"
    transfer.mkdir()
    prior = tmp_path / "known.json"
    write_json(prior, {"excluded_seeds": [13], "identity_keys_sha256": []})
    records = {}
    for pool in ("validation", "holdout"):
        (previous / pool).mkdir(parents=True)
        write_json(previous / pool / "dataset.json", {"digest": pool})
        records[pool] = {"dataset_sha256": pool, "external_inventory_sha256": digest(prior)}
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
    monkeypatch.setattr(
        trial.DatasetManifest,
        "model_validate_json",
        lambda value: SimpleNamespace(content_sha256=json.loads(value)["digest"]),
    )

    def build(datasets: list[Path], indexes: list[Path]) -> dict[str, object]:
        assert [p.parent.name for p in datasets] == ["validation", "holdout"]
        assert indexes == [prior]
        return {"excluded_seeds": [13, 23, 53], "identity_keys_sha256": ["prior-key"]}

    monkeypatch.setattr(trial, "build", build)
    assert trial.prior_exposure(previous, transfer, prior, ()) == {
        "excluded_seeds": [13, 23, 53, 73, 83],
        "identity_keys_sha256": ["prior-key"],
    }
    prior.write_bytes(b"{}")
    with pytest.raises(ValueError, match="different histories"):
        trial.prior_exposure(previous, transfer, prior, ())
