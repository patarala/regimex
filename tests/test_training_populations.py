"""Check which training changes can be attributed to size, horizon and optimizer budget."""

import importlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_nested_populations_and_fixed_update_contrasts(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    comparison = importlib.import_module("compare_training_populations")

    def member(steps: int, instruments: int, seed: int, partition: str) -> SimpleNamespace:
        return SimpleNamespace(
            step_count=steps,
            instrument_ids=tuple(range(instruments)),
            source_namespace=str(seed),
            split_group=SimpleNamespace(seed=seed),
            partition=partition,
        )

    original = [
        member(steps, 1 if i < 23 else 2, -steps - i, "train")
        for steps in (256, 512)
        for i in range(48)
    ]
    extra = [member(*entry) for entry in comparison.extra_experiments()]
    populations = comparison.uniform_cohorts(
        SimpleNamespace(members=[*original, member(512, 2, -1, "validation")]),
        SimpleNamespace(members=extra),
    )
    assert populations["short-96"] == tuple(original)
    seeds = {name: {m.split_group.seed for m in train} for name, train in populations.items()}
    assert {name: len(s) for name, s in seeds.items()} == {
        "short-96": 96,
        "mixed-96": 96,
        "short-192": 192,
        "mixed-192": 192,
    }
    assert seeds["short-96"] <= seeds["short-192"]
    assert seeds["mixed-96"] <= seeds["mixed-192"]
    for name, size in (("mixed-96", 16), ("mixed-192", 32)):
        assert Counter((m.step_count, len(m.instrument_ids)) for m in populations[name]) == {
            (steps, instruments): size for steps in (256, 512, 1024) for instruments in (1, 2)
        }
    assert all(m.partition == "train" for pop in populations.values() for m in pop)
    standard = comparison.policy_for("mixed-192")
    longer = comparison.policy_for("mixed-192-double")
    assert standard.seeds == longer.seeds
    assert standard.batch_size == longer.batch_size
    assert longer.updates == 2 * standard.updates
    assert longer.validation_updates == tuple(2 * step for step in standard.validation_updates)


def test_history_reports_actual_rows_and_keeps_the_first_best_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    comparison = importlib.import_module("compare_training_populations")
    policy = comparison.FixedUpdatePolicy(
        seeds=(23,), updates=3, validation_updates=(1, 3), batch_size=2
    )
    rows = [
        {
            "seed": 23,
            "update": update,
            "pass_index": update,
            "batch_in_pass": 1,
            "training_loss": 1.0,
            "trained_sequences": 2,
            "real_rows": 6,
            "padding_rows": 2,
            "eligible_rows": 5,
            "validation_segment_macro_f1": 0.75 if update in (1, 3) else None,
        }
        for update in (1, 2, 3)
    ]
    path = tmp_path / "history.json"
    path.write_bytes(comparison.json_bytes({"updates": rows}))
    assert comparison.per_seed_history(path, policy) == [
        {
            "seed": 23,
            "update": 1,
            "interval_macro_f1": 0.75,
            "optimizer_updates": 3,
            "trained_sequences": 6,
            "real_rows": 18,
            "padding_rows": 6,
            "eligible_rows": 15,
        }
    ]
    path.write_bytes(comparison.json_bytes({"updates": rows[:-1]}))
    with pytest.raises(ValueError, match="Recorded optimizer updates"):
        comparison.per_seed_history(path, policy)
