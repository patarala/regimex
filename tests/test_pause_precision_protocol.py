import json
from importlib import import_module
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def design(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return import_module("confirm_pause_precision")


def test_selection_requires_a_supported_false_rate_and_recall(design: ModuleType) -> None:
    def metrics(recall: float, point: float, upper: float) -> dict[str, float]:
        return {"recall": recall, "false_per_1000": point, "upper_false_per_1000": upper}

    scores = {
        "report": metrics(0.96, 0.7, 0.9),
        "uncertain": metrics(0.93, 0.48, 0.54),
        "strict": metrics(0.79, 0.2, 0.38),
        "eligible": metrics(0.9, 0.41, 0.49),
        "equally-sensitive": metrics(0.9, 0.4, 0.5),
    }
    assert design.choose(scores, scores) == "equally-sensitive"
    assert design.choose(["uncertain", "strict"], scores) == "report"


def test_confirmation_requires_all_four_gates(design: ModuleType) -> None:
    control = {"precision": 0.8}
    candidate = {"precision": 0.9, "recall": 0.85, "false_per_1000": 0.45}
    assert design.passes_final(control, candidate, {"upper_false_per_1000": -0.01})
    assert not design.passes_final(control, candidate, {"upper_false_per_1000": 0.0})
    for setting in ({"precision": 0.8}, {"recall": 0.79}, {"false_per_1000": 0.51}):
        assert not design.passes_final(
            control, {**candidate, **setting}, {"upper_false_per_1000": -0.01}
        )


def test_bootstrap_pools_detections_within_whole_groups(design: ModuleType) -> None:
    rows = []
    for seed, matched, extras in ((0, 1, 8), (1, 3, 0)):
        for shape, found, expected in (("three-pauses", matched, 3), ("one-slowdown", extras, 0)):
            rows.append(
                {
                    "group": seed,
                    "shape": shape,
                    "expected": expected,
                    "found": found,
                    "matched": found if expected else 0,
                    "later_matched": max(0, matched - 1) if expected else 0,
                }
            )
    metrics = design.summarize(rows)
    assert metrics["precision"] == 4 / 12 and metrics["recall"] == 4 / 6
    draws = design.group_draws(rows)
    assert set(draws["precision"]) == {1 / 9, 4 / 12, 1}
    assert design.paired_intervals(draws, draws)["false_per_1000"] == [0, 0]


def test_heldout_opening_and_source_validation(
    design: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = {"new_filter": "digest"}
    options = ["report", "threshold/4/0.25", "shape/flat"]
    monkeypatch.setattr(design, "candidate_sources", lambda: dict(sources))
    monkeypatch.setattr(design, "THRESHOLDS", {options[1]: (4.0, 0.25)})
    monkeypatch.setattr(design, "methods", lambda: options)

    def evaluate(phase: str, methods: list[str], workers: int) -> list[dict[str, object]]:
        return [
            {
                "group": seed,
                "market": market,
                "shape": shape,
                "method": method,
                "severity": "near-flat",
                "expected": expected,
                "matched": expected,
                "later_matched": max(0, expected - 1),
                "found": expected + int("slowdown" in shape and method == "report"),
            }
            for seed in (0, 2)
            for market in ("dense", "sparse")
            for shape, expected in (
                ("steady", 0),
                ("one-slowdown", 0),
                ("one-pause", 1),
                ("three-pauses", 3),
            )
            for method in methods
        ]

    monkeypatch.setattr(design, "evaluate", evaluate)
    design.reserve(tmp_path, [])
    design.develop(tmp_path, workers=1)
    frozen = design.selection(tmp_path)
    design.write_new_file(tmp_path / "freeze.json", design.json_bytes(frozen))
    assert design.read_freeze(tmp_path) == frozen
    sources["new_filter"] = "modified"
    with pytest.raises(ValueError, match="changed"):
        design.confirm(tmp_path, workers=1)
    assert not (tmp_path / "opened.json").exists()
    sources["new_filter"] = "digest"
    design.confirm(tmp_path, workers=1)
    results = json.loads((tmp_path / "results.json").read_bytes())
    selected = results["selected"]
    assert results["adopted"] and results["results"][selected]["overall"]["precision"] == 1
    assert results["results"][selected]["market"]["sparse"]["all_three"] == 2
    with pytest.raises(FileExistsError):
        design.confirm(tmp_path, workers=1)
    with pytest.raises(FileExistsError):
        design.develop(tmp_path, workers=1)


def test_fixed_replication_rejects_exposed_candidates_and_seed_overlap(
    design: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replica = import_module("replicate_pause_precision")
    frozen = {"finalists": ["report", "shape/shape-5"], "candidates": {}}
    monkeypatch.setattr(replica, "read_freeze", lambda path: frozen)
    (tmp_path / "freeze.json").write_bytes(design.json_bytes(frozen))
    protocol = design.protocol()
    (tmp_path / "plan.json").write_bytes(
        design.json_bytes({"inventories": {}, "protocol": protocol})
    )
    (tmp_path / "results.json").write_bytes(
        design.json_bytes(
            {"adopted": False, "freeze_sha256": design.digest(tmp_path / "freeze.json")}
        )
    )
    plan = replica.reserve_protocol(tmp_path)
    assert plan["methods"] == ["report", "shape/shape-6"]
    assert len(plan["group_seeds"]) == 384
    with monkeypatch.context() as local:
        local.setattr(replica, "FIRST_SEED", replica.TCN_COHORTS[0] + 1)
        with pytest.raises(ValueError, match="overlap"):
            replica.reserve_protocol(tmp_path)
    frozen["finalists"] = ["report", "shape/shape-6"]
    with pytest.raises(ValueError, match="untested"):
        replica.reserve_protocol(tmp_path)
