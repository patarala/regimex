import json
from importlib import import_module
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def recovery(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return import_module("compare_pause_recovery")


def test_recovery_selection_prioritizes_qualified_recall(recovery: ModuleType) -> None:
    def score(recall: float, false: float) -> dict[str, float]:
        return {"recall": recall, "false_per_1000": false}

    scores = {
        "control": score(0.75, 0.35),
        "gain": score(0.8, 0.38),
        "same-recall": score(0.8, 0.37),
        "over-budget": score(0.9, 0.40),
        "no-gain": score(0.75, 0.1),
    }
    assert recovery.choose(scores, scores) == {"winner": "same-recall", "qualified": True}
    assert recovery.choose(["no-gain", "over-budget"], scores) == {
        "winner": "no-gain",
        "qualified": False,
    }
    scores["control"] = score(0.75, 0.62)
    scores["over-budget"] = score(0.99, 0.51)
    assert recovery.choose(scores, scores)["winner"] == "same-recall"


def test_paired_recovery_gates(recovery: ModuleType) -> None:
    baseline = {"precision": 0.9}
    candidate = {"precision": 0.89, "false_per_1000": 0.50}
    paired = {"lower_recall": 0.001, "upper_false_per_1000": 0.08}
    assert recovery.passes_final(True, baseline, candidate, paired)
    assert not recovery.passes_final(False, baseline, candidate, paired)
    for delta in ({"lower_recall": 0}, {"upper_false_per_1000": 0.081}):
        assert not recovery.passes_final(True, baseline, candidate, {**paired, **delta})
    for new in ({"false_per_1000": 0.501}, {"precision": 0.889}, {"precision": None}):
        assert not recovery.passes_final(True, baseline, {**candidate, **new}, paired)


def test_reservation_excludes_other_schedules_and_transfer(
    recovery: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inventory = tmp_path / "history.json"
    first, _ = recovery.COHORTS["transfer"]
    inventory.write_text(json.dumps({"excluded_seeds": [first + 1]}))
    with pytest.raises(ValueError, match="overlap"):
        recovery.reserve(tmp_path, [inventory])
    monkeypatch.setitem(recovery.COHORTS, "final", (recovery.FIRST_SEED + 1, 1))
    with pytest.raises(ValueError, match="overlap"):
        recovery.reserve(tmp_path, [])


def test_freeze_checks_sources_and_one_time_opening(
    recovery: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = ["control", "alternative"]
    sources = {"plan_sha256": "abc", "methods": options, "candidate_sha256": "1"}
    monkeypatch.setattr(recovery, "fingerprint", lambda _: dict(sources))

    def rows(phase: str, methods: list[str], workers: int) -> list[dict[str, object]]:
        return [
            {
                "method": method,
                "market": market,
                "shape": shape,
                "group": 0,
                "expected": expected,
                "found": found,
                "matched": found,
                "later_matched": 0,
                "onset_errors": [3] * found,
                "end_errors": [4] * found,
            }
            for method in methods
            for market in ("dense", "sparse")
            for shape, expected in (("one-slowdown", 0), ("three-pauses", 3))
            for found in [min(expected, 1 if method == "control" else 2)]
        ]

    monkeypatch.setattr(recovery, "evaluate", rows)
    recovery.develop(tmp_path, workers=1)
    frozen = recovery.read_freeze(tmp_path)
    assert frozen["winner"] == "alternative" and frozen["qualified"]
    sources["candidate_sha256"] = "2"
    with pytest.raises(ValueError, match="changed"):
        recovery.confirm(tmp_path, workers=1)
    assert not (tmp_path / "opened.json").exists()
    sources["candidate_sha256"] = "1"
    recovery.confirm(tmp_path, workers=1)
    result = json.loads((tmp_path / "results.json").read_bytes())
    assert result["passed_primary"]
    assert result["scores"]["alternative"]["median_absolute_onset_errors"] == 3
    with pytest.raises(FileExistsError):
        recovery.confirm(tmp_path, workers=1)


def test_moving_unchanged_candidate_into_package_preserves_freeze(
    recovery: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = ModuleType("example")
    monkeypatch.setattr(module, "SETTINGS", ("fixed",), raising=False)
    script = tmp_path / "example.py"
    script.write_text("# selected candidate\n")
    module.__file__ = str(script)
    monkeypatch.setattr(recovery, "FAMILIES", {"example": "example"})
    monkeypatch.setattr(recovery, "candidate_module", lambda _: module)
    recovery.reserve(tmp_path, [])
    fingerprint = recovery.fingerprint(tmp_path)
    package = tmp_path / "regimex"
    package.mkdir()
    installed = script.rename(package / script.name)
    module.__file__ = str(installed)
    assert recovery.fingerprint(tmp_path) == fingerprint
    installed.write_text("# changed candidate\n")
    assert recovery.fingerprint(tmp_path) != fingerprint
