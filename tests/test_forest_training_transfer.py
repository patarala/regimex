"""Guard the new transfer groups against both earlier evaluations and the training extension."""

from importlib import import_module
from pathlib import Path

import pytest


def test_new_groups_are_disjoint_from_previous_and_current_exposures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    transfer = import_module("check_forest_training_transfer")
    previous = [8, 11]
    transfer.check_seeds(previous, {3, 8, 11}, {12, 13})
    with pytest.raises(ValueError, match="omits"):
        transfer.check_seeds(previous, {8}, set())
    for prior, native in (({8, 11, transfer.GROUPS[1]}, set()), ({8, 11}, {transfer.GROUPS[0]})):
        with pytest.raises(ValueError, match="intersect"):
            transfer.check_seeds(previous, prior, native)


def test_paired_replay_keeps_model_scores_and_observed_denominators_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from regimex.model_contracts import ObservationInputs

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    transfer = import_module("check_forest_training_transfer")
    pair = next(transfer.replay.pairs(4317, 0))
    full, _, _ = pair
    assert full.shape == "steady"
    monkeypatch.setattr(transfer.replay, "pairs", lambda seed, index: [pair])
    trend = int(full.directions[0])
    keys = ("always-sideways", "always-trend")

    def probabilities(inputs: ObservationInputs) -> dict[str, object]:
        return {
            method: np.tile(np.eye(3)[direction], (inputs.row_count, 1))
            for method, direction in zip(keys, (0, trend), strict=True)
        }

    counters = {method: transfer.replay.Totals() for method in keys}
    transfer.evaluate_group(4317, 0, probabilities, counters)
    baseline, matched = (c.summary() for c in counters.values())
    name = transfer.DIRECTION_NAMES[trend]
    assert baseline["interval"]["pooled"]["per_class"][name]["recall"] == 0
    assert matched["interval"]["pooled"]["per_class"][name]["recall"] == 1
    assert matched["changed"]["pooled"] == baseline["changed"]["pooled"]
    assert matched["changed"]["pooled"]["common"] == 448 - 40 - 32
    assert matched["changed"]["pooled"]["changed"] == 0


def test_release_winning_validation_preserves_the_new_transfer_population(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    transfer = import_module("check_forest_training_transfer")
    native = import_module("compare_forest_training_mix")
    monkeypatch.setattr(transfer, "design", lambda *_: {})
    monkeypatch.setattr(native, "read_freeze", lambda _: {"winner": native.CONTROL})
    (tmp_path / "plan.json").write_text("{}")
    with pytest.raises(ValueError, match="remain reserved"):
        transfer.score(tmp_path, tmp_path, tmp_path, tmp_path)
    assert not (tmp_path / "opened.json").exists()
