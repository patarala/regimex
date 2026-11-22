"""Pair saved winner scores with the release by experiment identity."""

from importlib import import_module
from pathlib import Path

import pytest


def test_closed_winner_counts_match_identities_and_reported_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    replay = import_module("check_direction_release")
    original = np.stack((np.arange(20), np.ones(20, dtype=int)))
    score = replay.summary(original)
    primary = {
        "experiment_counts": {"context": {"b": original[1].tolist(), "a": original[0].tolist()}},
        "scores": {"context": score},
    }
    control = (["a", "b"], np.array([256, 512]), np.zeros_like(original))
    ids, lengths, rows = replay.winner_counts(primary, "context", control)
    assert ids == control[0]
    np.testing.assert_array_equal(lengths, control[1])
    np.testing.assert_array_equal(rows, original)
    with pytest.raises(ValueError, match="identities"):
        replay.winner_counts(primary, "context", (["b", "c"], lengths, rows))
    score["interval_macro_f1"] = -1
    with pytest.raises(ValueError, match="closed score"):
        replay.winner_counts(primary, "context", control)
