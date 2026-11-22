"""Check the path-level denominators used to compare synthetic annotations."""

import importlib
from pathlib import Path

import pytest


def test_short_recall_and_false_trends_use_different_path_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    confirmation = importlib.import_module("confirm_trend_pauses")
    # Axes are path, method, direction and (true positive, false positive, false negative).
    groups = {key: np.zeros((1, 3, 3, 3), dtype=int) for key in ("flat", "plateau-32", "slowdown")}
    groups["flat"][:, :, 0, 0] = 1
    groups["flat"][:, :, 2, 1] = 2
    groups["slowdown"][:, :, 0, 1] = 1
    groups["plateau-32"][:, :2, 0, 2] = 1
    groups["plateau-32"][:, 2, 0, 0] = 1
    current, refined = (confirmation.scores(groups)[method] for method in ("current", "refined"))
    assert current["short_sideways_matched"] == 0
    assert refined["short_sideways_matched"] == refined["short_sideways_references"] == 1
    assert refined["false_trends_on_level_paths"] == 2
    assert refined["false_sideways_on_trends"] == 1
    bounds = confirmation.paired_bounds(groups)
    assert bounds["short_sideways_recall"] == [1, 1]
    assert bounds["interval_macro_f1"] == pytest.approx(
        [refined["interval_macro_f1"] - current["interval_macro_f1"]] * 2
    )
