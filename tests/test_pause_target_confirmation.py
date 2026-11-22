"""New-pause recall compares the same two frozen predictors and the same references."""

import importlib
from pathlib import Path

import pytest

from regimex.directions import Direction


def test_short_reference_must_be_new_to_count_in_primary_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    from test_evaluation import intervals, rows

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    comparison = importlib.import_module("confirm_pause_targets")
    observations = rows(160)
    growth, flat = Direction.GROWTH, Direction.SIDEWAYS
    original = intervals(observations, [(0, 32, flat), (32, 160, growth)])
    refined = intervals(
        observations, [(0, 32, flat), (32, 64, growth), (64, 96, flat), (96, 160, growth)]
    )
    # Hitting only the existing short sideways interval is a miss on the newly inserted pause.
    old = comparison.summaries(comparison.sequence_counts(original, refined, original))
    assert old["original_references"]["short_sideways_matched"] == 1
    assert old["refined_references"]["short_sideways_matched"] == 1
    assert (old["new_pause_matched"], old["new_pause_references"]) == (0, 1)
    new = comparison.summaries(comparison.sequence_counts(original, refined, refined))
    assert (new["new_pause_matched"], new["new_pause_references"]) == (1, 1)
    assert new["original_references"]["false_sideways_intervals"] == 1
    assert new["refined_references"]["false_sideways_intervals"] == 0


def test_no_new_reference_leaves_primary_recall_undefined(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    from test_evaluation import intervals, rows

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    comparison = importlib.import_module("confirm_pause_targets")
    labels = intervals(rows(96), [(0, 96, Direction.GROWTH)])
    counts = comparison.sequence_counts(labels, labels, labels).tolist()
    records = [
        {"steps": size, "counts": {"original": counts, "refined": counts}}
        for size in comparison.LENGTHS
    ]
    monkeypatch.setattr(comparison, "DRAWS", 20)
    scores = comparison.score_groups(records)
    assert scores["original"]["new_pause_recall"] is None
    assert scores["paired_differences"]["new_pause_recall"] == {
        "difference": None,
        "interval_95pct": None,
        "defined_draws": 0,
    }
    assert scores["paired_differences"]["original_references"]["interval_95pct"] == [0.0, 0.0]


def test_incomplete_predictor_pair_keeps_holdout_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    comparison = importlib.import_module("confirm_pause_targets")
    training = importlib.import_module("fit_pause_targets")
    monkeypatch.setattr(training, "training_protocol", dict)
    monkeypatch.setattr(training, "load_predictors", lambda *_: {"original": object()})
    monkeypatch.setattr(comparison, "scientific_digests", dict)
    plan = comparison.content_hash(comparison.protocol())
    (tmp_path / "holdout").mkdir()
    comparison.write(tmp_path / "plan.json", comparison.protocol())
    comparison.write(tmp_path / "holdout" / "locations.json", {})
    comparison.write(tmp_path / "prior.json", {})
    digest = comparison.sha256
    assert comparison.__file__ and training.__file__
    comparison.write(
        tmp_path / "reservation.json",
        {
            "evaluator_sha256": digest(Path(comparison.__file__)),
            "training_script_sha256": digest(Path(training.__file__)),
            "plan_sha256": plan,
            "training_protocol": {},
            "environment": {},
            "locations_sha256": digest(tmp_path / "holdout" / "locations.json"),
            "external_inventory_sha256": digest(tmp_path / "prior.json"),
        },
    )
    (tmp_path / "fitted").mkdir()
    comparison.write(
        tmp_path / "fitted" / "freeze.json",
        {"reservation_sha256": digest(tmp_path / "reservation.json")},
    )

    def opened(*args: object, **kwargs: object) -> None:
        pytest.fail("Holdout opened before checking both registered predictors")

    monkeypatch.setattr(comparison, "Corpus", opened)
    with pytest.raises(ValueError, match="Both registered models"):
        comparison.score(tmp_path, tmp_path)
    assert not (tmp_path / "opened.json").exists()
