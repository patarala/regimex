"""Verify forest training membership, per-instrument weights and final selection."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from regimex.storage import write_json


@pytest.fixture
def trial(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return import_module("compare_forest_training_mix")


def test_weights_restart_on_each_instrument(
    trial: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    short, long = np.zeros(32, dtype=np.int64), np.zeros(512, dtype=np.int64)
    batch = SimpleNamespace(
        inputs=SimpleNamespace(info=[SimpleNamespace(step_count=len(y)) for y in (short, long)]),
        references=(short, long),
    )
    monkeypatch.setattr(trial, "batches", lambda *_: [batch])
    monkeypatch.setattr(trial, "matrix", lambda *_: np.arange(544)[:, None])
    monkeypatch.setattr(trial, "dense_labels", lambda reference, _: reference)
    x, y, w = trial.training_arrays(((object(), ["experiment"]),))
    np.testing.assert_array_equal(x[:, 0], np.arange(len(y)))
    np.testing.assert_array_equal(w, np.repeat([2, 0.5], [32, 512]))


def test_fit_reads_only_retained_train_and_whole_extension(
    trial: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from regimex.storage import digest

    write_json(tmp_path / "locations.json", {})
    receipt = {"training_locations_sha256": digest(tmp_path / "locations.json")}
    monkeypatch.setattr(trial, "checked_reservation", lambda _: receipt)
    members = tuple(SimpleNamespace(partition=p) for p in ("test", "train", "validation", "train"))
    extension = tuple(SimpleNamespace(partition=p) for p in ("test", "validation", "train"))
    monkeypatch.setattr(
        trial, "Corpus", lambda *_, **__: SimpleNamespace(dataset=SimpleNamespace(members=members))
    )

    def pool(output: Path, registered: dict[str, str], name: str) -> object:
        assert output == tmp_path and registered == receipt and name == "extension"
        return SimpleNamespace(dataset=SimpleNamespace(members=extension))

    def arrays(corpora: tuple[tuple[object, tuple[SimpleNamespace, ...]], ...]) -> None:
        assert list(corpora[0][1]) == [members[1], members[3]]
        assert corpora[1][1] == extension
        raise RuntimeError("checked training membership")

    monkeypatch.setattr(trial, "registered_corpus", pool)
    monkeypatch.setattr(trial, "fingerprint", lambda: {})
    monkeypatch.setattr(trial, "training_arrays", arrays)
    with pytest.raises(RuntimeError, match="training membership"):
        trial.train(tmp_path, tmp_path)


def test_final_opens_once_and_only_for_one_winner(
    trial: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scores = {name: {"interval_macro_f1": 0.75} for name in trial.CANDIDATES}
    assert trial.choose(scores) == trial.CONTROL
    monkeypatch.setattr(trial, "read_freeze", lambda _: {"winner": trial.choose(scores)})
    with pytest.raises(ValueError, match="holdout remains reserved"):
        trial.score(tmp_path, tmp_path)
    scores["balanced"]["interval_macro_f1"] = 0.76
    write_json(tmp_path / "freeze.json", scores)

    def measure(path: Path, pool: str, names: tuple[str, ...], _: Path) -> None:
        assert (path / "opened.json").exists() and pool == "holdout"
        assert names == (trial.CONTROL, "balanced")
        raise RuntimeError("checked final selection")

    monkeypatch.setattr(trial, "measure", measure)
    with pytest.raises(RuntimeError, match="checked final selection"):
        trial.score(tmp_path, tmp_path)
    with pytest.raises(FileExistsError):
        trial.score(tmp_path, tmp_path)


def test_pooled_probabilities_weight_each_observed_row_and_sort_paired_groups(
    trial: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    records = (SimpleNamespace(name="z", step_count=256), SimpleNamespace(name="a", step_count=512))
    probabilities = {"z": np.array([[1.0, 0, 0]]), "a": np.full((3, 3), 1 / 3)}
    labels = {key: np.zeros(len(value), dtype=np.int64) for key, value in probabilities.items()}
    monkeypatch.setattr(trial, "checked_reservation", lambda _: {})
    monkeypatch.setattr(
        trial,
        "registered_corpus",
        lambda *_: SimpleNamespace(dataset=SimpleNamespace(members=records)),
    )
    monkeypatch.setattr(trial, "experiment_id", lambda record: record.name)
    monkeypatch.setattr(
        trial, "batches", lambda _, items: [SimpleNamespace(inputs=r.name, record=r) for r in items]
    )
    monkeypatch.setattr(trial, "targets", lambda batch: labels[batch.inputs])
    monkeypatch.setattr(
        trial, "load_predictors", lambda *_: lambda name: {trial.CONTROL: probabilities[name]}
    )
    monkeypatch.setattr(
        trial,
        "experiment_counts",
        lambda batch, _: (
            [batch.inputs],
            np.array([batch.record.step_count]),
            np.zeros((1, 20), dtype=np.int64),
        ),
    )
    results, groups = trial.measure(tmp_path, "validation", (trial.CONTROL,), tmp_path)
    expected = trial.probability_metrics(
        np.concatenate(list(labels.values())), np.concatenate(list(probabilities.values()))
    )
    assert results[trial.CONTROL]["probabilities"] == pytest.approx(expected)
    ids, lengths, _ = groups[trial.CONTROL]
    assert ids == ["a", "z"] and lengths.tolist() == [512, 256]
