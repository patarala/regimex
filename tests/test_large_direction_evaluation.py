"""Keep final experiment access bound to a frozen, reproducible challenger."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from regimex.storage import digest, write_json


@pytest.fixture
def trial(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return import_module("evaluate_large_direction")


def test_frozen_challenger_can_be_measured_when_release_wins_validation(
    trial: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation = {name: {"interval_macro_f1": 0.8} for name in (trial.CONTROL, *trial.CHALLENGERS)}
    validation[trial.CONTROL]["interval_macro_f1"] = 0.9
    winner = trial.choose(validation, (trial.CONTROL, *trial.CHALLENGERS))
    new = trial.choose(validation, trial.CHALLENGERS)
    assert winner == trial.CONTROL and new == trial.CHALLENGERS[0]
    selection = tmp_path / "selection"
    selection.mkdir()
    protocol = tmp_path / "protocol.json"
    write_json(protocol, trial.protocol())
    frozen = {
        "control_bundle_sha256": trial.BUNDLE_SHA,
        "bundle_sha256": "0" * 64,
        "protocol_sha256": digest(protocol),
        "inputs": {},
        "implementation": {},
        "validation_winner": winner,
        "challenger": new,
        "validation": validation,
    }
    write_json(selection / "freeze.json", frozen)
    monkeypatch.setattr(trial, "implementation", dict)
    monkeypatch.setattr(trial, "evidence_files", lambda _: {})
    monkeypatch.setattr(
        trial, "AdaptiveBackend", lambda: SimpleNamespace(load_bundle=lambda *a, **k: None)
    )
    monkeypatch.setattr(trial, "development", lambda _: (None, None))
    monkeypatch.setattr(trial, "PartitionCorpus", lambda *args: None)

    def measure(corpus, references, models):
        assert references is None and list(models) == [trial.CONTROL, new]
        assert trial.read(tmp_path / "opened.json") == {
            "freeze_sha256": digest(selection / "freeze.json")
        }
        raise RuntimeError("stop at first prediction")

    monkeypatch.setattr(trial, "measure", measure)
    with pytest.raises(RuntimeError, match="first prediction"):
        trial.score(tmp_path, tmp_path, tmp_path)
    # A failed attempt may resume only against the same freeze.
    write_json(selection / "freeze.json", {**frozen, "other": True})
    with pytest.raises(ValueError, match="another freeze"):
        trial.score(tmp_path, tmp_path, tmp_path)
    (tmp_path / "opened.json").unlink()
    write_json(
        selection / "freeze.json",
        {**frozen, "bundle_sha256": trial.BUNDLE_SHA, "challenger": "other"},
    )
    with pytest.raises(ValueError, match="inputs changed"):
        trial.score(tmp_path, tmp_path, tmp_path)
    assert not (tmp_path / "opened.json").exists()


def test_partition_loader_limits_shared_catalog_to_its_own_locations(
    trial: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roles = ("train", "validation", "test")
    members = tuple(SimpleNamespace(partition=role, source_namespace=role) for role in roles)
    dataset = SimpleNamespace(content_sha256="shared", members=members)
    locations = {"validation": "exports/1.json"}

    def load_corpus(corpus, path, expected_digest):
        assert expected_digest == "shared" and path == tmp_path / "validation"
        corpus.dataset = dataset
        corpus.locations = locations

    monkeypatch.setattr(trial.Corpus, "__init__", load_corpus)
    assert trial.PartitionCorpus(tmp_path, "validation", dataset).members == (members[1],)
    locations["test"] = "exports/2.json"
    with pytest.raises(ValueError, match="differ from the registered split"):
        trial.PartitionCorpus(tmp_path, "validation", dataset)
