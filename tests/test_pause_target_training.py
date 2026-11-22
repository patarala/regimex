"""The offline teacher can only replace training targets on the registered grid."""

import importlib
from pathlib import Path
from types import ModuleType

import pytest
from test_data_catalog import dataset, raw_training_data

from regimex.storage import digest, write_json


@pytest.fixture
def comparison(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("fit_pause_targets")


@pytest.mark.parametrize("source", ["validation", "duplicate", "mixed"])
def test_refinement_rejects_foreign_members_before_annotating(
    source: str, comparison: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, _ = dataset()
    raw = raw_training_data()
    if source == "mixed":
        samples = raw.train + raw.validation
    elif source == "duplicate":
        samples = raw.train * 2
    else:
        samples = raw.validation

    def unexpected(*args: object) -> None:
        pytest.fail("The teacher received observations outside the registered training partition")

    monkeypatch.setattr(comparison, "annotate_pauses", unexpected)
    with pytest.raises(ValueError, match="complete ordered training partition"):
        comparison.refined_targets(manifest, samples)


@pytest.mark.parametrize("damaged", ["weights", "history", "reservation"])
def test_scoring_rejects_tampered_fit_before_loading_a_model(
    damaged: str, comparison: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = "a" * 64
    fitted = tmp_path / "fitted"
    fitted.mkdir()
    weights = fitted / "weights.npz"
    weights.write_bytes(b"weights")
    history = fitted / "history.json"
    history.write_text("{}")
    reservation = tmp_path / "reservation.json"
    write_json(reservation, {"control_sha256": control})
    write_json(
        fitted / "freeze.json",
        {
            "control_sha256": control,
            "reservation_sha256": digest(reservation),
            "weights_sha256": digest(weights),
            "history_sha256": digest(history),
        },
    )
    {"weights": weights, "history": history, "reservation": reservation}[damaged].write_bytes(
        b"changed"
    )
    monkeypatch.setattr(comparison, "checked_receipt", lambda _: {"control_sha256": control})
    monkeypatch.setattr(comparison, "load_control", lambda *_: pytest.fail("Loaded a changed fit"))
    with pytest.raises(ValueError, match="frozen weights, training history or reservation changed"):
        comparison.load_predictors(tmp_path, tmp_path)
