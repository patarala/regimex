"""Check that the secondary postprocessor receives only the control's predicted intervals."""

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_data_catalog import snapshot

from regimex.data_catalog import create_snapshot_member


def test_hybrid_reuses_single_control_forward_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    import numpy as np

    from regimex.model_contracts import ObservationInputs

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    comparison = importlib.import_module("confirm_local_pause_features")
    source = snapshot(1, instrument_count=2)
    member = create_snapshot_member(source, "test")
    corpus = SimpleNamespace(dataset=SimpleNamespace(members=[member]), snapshot=lambda _: source)

    class Predictor:
        def __init__(self, direction: int) -> None:
            self.direction = direction
            self.calls = 0

        def predict(self, inputs: ObservationInputs, *, runtime: object) -> Any:
            self.calls += 1
            assert runtime == comparison.RUNTIME
            result = np.zeros((inputs.row_count, 3))
            result[:, self.direction] = 1
            return result

    models = {"control": Predictor(1), "candidate": Predictor(2)}
    decoded = []
    original_decode = comparison.decode

    def capture(*args: object) -> Any:
        intervals = original_decode(*args)
        decoded.append(intervals)
        return intervals

    def refine(rows: object, intervals: object, config: object) -> object:
        assert intervals is decoded[-2]
        return intervals

    monkeypatch.setattr(comparison, "decode", capture)
    monkeypatch.setattr(comparison, "refine_decoded", refine)
    (record,) = comparison.evaluate(corpus, models)
    assert [m.calls for m in models.values()] == [1, 1]
    assert len(decoded) == 4
    assert record["counts"]["hybrid"] == record["counts"]["control"]


def test_substituted_freeze_does_not_open_benchmark(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    evaluator = importlib.import_module("confirm_local_pause_features")
    fitting = importlib.import_module("fit_local_pause_features")
    monkeypatch.setattr(fitting, "training_protocol", dict)
    monkeypatch.setattr(evaluator, "evaluator_digests", dict)
    prior = tmp_path / "previous"
    for path in (
        prior / "fitted/freeze.json",
        prior / "fitted/weights.npz",
        tmp_path / "prior.json",
        tmp_path / "holdout/locations.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}")
    digest = evaluator.sha256
    evaluator.write(tmp_path / "plan.json", evaluator.protocol())
    evaluator.write(
        tmp_path / "reservation.json",
        {
            "plan_sha256": digest(tmp_path / "plan.json"),
            "previous_freeze_sha256": digest(prior / "fitted/freeze.json"),
            "control_weights_sha256": digest(prior / "fitted/weights.npz"),
            "external_inventory_sha256": digest(tmp_path / "prior.json"),
            "locations_sha256": digest(tmp_path / "holdout/locations.json"),
            "environment": {},
            "training_protocol": {},
        },
    )
    (tmp_path / "fitted").mkdir()
    evaluator.write(tmp_path / "fitted/freeze.json", {"reservation_sha256": "different"})
    with pytest.raises(ValueError, match="frozen against this reservation"):
        evaluator.score(tmp_path, tmp_path, prior)
    assert not (tmp_path / "opened.json").exists()
