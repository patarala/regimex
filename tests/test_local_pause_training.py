"""Check that the extra channels preserve the baseline initialization and the data split."""

import importlib
from pathlib import Path
from types import ModuleType

import pytest

from regimex.storage import digest, write_json


@pytest.fixture
def fitting(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("fit_local_pause_features")


def test_extra_channels_start_at_zero_with_identical_native_parameters_and_rng(
    fitting: ModuleType,
) -> None:
    torch = pytest.importorskip("torch")
    width = len(fitting.TEMPORAL_NAMES)
    params = fitting.PARAMETERS
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        reference = fitting.TCNClassifier(width, params.channels, params.dropout, params.dilations)
        following = torch.rand(5)
        torch.manual_seed(23)
        expanded = fitting.build(width + 2 * len(fitting.PAUSE_FEATURE_NAMES))
        assert torch.equal(torch.rand(5), following)
        for name, expected in reference.state_dict().items():
            actual = expanded.state_dict()[name]
            if name == "input.weight":
                assert torch.count_nonzero(actual[:, width:]) == 0
                actual = actual[:, :width]
            assert torch.equal(actual, expected)


def test_extra_features_fit_on_training_and_keep_missing_value_masks(fitting: ModuleType) -> None:
    np = pytest.importorskip("numpy")
    from regimex.prediction_inputs import PredictionSequenceInfo

    width = len(fitting.PAUSE_FEATURE_NAMES)
    training = np.tile(np.array([1.0, 3.0, np.nan])[:, None], (1, width))
    training[:, 0] = np.nan
    prep = fitting.extra_preprocessor(training)
    assert prep.fill.tolist() == [0.0] + [2.0] * (width - 1)
    assert prep.mean.tolist() == prep.fill.tolist()
    assert prep.scale[0] == 1.0
    probe = np.array([[np.nan] + [10.0] * (width - 1)])
    raw = fitting.PreparedPredictionInputs(
        np.zeros((1, 2), dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
        (PredictionSequenceInfo(experiment_id="one", instrument_id="A/B", step_count=1),),
    )
    transformed = fitting.extend(raw, probe, prep).values[0]
    np.testing.assert_array_equal(transformed[:2], raw.values[0])
    np.testing.assert_allclose(transformed[2 : 2 + width], prep.transform(probe)[0])
    assert transformed[2 + width :].tolist() == [1.0] + [0.0] * (width - 1)
    assert transformed[3] > 5.0


def test_changed_preprocessing_blocks_model_loading(
    fitting: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "fitted"
    folder.mkdir()
    receipt = tmp_path / "reservation.json"
    write_json(receipt, {})
    artifacts = {}
    for name in ("weights.npz", "preprocessing.npz", "history.json"):
        path = folder / name
        path.write_bytes(name.encode())
        artifacts[name] = digest(path)
    write_json(
        folder / "freeze.json", {"reservation_sha256": digest(receipt), "artifacts": artifacts}
    )
    (folder / "preprocessing.npz").write_bytes(b"other training population")
    monkeypatch.setattr(fitting, "check_receipt", lambda *args: {})
    with pytest.raises(ValueError, match="Frozen model, preprocessing"):
        fitting.load_predictors(tmp_path, tmp_path, tmp_path)
