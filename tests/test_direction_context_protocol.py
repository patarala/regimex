"""Guard the split, validation-only choice and reconstruction of direction candidates."""

import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from regimex.storage import digest, write_json


@pytest.fixture
def comparison(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("compare_direction_context")


def test_native_lengths_and_control_tie_break(comparison: ModuleType) -> None:
    import numpy as np

    seen: set[int] = set()
    for pool, count in (("validation", 24), ("holdout", 48)):
        plans = [s.plan for s in comparison.specs(pool)]
        assert [p.step_count for p in plans] == [256, 512, 1024]
        for plan in plans:
            assert plan.experiment_count == count
            seeds = {
                int(np.random.SeedSequence([plan.seed, index]).generate_state(1)[0])
                for index in range(plan.experiment_count)
            }
            assert len(seeds) == count and seeds.isdisjoint(seen)
            seen.update(seeds)
    scores = {name: {"interval_macro_f1": 0.6} for name in comparison.CANDIDATES}
    scores["release-native"] = {"interval_macro_f1": 0.99}
    assert comparison.choose(scores) == comparison.CONTROL
    scores["adaptive+tcn"] = {"interval_macro_f1": 0.61}
    assert comparison.choose(scores) == "adaptive+tcn"


def test_split_features_and_restore_only_requested_predictors(
    comparison: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    np = pytest.importorskip("numpy")
    events: list[str] = []
    native = [0.1, 0.7, 0.2]
    learned = [0.3, 0.5, 0.2]

    def control_predict(inputs: SimpleNamespace, *, runtime: object) -> object:
        events.append("control predict")
        assert runtime is comparison.RUNTIME
        return np.tile(native, (sum(map(len, inputs.observations)), 1))

    def load_control(folder: Path, *, manifest_sha256: str, runtime: object) -> object:
        assert folder == tmp_path
        assert manifest_sha256 == comparison.BUNDLES[comparison.CONTROL]
        events.append("load control")
        return SimpleNamespace(analysis=comparison.PROFILE, predict=control_predict)

    def load_learned(path: Path) -> object:
        assert path == tmp_path / "fitted/context.npz"
        events.append("load context")
        return SimpleNamespace(predict=lambda x: np.tile(learned, (len(x), 1)))

    def features(rows: tuple[int, ...], profile: object) -> object:
        assert profile is comparison.PROFILE
        events.append(f"features {rows}")
        return np.ones((len(rows), 2))

    module = SimpleNamespace(
        features=features, DirectionContextModel=SimpleNamespace(load=load_learned)
    )
    monkeypatch.setattr(comparison, "TCNBackend", lambda: SimpleNamespace(load_bundle=load_control))
    monkeypatch.setattr(
        comparison,
        "import_module",
        lambda name: {comparison.MODULES["context"]: module}[name],
    )
    names = (comparison.CONTROL, "context", "context+tcn")
    predict, _ = comparison.load_predictors(tmp_path, names, {comparison.CONTROL: tmp_path})
    result = predict(SimpleNamespace(observations=((1, 2), (3,))))
    assert tuple(result) == names
    np.testing.assert_allclose(result["context+tcn"], np.tile([0.2, 0.6, 0.2], (3, 1)))
    assert events == [
        "load control",
        "load context",
        "control predict",
        "features (1, 2)",
        "features (3,)",
    ]


def test_uncertain_gain_and_local_regressions_block_qualification(comparison: ModuleType) -> None:
    def scores(point: float, direction: float, long: float) -> dict[str, object]:
        return {
            "point_macro_f1": point,
            "interval_per_class": {"decline": {"f1": direction}},
            "by_length": {"1024": {"interval_macro_f1": long}},
        }

    results = {comparison.CONTROL: scores(0.80, 0.80, 0.80), "context": scores(0.77, 0.75, 0.76)}
    comparisons = {
        comparison.CONTROL: {"interval_macro_f1": {"interval_95pct": [0, 0.2]}},
        "release-native": {"interval_macro_f1": {"interval_95pct": [0.1, 0.2]}},
    }
    assert comparison.qualification("context", results, comparisons) == {
        "new_validation_winner": True,
        "gain_over_both_controls": False,
        "point_f1": False,
        "per_class_interval_f1": False,
        "per_length_interval_f1": False,
    }
    results["context"] = scores(0.79, 0.80, 0.79)
    comparisons[comparison.CONTROL]["interval_macro_f1"]["interval_95pct"] = [0.01, 0.02]
    assert all(comparison.qualification("context", results, comparisons).values())


def test_confirmation_verifies_choice_before_first_open_and_records_access(
    comparison: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(comparison, "checked_reservation", lambda _: {})
    monkeypatch.setattr(comparison, "check_fit", lambda _: {})
    (tmp_path / "fitted").mkdir()
    fit = tmp_path / "fitted/fit.json"
    write_json(fit, {})
    freeze = tmp_path / "freeze.json"
    scores = {name: {"interval_macro_f1": 0.6} for name in comparison.CANDIDATES}
    write_json(freeze, {"scores": scores, "winner": "adaptive", "fit_sha256": digest(fit)})
    calls = []

    def open_pool(output: Path, receipt: object, pool: str) -> None:
        assert (output / "opened.json").is_file()
        calls.append(pool)
        raise RuntimeError("stop before reading the holdout")

    monkeypatch.setattr(comparison, "open_pool", open_pool)
    with pytest.raises(ValueError, match="validation scores"):
        comparison.score(tmp_path, {})
    assert not (tmp_path / "opened.json").exists()
    write_json(freeze, {"scores": scores, "winner": comparison.CONTROL, "fit_sha256": digest(fit)})
    with pytest.raises(RuntimeError, match="stop before reading"):
        comparison.score(tmp_path, {})
    with pytest.raises(FileExistsError):
        comparison.score(tmp_path, {})
    assert calls == ["holdout"]


def test_probability_scores_exclude_unavailable_rows(comparison: ModuleType) -> None:
    np = pytest.importorskip("numpy")
    p = np.array([[0.8, 0.1, 0.1], [0.0, 0.0, 1.0]])
    assert comparison.probability_metrics(np.array([0, -1]), p) == pytest.approx(
        {"rows": 1, "log_loss": -np.log(0.8), "brier": 0.06}
    )
    assert comparison.probability_metrics(np.array([-1, -1]), p) == {
        "rows": 0,
        "log_loss": None,
        "brier": None,
    }


def test_frozen_weights_checked_before_scoring(
    comparison: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    folder = tmp_path / "fitted"
    folder.mkdir()
    write_json(tmp_path / "reservation.json", {})
    hashes = {}
    for name in comparison.MODULES:
        weight = folder / (name + ".npz")
        weight.write_bytes(name.encode())
        hashes[name] = digest(weight)
    monkeypatch.setattr(comparison, "software_fingerprint", lambda: {})
    write_json(
        folder / "fit.json",
        {
            "software": {},
            "reservation_sha256": digest(tmp_path / "reservation.json"),
            "weights": hashes,
        },
    )
    assert comparison.check_fit(tmp_path)["weights"] == hashes
    (folder / "adaptive.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Learned model"):
        comparison.check_fit(tmp_path)


def test_freeze_reads_only_validation_and_is_single_use(
    comparison: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(comparison, "checked_reservation", lambda _: {})
    monkeypatch.setattr(comparison, "check_fit", lambda _: {})
    (tmp_path / "fitted").mkdir()
    write_json(tmp_path / "fitted/fit.json", {})
    calls: list[str] = []
    scores = {name: {"interval_macro_f1": 0.5} for name in comparison.CANDIDATES}
    scores["context"] = {"interval_macro_f1": 0.6}

    def open_pool(root: Path, receipt: object, name: str) -> str:
        assert root == tmp_path
        calls.append(name)
        return name

    def measure(root: Path, evaluation: str, bundles: object, names: tuple[str, ...]) -> object:
        assert evaluation == "validation"
        assert set(comparison.CANDIDATES) <= set(names)
        return scores, {}

    monkeypatch.setattr(comparison, "open_pool", open_pool)
    monkeypatch.setattr(comparison, "measure", measure)
    comparison.freeze(tmp_path, {})
    assert comparison.read_freeze(tmp_path)["winner"] == "context"
    with pytest.raises(ValueError, match="already frozen"):
        comparison.freeze(tmp_path, {})
    assert calls == ["validation"]
