from pathlib import Path

import pytest
from pydantic import JsonValue

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")


def test_workspace_fit_records_epochs_and_restores_identical_probabilities(tmp_path: Path) -> None:
    from test_tcn_backend import inputs, policy, provenance

    from regimex.backends.persistence import PersistenceBackend, PersistenceNetwork, SegmentTable
    from regimex.directions import AnalysisConfig
    from regimex.model_contracts import TrainingInputs
    from regimex.model_registry import (
        InferenceRuntime,
        PersistenceParameters,
        TrainingRuntime,
        configured_parameter_count,
        persistence_descriptor,
    )
    from regimex.resource_limits import ResourceBudgetExceeded

    train, validation = inputs("train", (64, 64, 64)), inputs("validation", (90, 80, 64))
    targets = np.repeat(np.arange(3), 64).astype(np.int64)
    targets[::13] = -1
    params = PersistenceParameters(hidden_width=12)
    assert configured_parameter_count("persistence", params.model_dump(mode="json")) == sum(
        p.numel() for p in PersistenceNetwork(12).parameters()
    )
    descriptor = persistence_descriptor()
    descriptor.validate_training_policy(policy())
    backend = PersistenceBackend()
    progress: list[dict[str, JsonValue]] = []
    settings = policy().model_copy(update={"patience": 1})
    fitted = backend.fit(
        TrainingInputs(train, targets, on_history=progress.append),
        validation,
        parameters=params,
        analysis=AnalysisConfig(min_duration=4),
        policy=settings,
        runtime=TrainingRuntime(threads=1),
        evaluator=lambda _: 0.6,
    )
    assert len(progress) == 4 and progress[1]["stop_reason"] == "early_stopping"
    assert [r["seed"] for r in fitted.history] == [11, 11, 23, 23]
    assert fitted.selected_seed == 11 and fitted.selected_settings == {"epoch": 1}
    assert all(isinstance(r["training_loss"], float) for r in fitted.history)
    preprocessor = SegmentTable.prepare(train, fitted.predictor.analysis)
    runtime = InferenceRuntime(threads=1)
    checksum = backend.write_bundle(
        fitted,
        tmp_path / "bundle",
        provenance=provenance(fitted),
        analysis=fitted.predictor.analysis,
    )
    reloaded = backend.load_bundle(tmp_path / "bundle", manifest_sha256=checksum, runtime=runtime)
    np.testing.assert_array_equal(
        reloaded.scale.mean, preprocessor.blocks.numpy().astype(np.float64).mean(0)
    )
    raw = reloaded.predict(validation, runtime=runtime)
    assert raw.shape == (validation.row_count, 3)
    np.testing.assert_array_equal(raw, fitted.predictor.predict(validation, runtime=runtime))
    np.testing.assert_allclose(raw.sum(1), 1, atol=1e-6)
    small_cache = InferenceRuntime(threads=1, model_cache_budget_bytes=32)
    with pytest.raises(ResourceBudgetExceeded, match="model cache budget"):
        reloaded.predict(validation, runtime=small_cache)
    with pytest.raises(ResourceBudgetExceeded, match="model cache budget"):
        backend.load_bundle(tmp_path / "bundle", manifest_sha256=checksum, runtime=small_cache)
    with pytest.raises(ValueError, match="disjoint"):
        backend.fit(
            TrainingInputs(train, targets),
            train,
            parameters=params,
            analysis=fitted.predictor.analysis,
            policy=settings,
            runtime=TrainingRuntime(threads=1),
            evaluator=lambda _: 0.1,
        )


def test_context_network_reflects_direction_and_reports_bounded_state() -> None:
    from regimex.backends.persistence import PersistenceNetwork
    from regimex.model_registry import PersistenceParameters, persistence_descriptor

    descriptor = persistence_descriptor()
    assert descriptor.required_observation_fields == ("last_price",)
    with pytest.raises(ValueError):
        PersistenceParameters(hidden_width=129)
    rng = np.random.default_rng(13)
    x = torch.tensor(rng.normal(size=(12, 9, 4)), dtype=torch.float32)
    x[:, :, 1::2] = x[:, :, 1::2].sigmoid()
    block = torch.tensor(rng.normal(size=(12, 28)), dtype=torch.float32)
    mirrored = x.clone()
    mirrored[:, :, [0, 2]] *= -1
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        model = PersistenceNetwork(8)
        torch.testing.assert_close(model(x, block), model(mirrored, block)[:, [0, 2, 1]])
