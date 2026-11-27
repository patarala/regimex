import json
from pathlib import Path

import pytest
from pydantic import JsonValue

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")


def test_workspace_fit_records_epochs_and_restores_identical_probabilities(tmp_path: Path) -> None:
    from test_tcn_backend import inputs, policy, provenance

    from regimex.backends.segment_moe import SegmentMoEBackend, SegmentTable
    from regimex.backends.segment_moe_features import SegmentFeatureVersion
    from regimex.backends.segment_moe_network import SegmentMoE
    from regimex.directions import AnalysisConfig
    from regimex.model_contracts import TrainingInputs
    from regimex.model_registry import (
        InferenceRuntime,
        SegmentMoEParameters,
        TrainingRuntime,
        configured_parameter_count,
        segment_moe_descriptor,
    )
    from regimex.resource_limits import ResourceBudgetExceeded
    from regimex.storage import digest, write_json

    train, validation = inputs("train", (64, 64, 64)), inputs("validation", (90, 80, 64))
    targets = np.repeat(np.arange(3), 64).astype(np.int64)
    targets[::13] = -1
    params = SegmentMoEParameters(hidden_width=12)
    assert configured_parameter_count("segment-moe", params.model_dump(mode="json")) == sum(
        p.numel() for p in SegmentMoE(12).parameters()
    )
    descriptor = segment_moe_descriptor()
    descriptor.validate_training_policy(policy())
    backend = SegmentMoEBackend()
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

    manifest_path = tmp_path / "bundle" / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    config_path = manifest_path.parent / "configuration.json"
    config = json.loads(config_path.read_bytes())
    assert manifest["architecture_id"] == "segment-moe"
    assert manifest["feature_version"] == "segment-moe/1.3"
    assert manifest["preprocessing_version"] == "segment-moe-block-standardization/1.0"
    assert config["codec_version"] == "segment-moe/1.0"

    # Historical bundles use the same flat weights but select their own feature extractor.
    manifest["architecture_id"] = "persistence"
    manifest["preprocessing_version"] = "persistence-block-standardization/1.0"
    config["codec_version"] = "direction-persistence/1.0"
    write_json(config_path, config)
    config_member = next(
        member for member in manifest["members"] if member["role"] == "configuration"
    )
    config_member.update(size_bytes=config_path.stat().st_size, sha256=digest(config_path))
    for version in SegmentFeatureVersion:
        manifest["feature_version"] = version.value.replace(
            "segment-moe/", "direction-persistence/"
        )
        write_json(manifest_path, manifest)
        historical_checksum = digest(manifest_path)
        restored_legacy = backend.load_bundle(
            manifest_path.parent, manifest_sha256=historical_checksum, runtime=runtime
        )
        assert digest(manifest_path) == historical_checksum
        assert digest(config_path) == config_member["sha256"]
        assert restored_legacy.architecture_id == "segment-moe"
        assert restored_legacy.feature_version == version
        legacy_input = SegmentTable.prepare(validation, reloaded.analysis, feature_version=version)
        expected = legacy_input.standardized(reloaded.scale).predict(
            reloaded.model, runtime.max_batch_size
        )
        if version == SegmentFeatureVersion.OBSERVED_PRICES:
            assert np.any(raw != expected)
        np.testing.assert_array_equal(
            restored_legacy.predict(validation, runtime=runtime), expected
        )
    manifest["feature_version"] = "direction-persistence/unknown"
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="feature layout"):
        backend.load_bundle(
            manifest_path.parent, manifest_sha256=digest(manifest_path), runtime=runtime
        )


@pytest.mark.parametrize("batch_size", [1, 2])
def test_segment_weight_affects_updates_even_in_a_singleton_batch(batch_size: int) -> None:
    from copy import deepcopy

    from torch import Tensor
    from torch.nn.utils import parameters_to_vector

    from regimex.backends.segment_moe import TrainingSegments, _train_epoch
    from regimex.backends.segment_moe_network import SegmentMoE

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(7)
        initial = SegmentMoE(8)
        contexts = torch.rand(3, 9, 4)
        blocks = torch.rand(3, 28)
    last = np.random.default_rng(17).permutation(3)[-1]

    def epoch(last_weight: float) -> Tensor:
        model = deepcopy(initial)
        weights = torch.ones(3)
        weights[last] = last_weight
        data = TrainingSegments(contexts, blocks, torch.eye(3), weights)
        _train_epoch(
            model,
            torch.optim.SGD(model.parameters(), lr=0.01),
            data,
            np.random.default_rng(17),
            batch_size,
            lambda: None,
        )
        return parameters_to_vector(model.parameters())

    assert not torch.equal(epoch(1.0), epoch(4.0))


def test_context_network_reflects_direction_and_reports_bounded_state() -> None:
    from regimex.backends.segment_moe_network import SegmentMoE
    from regimex.model_registry import SegmentMoEParameters, segment_moe_descriptor

    descriptor = segment_moe_descriptor()
    assert descriptor.required_observation_fields == ("last_price",)
    with pytest.raises(ValueError):
        SegmentMoEParameters(hidden_width=129)
    rng = np.random.default_rng(13)
    x = torch.tensor(rng.normal(size=(12, 9, 4)), dtype=torch.float32)
    x[:, :, 1::2] = x[:, :, 1::2].sigmoid()
    block = torch.tensor(rng.normal(size=(12, 28)), dtype=torch.float32)
    mirrored = x.clone()
    mirrored[:, :, [0, 2]] *= -1
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        model = SegmentMoE(8)
        torch.testing.assert_close(model(x, block), model(mirrored, block)[:, [0, 2, 1]])


def test_v1_flat_weights_keep_their_original_tensor_order(tmp_path: Path) -> None:
    from regimex.backends.segment_moe import _load_weights
    from regimex.backends.segment_moe_network import SegmentMoE

    # Version 1 stores a stream of little-endian float32s without parameter names.
    # The expected probabilities come from the original 4-unit implementation.
    path = tmp_path / "weights.f32"
    values = 0.2 * np.sin(0.03 * np.arange(1_017, dtype=np.float64))
    path.write_bytes(values.astype("<f4").tobytes())
    model = SegmentMoE(4)
    _load_weights(model, path)
    contexts = torch.arange(3 * 9 * 4, dtype=torch.float32).reshape(3, 9, 4) / 80 - 0.6
    blocks = torch.arange(3 * 28, dtype=torch.float32).reshape(3, 28) / 50 - 0.4
    expected = torch.tensor(
        [
            [0.59678513, 0.20436732, 0.19884755],
            [0.67201078, 0.16466509, 0.16332415],
            [0.40740439, 0.44255340, 0.15004221],
        ]
    )
    with torch.inference_mode():
        torch.testing.assert_close(model(contexts, blocks), expected, rtol=1e-5, atol=1e-7)


def test_forward_and_block_scaling_preserve_the_input_device() -> None:
    from regimex.backends.segment_moe import BlockScale
    from regimex.backends.segment_moe_network import SegmentMoE

    # Meta tensors have no host storage. A NumPy conversion in either path would fail.
    with torch.device("meta"):
        model = SegmentMoE(8)
        contexts = torch.empty(2, 9, 4)
        blocks = torch.empty(2, 28)
    scale = BlockScale(mean=(0.0,) * 28, scale=(1.0,) * 28)
    with torch.inference_mode():
        scaled = scale.transform(blocks)
        scores = model(contexts, scaled)
    assert scaled.device == scores.device == blocks.device
    assert scores.shape == (2, 3)
