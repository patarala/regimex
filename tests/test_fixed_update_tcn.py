"""Fixed budgets and native update artifacts preserve selection and restoration."""

import json
from pathlib import Path
from typing import cast

import pytest
from pydantic import RootModel

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from test_application_training import prepared_data
from test_tcn_application import registry, request
from test_tcn_backend import inputs, provenance, rehash

from regimex.admin_api import TrainingRunRequest
from regimex.admin_contracts import AdminSpec, TrainingSpec
from regimex.application.training import observation_inputs, train_candidate
from regimex.backends.tcn import FixedUpdateTCNConfiguration, TCNBackend, TCNPredictor
from regimex.directions import AnalysisConfig
from regimex.features import DoubleArray
from regimex.model_contracts import FitResult, ObservationInputs, TrainingInputs
from regimex.model_registry import (
    FixedUpdatePolicy,
    InferenceRuntime,
    RegistryError,
    TCNParameters,
    TrainingPolicy,
    TrainingRuntime,
    baseline_descriptor,
    tcn_descriptor,
)
from regimex.service_settings import ServiceSettings
from regimex.storage import write_json
from regimex.temporal import fit_tcn, predict_tcn
from regimex.temporal_updates import fit_tcn_updates


def fitted_updates() -> tuple[TCNBackend, FitResult, ObservationInputs]:
    backend = TCNBackend()
    training, validation = inputs("train"), inputs("validation")
    targets = np.repeat(np.arange(3), (24, 32, 28)).astype(np.int64)
    targets[::11] = -1
    result = backend.fit(
        TrainingInputs(training, targets),
        validation,
        parameters=TCNParameters(channels=4, dilations=(1, 2)),
        policy=FixedUpdatePolicy(
            seeds=(11, 23), updates=4, validation_updates=(2, 4), batch_size=2
        ),
        analysis=AnalysisConfig(min_duration=4),
        runtime=TrainingRuntime(threads=1),
        evaluator=lambda probabilities: float(probabilities[:, 0].mean()),
    )
    return backend, result, validation


@pytest.mark.parametrize(
    "extra",
    [
        {"epochs": 30},
        {"patience": 5},
        {"validation_interval": 2},
        {"optimizer": "lbfgs"},
        {"updates": "4"},
        {"validation_updates": [2, 2, 4]},
        {"validation_updates": [4, 2]},
        {"validation_updates": [2]},
        {"seeds": [11, 11]},
        {"updates": 16_001, "validation_updates": [16_001], "seeds": [11, 23]},
    ],
)
def test_fixed_policy_rejects_ineffective_fields_and_invalid_schedules(
    extra: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        FixedUpdatePolicy.model_validate_json(
            json.dumps({"updates": 4, "validation_updates": [2, 4], **extra})
        )


def test_backend_capability_is_explicit_and_defaults_remain_epochs() -> None:
    policy = FixedUpdatePolicy(updates=4, validation_updates=(2, 4))
    tcn_descriptor().validate_training_policy(policy)
    assert isinstance(tcn_descriptor().default_training_policy, TrainingPolicy)
    assert tcn_descriptor().describe().supported_training_budgets == ("epochs", "updates")
    with pytest.raises(RegistryError, match="training budgets"):
        baseline_descriptor().validate_training_policy(policy)
    descriptor = tcn_descriptor().model_copy(update={"supported_training_budgets": ("epochs",)})
    with pytest.raises(RegistryError, match="training budgets"):
        descriptor.validate_training_policy(policy)


def test_fixed_policy_candidate_uses_shared_training_and_exact_bundle_restore(
    tmp_path: Path,
) -> None:
    data, _ = prepared_data()
    original = request()
    policy = FixedUpdatePolicy(seeds=(11,), updates=4, validation_updates=(2, 4), batch_size=1)
    intake = original.model_copy(update={"policy": policy})
    result = train_candidate(
        data, intake, tmp_path / "bundle", registry=registry(), worker_id="fixture"
    )
    restored = TCNBackend().load_bundle(
        tmp_path / "bundle",
        manifest_sha256=result.manifest_sha256,
        runtime=original.verification_runtime,
    )
    assert isinstance(restored.configuration, FixedUpdateTCNConfiguration)
    assert restored.configuration.codec_version == "temporal-tcn/3.0"
    assert restored.implementation_version == "0.2.0"
    assert restored.predict(
        observation_inputs(data.validation), runtime=original.verification_runtime
    ).shape == (192, 3)


def test_update_artifact_roundtrip_retains_exact_numeric_state(tmp_path: Path) -> None:
    backend, result, validation = fitted_updates()
    sha = backend.write_bundle(
        result,
        tmp_path / "bundle",
        provenance=provenance(result),
        analysis=result.predictor.analysis,
    )
    restored = backend.load_bundle(
        tmp_path / "bundle", manifest_sha256=sha, runtime=InferenceRuntime(threads=1)
    )
    np.testing.assert_array_equal(
        restored.predict(validation, runtime=InferenceRuntime(threads=1)),
        result.predictor.predict(validation, runtime=InferenceRuntime(threads=1)),
    )
    history = json.loads((tmp_path / "bundle/history.json").read_bytes())
    assert "epochs" not in history and len(history["updates"]) == 8
    assert [(row["seed"], row["update"]) for row in history["updates"]] == [
        (seed, step) for seed in (11, 23) for step in range(1, 5)
    ]
    assert set(result.selected_settings) == {"update"}


@pytest.mark.parametrize("damage", ["omission", "schedule", "batch", "selected_update"])
def test_rehashed_inconsistent_update_history_cannot_load(tmp_path: Path, damage: str) -> None:
    backend, result, _ = fitted_updates()
    root = tmp_path / "bundle"
    backend.write_bundle(
        result, root, provenance=provenance(result), analysis=result.predictor.analysis
    )
    if damage == "selected_update":
        value = json.loads((root / "validation.json").read_bytes())
        value["selected_update"] = 3
        write_json(root / "validation.json", value)
    else:
        value = json.loads((root / "history.json").read_bytes())
        if damage == "omission":
            value["updates"].pop()
        elif damage == "schedule":
            value["updates"][0]["validation_segment_macro_f1"] = 0.5
        else:
            value["updates"][0]["batch_in_pass"] = 100
        write_json(root / "history.json", value)
    with pytest.raises(ValueError):
        backend.load_bundle(root, manifest_sha256=rehash(root), runtime=InferenceRuntime(threads=1))


def test_one_complete_pass_matches_existing_epoch_optimizer_exactly() -> None:
    _, fitted, _ = fitted_updates()
    predictor = cast(TCNPredictor, fitted.predictor)
    from regimex.backends.tcn import _prepared

    prepared = _prepared(inputs("synthetic"), predictor.preprocessor)
    targets = np.repeat(np.arange(3), (24, 32, 28)).astype(np.int64)
    parameters = TCNParameters(channels=4, dilations=(1, 2))
    runtime = TrainingRuntime(threads=1)
    old, _, _ = fit_tcn(
        prepared,
        targets,
        prepared,
        parameters=parameters,
        policy=TrainingPolicy(optimizer="adamw", seeds=(11,), epochs=1, batch_size=3),
        runtime=runtime,
        scorer=lambda _probabilities: 0.5,
    )
    new, _, _ = fit_tcn_updates(
        prepared,
        targets,
        prepared,
        parameters=parameters,
        policy=FixedUpdatePolicy(seeds=(11,), updates=1, validation_updates=(1,), batch_size=3),
        runtime=runtime,
        scorer=lambda _probabilities: 0.5,
    )
    for name, tensor in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[name], tensor, rtol=0, atol=0)
    np.testing.assert_array_equal(predict_tcn(new, prepared), predict_tcn(old, prepared))


def test_updates_ignore_unlabelled_batches_and_restore_first_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, fitted, _ = fitted_updates()
    predictor = cast(TCNPredictor, fitted.predictor)
    from regimex.backends.tcn import _prepared

    prepared = _prepared(inputs("synthetic", lengths=(8, 9, 10, 11)), predictor.preprocessor)
    targets = np.repeat(np.asarray([-1, 0, 1, 2]), (8, 9, 10, 11)).astype(np.int64)
    observed, calls = [], []
    original = torch.optim.AdamW.step

    def step(optimizer: object, *args: object, **kwargs: object) -> object:
        calls.append(1)
        return original(optimizer, *args, **kwargs)

    def score(probabilities: DoubleArray) -> float:
        observed.append(probabilities.copy())
        return 0.5

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    model, history, seed = fit_tcn_updates(
        prepared,
        targets,
        prepared,
        parameters=TCNParameters(channels=4, dilations=(1,)),
        policy=FixedUpdatePolicy(
            seeds=(11, 23), updates=4, validation_updates=(2, 4), batch_size=1
        ),
        runtime=TrainingRuntime(threads=1),
        scorer=score,
    )
    assert len(calls) == len(history) == 8 and len(observed) == 4 and seed == 11
    np.testing.assert_array_equal(predict_tcn(model, prepared), observed[0])


def test_service_exposes_an_explicit_update_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REGIMEX_DATABASE_DSN_FILE", raising=False)
    monkeypatch.setenv("REGIMEX_DATABASE_DSN", "postgresql://fixture")
    monkeypatch.setenv("REGIMEX_MAX_TRAINING_UPDATES", "1200")
    assert ServiceSettings.from_env().max_training_updates == 1200


@pytest.mark.parametrize("fixed", [False, True])
def test_training_intake_and_queue_roundtrip_keep_budget_type(fixed: bool) -> None:
    policy = (
        FixedUpdatePolicy(seeds=(11,), updates=4, validation_updates=(2, 4))
        if fixed
        else TrainingPolicy(optimizer="adamw", seeds=(11,))
    )
    body = TrainingRunRequest(
        dataset_id="development",
        labels_reference_set_id="training-labels",
        validation_reference_set_id="validation-references",
        architecture_id="tcn",
        parameters={},
        policy=policy,
        benchmark_selection_id="reserved",
    )
    parsed = TrainingRunRequest.model_validate_json(body.model_dump_json())
    spec = TrainingSpec.model_validate(
        {**parsed.model_dump(), "model_version_id": "candidate", "build_id": "source-fixture"}
    )
    restored = RootModel[AdminSpec].model_validate_json(spec.model_dump_json()).root
    assert isinstance(restored, TrainingSpec)
    assert type(restored.policy) is type(policy)
    assert restored == spec


@pytest.mark.parametrize("score", [float("nan"), -0.1, 1.1])
def test_fixed_budget_rejects_invalid_checkpoint_score(score: float) -> None:
    training, validation = inputs("train"), inputs("validation")
    with pytest.raises(ValueError, match="validation score"):
        TCNBackend().fit(
            TrainingInputs(training, np.repeat(np.arange(3), (24, 32, 28)).astype(np.int64)),
            validation,
            parameters=TCNParameters(channels=4, dilations=(1,)),
            policy=FixedUpdatePolicy(seeds=(11,), updates=1, validation_updates=(1,)),
            analysis=AnalysisConfig(min_duration=4),
            runtime=TrainingRuntime(threads=1),
            evaluator=lambda _probabilities: score,
        )
