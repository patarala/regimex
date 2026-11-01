"""A separately implemented backend uses the unchanged candidate-training workflow.

The fitted class-prior model is a behavior fixture, never a quality benchmark.
"""

import json
import platform
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Literal, cast

import pytest

np = pytest.importorskip("numpy")
# The shared fixture module also verifies the aggregate backend when run alone.
pytest.importorskip("sklearn")

from pydantic import Field, JsonValue
from test_application_training import prepared_data, request

from regimex.application.training import observation_inputs, train_candidate
from regimex.contracts import StrictModel
from regimex.directions import DIRECTION_NAMES, AnalysisConfig
from regimex.features import DoubleArray
from regimex.model_bundles import BundleMember, BundleRole, ModelBundle, verify_model_bundle
from regimex.model_contracts import (
    BundleProvenance,
    FitResult,
    LoadedPredictor,
    ObservationInputs,
    SelectionEvaluator,
    TrainingInputs,
)
from regimex.model_registry import (
    ArchitectureDescriptor,
    ArchitectureRegistry,
    InferenceRuntime,
    TrainingPolicy,
    TrainingRuntime,
)
from regimex.segmentation import DECODER_VERSION
from regimex.storage import digest, write_json

ARCHITECTURE = "fixture-class-prior"
IMPLEMENTATION = "1.0.0"
EVENTS: list[str] = []
FAULT: str | None = None


class PriorParameters(StrictModel):
    pseudocount: float = Field(default=0.5, gt=0)


class PriorConfiguration(StrictModel):
    version: Literal["1.0"]
    parameters: PriorParameters
    policy: TrainingPolicy
    runtime: TrainingRuntime
    analysis: AnalysisConfig
    environment: dict[str, JsonValue]


class PriorWeights(StrictModel):
    version: Literal["1.0"]
    class_names: tuple[Literal["sideways"], Literal["growth"], Literal["decline"]]
    probabilities: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class PriorPredictor:
    probabilities: DoubleArray
    analysis: AnalysisConfig
    configuration: PriorConfiguration
    fault: str | None = None
    architecture_id: str = ARCHITECTURE
    implementation_version: str = IMPLEMENTATION
    class_names: tuple[str, ...] = DIRECTION_NAMES

    def predict(self, inputs: ObservationInputs, *, runtime: InferenceRuntime) -> DoubleArray:
        assert type(inputs) is ObservationInputs
        assert {field.name for field in fields(inputs)} == {
            "info",
            "observations",
            "time_grids",
            "instruments",
        }
        assert all(not hasattr(item, "seed") for item in inputs.info)
        assert not hasattr(inputs, "targets") and not hasattr(inputs, "references")
        InferenceRuntime.model_validate_json(runtime.model_dump_json())
        values = np.tile(self.probabilities, (inputs.row_count, 1))
        EVENTS.append("predict")
        if self.fault == "lost-timestamp":
            return cast(DoubleArray, values[:-1])
        if self.fault == "nonfinite":
            values[0, 0] = np.nan
        if self.fault == "unnormalized":
            values *= 2
        if self.fault == "wrong-dtype":
            return cast(DoubleArray, values.astype(np.float32))
        return cast(DoubleArray, values)


class PriorBackend:
    architecture_id = ARCHITECTURE
    implementation_version = IMPLEMENTATION

    def fit(
        self,
        training: TrainingInputs,
        validation: ObservationInputs,
        *,
        parameters: StrictModel,
        policy: TrainingPolicy,
        analysis: AnalysisConfig,
        runtime: TrainingRuntime,
        evaluator: SelectionEvaluator,
    ) -> FitResult:
        assert type(training) is TrainingInputs
        assert type(parameters) is PriorParameters
        assert type(validation) is ObservationInputs
        assert not hasattr(training, "test") and not hasattr(validation, "labels")
        if len(policy.seeds) != 1:
            raise ValueError("class-prior fixture uses one deterministic seed")
        selected = training.targets[training.targets >= 0]
        if not len(selected):
            raise ValueError("class-prior fixture requires observed training targets")
        counts = np.bincount(selected, minlength=len(DIRECTION_NAMES)).astype(np.float64)
        counts += parameters.pseudocount
        probabilities = counts / counts.sum()
        environment: dict[str, JsonValue] = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scope": "behavior-fixture",
        }
        configuration = PriorConfiguration(
            version="1.0",
            parameters=parameters,
            policy=policy,
            runtime=runtime,
            analysis=analysis,
            environment=environment,
        )
        predictor = PriorPredictor(probabilities, analysis, configuration)
        score = evaluator(predictor.predict(validation, runtime=InferenceRuntime(threads=1)))
        if FAULT == "class-order":
            predictor = replace(predictor, class_names=("growth", "sideways", "decline"))
        elif FAULT in {"lost-timestamp", "nonfinite", "unnormalized", "wrong-dtype"}:
            predictor = replace(predictor, fault=FAULT)
        EVENTS.append("fit")
        return FitResult(
            predictor=predictor,
            history=({"validation_segment_macro_f1": score},),
            selected_seed=policy.seeds[0],
            validation_score=score,
            selected_settings={"pseudocount": parameters.pseudocount},
            environment=environment,
        )

    def write_bundle(
        self,
        result: FitResult,
        root: Path,
        *,
        provenance: BundleProvenance,
        analysis: AnalysisConfig,
    ) -> str:
        assert isinstance(result.predictor, PriorPredictor)
        predictor = result.predictor
        assert analysis == predictor.analysis
        write_json(root / "configuration.json", predictor.configuration.model_dump(mode="json"))
        write_json(root / "preprocessing.json", {"version": "1.0", "kind": "no-feature-transform"})
        weights = PriorWeights(
            version="1.0",
            class_names=("sideways", "growth", "decline"),
            probabilities=cast(tuple[float, float, float], tuple(predictor.probabilities.tolist())),
        )
        write_json(root / "weights.json", weights.model_dump(mode="json"))
        write_json(root / "validation.json", {"score": result.validation_score})
        write_json(root / "history.json", list(result.history))
        members = tuple(
            BundleMember(
                path=role + ".json",
                role=role,
                size_bytes=(root / (role + ".json")).stat().st_size,
                sha256=digest(root / (role + ".json")),
            )
            for role in cast(
                tuple[BundleRole, ...],
                ("configuration", "preprocessing", "weights", "validation", "history"),
            )
        )
        manifest = ModelBundle(
            bundle_version="1.0",
            model_version_id=provenance.model_version_id,
            architecture_id=ARCHITECTURE,
            implementation_version=IMPLEMENTATION,
            interface_version="1.0",
            observation_version="2.1",
            feature_version="prior-counts/1.0",
            preprocessing_version="no-feature-transform/1.0",
            decoder_version=DECODER_VERSION,
            analysis_profile=provenance.analysis_profile,
            analysis_profile_sha256=provenance.analysis_profile_sha256,
            class_names=("sideways", "growth", "decline"),
            training=provenance.training,
            build_id=provenance.build_id,
            environment_sha256=provenance.environment_sha256,
            members=members,
        )
        if FAULT == "bundle-identity":
            manifest = manifest.model_copy(update={"model_version_id": "another-candidate"})
        write_json(root / "manifest.json", manifest.model_dump(mode="json"))
        EVENTS.append("write")
        return digest(root / "manifest.json")

    def load_bundle(
        self, root: Path, *, manifest_sha256: str, runtime: InferenceRuntime
    ) -> LoadedPredictor:
        InferenceRuntime.model_validate_json(runtime.model_dump_json())
        manifest = verify_model_bundle(
            root,
            manifest_sha256=manifest_sha256,
            architecture_id=ARCHITECTURE,
            implementation_version=IMPLEMENTATION,
        )
        if {member.path for member in manifest.members} != {
            name + ".json"
            for name in ("configuration", "preprocessing", "weights", "validation", "history")
        }:
            raise ValueError("fixture prior codec paths differ")
        configuration = PriorConfiguration.model_validate_json(
            (root / "configuration.json").read_bytes()
        )
        weights = PriorWeights.model_validate_json((root / "weights.json").read_bytes())
        preprocessing = json.loads((root / "preprocessing.json").read_text())
        if preprocessing != {"version": "1.0", "kind": "no-feature-transform"}:
            raise ValueError("fixture prior preprocessing codec differs")
        probabilities = np.asarray(weights.probabilities, dtype=np.float64)
        if (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1):
            raise ValueError("fixture prior weights must be normalized nonnegative probabilities")
        if FAULT == "reload-drift":
            probabilities = np.array([0.8, 0.1, 0.1], dtype=np.float64)
        EVENTS.append("load")
        return PriorPredictor(probabilities, configuration.analysis, configuration)


def create_prior_backend() -> PriorBackend:
    return PriorBackend()


def probe_prior_runtime(device: str) -> None:
    if device != "cpu":
        raise ValueError("class-prior fixture supports CPU only")
    np.testing.assert_array_equal(np.bincount(np.array([0, 1, 1, 2])), [1, 2, 1])
    EVENTS.append("probe")


def prior_descriptor() -> ArchitectureDescriptor:
    return ArchitectureDescriptor(
        architecture_id=ARCHITECTURE,
        implementation_version=IMPLEMENTATION,
        parameters_model=PriorParameters,
        default_training_policy=TrainingPolicy(optimizer="lbfgs", seeds=(11,)),
        supported_optimizers=("lbfgs",),
        required_observation_fields=("last_price",),
        dependency_modules=("numpy",),
        backend_factory=__name__ + ":create_prior_backend",
        worker_probe=__name__ + ":probe_prior_runtime",
    )


@pytest.fixture(autouse=True)
def reset_fixture_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    EVENTS.clear()
    monkeypatch.setattr(__import__(__name__), "FAULT", None)


def test_third_party_backend_trains_and_restores_through_unchanged_application(
    tmp_path: Path,
) -> None:
    # This public fixture loader raises immediately if observations or reference
    # payloads from the test/final partition are requested.
    data, dataset = prepared_data()
    configured = request().model_copy(
        update={"architecture_id": ARCHITECTURE, "parameters": {"pseudocount": 0.5}}
    )
    result = train_candidate(
        data,
        configured,
        tmp_path / "candidate",
        registry=ArchitectureRegistry((prior_descriptor(),)),
        worker_id="fixture-plugin-worker",
    )
    assert result.architecture_id == ARCHITECTURE
    assert result.dataset_sha256 == dataset.content_sha256
    assert result.quality_status == "validation_only"
    assert "test" not in type(data).model_fields
    assert EVENTS[0] == "probe"
    assert EVENTS.count("fit") == 1 and EVENTS.count("write") == 1 and EVENTS.count("load") == 1
    assert EVENTS.count("predict") == 3
    restored = PriorBackend().load_bundle(
        tmp_path / "candidate",
        manifest_sha256=result.manifest_sha256,
        runtime=configured.verification_runtime,
    )
    validation = observation_inputs(data.validation)
    probabilities = restored.predict(validation, runtime=configured.verification_runtime)
    assert probabilities.shape == (192, 3)
    np.testing.assert_array_equal(probabilities, np.full((192, 3), 1 / 3))
    manifest = verify_model_bundle(
        tmp_path / "candidate",
        manifest_sha256=result.manifest_sha256,
        architecture_id=ARCHITECTURE,
        implementation_version=IMPLEMENTATION,
    )
    assert manifest.feature_version == "prior-counts/1.0"
    assert manifest.training.labels.kind == "automatic"
    assert manifest.training.validation_references.kind == "human"


def test_fixture_backend_fits_unequal_class_priors_from_training_targets() -> None:
    data, _ = prepared_data()
    raw = observation_inputs(data.train)
    targets = np.repeat(np.arange(3), (24, 48, 120)).astype(np.int64)
    fitted = PriorBackend().fit(
        TrainingInputs(raw, targets),
        observation_inputs(data.validation),
        parameters=PriorParameters(pseudocount=0.5),
        policy=TrainingPolicy(optimizer="lbfgs", seeds=(11,)),
        analysis=data.analysis_profile,
        runtime=TrainingRuntime(threads=1),
        evaluator=lambda _: 0.0,
    )
    np.testing.assert_array_equal(
        cast(PriorPredictor, fitted.predictor).probabilities,
        (np.array([24, 48, 120]) + 0.5) / 193.5,
    )
    assert fitted.selected_settings == {"pseudocount": 0.5}


@pytest.mark.parametrize(
    "fault",
    [
        "class-order",
        "lost-timestamp",
        "nonfinite",
        "unnormalized",
        "wrong-dtype",
        "reload-drift",
        "bundle-identity",
    ],
)
def test_plugin_contract_failures_cannot_publish_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    monkeypatch.setattr(__import__(__name__), "FAULT", fault)
    data, _ = prepared_data()
    configured = request().model_copy(update={"architecture_id": ARCHITECTURE, "parameters": {}})
    with pytest.raises(ValueError):
        train_candidate(
            data,
            configured,
            tmp_path / "candidate",
            registry=ArchitectureRegistry((prior_descriptor(),)),
            worker_id="fixture-plugin-worker",
        )
    assert not (tmp_path / "candidate").exists()
    assert not list(tmp_path.glob(".candidate-*"))


def test_plugin_required_observation_capabilities_are_checked_before_fit(tmp_path: Path) -> None:
    data, _ = prepared_data()
    descriptor = prior_descriptor().model_copy(
        update={"required_observation_fields": ("best_bid",)}
    )
    configured = request().model_copy(update={"architecture_id": ARCHITECTURE, "parameters": {}})
    with pytest.raises(ValueError, match="best_bid"):
        train_candidate(
            data,
            configured,
            tmp_path / "candidate",
            registry=ArchitectureRegistry((descriptor,)),
            worker_id="fixture-plugin-worker",
        )
    assert "fit" not in EVENTS and "write" not in EVENTS
    assert not (tmp_path / "candidate").exists()


def test_reload_verification_rejects_drift_with_reused_prediction_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_predict = PriorPredictor.predict
    buffer: DoubleArray | None = None

    def predict_with_reused_buffer(
        self: PriorPredictor, inputs: ObservationInputs, *, runtime: InferenceRuntime
    ) -> DoubleArray:
        nonlocal buffer
        probabilities = original_predict(self, inputs, runtime=runtime)
        if buffer is None:
            buffer = probabilities
        else:
            np.copyto(buffer, probabilities)
        return buffer

    monkeypatch.setattr(PriorPredictor, "predict", predict_with_reused_buffer)
    monkeypatch.setattr(__import__(__name__), "FAULT", "reload-drift")
    data, _ = prepared_data()
    configured = request().model_copy(update={"architecture_id": ARCHITECTURE, "parameters": {}})
    with pytest.raises(ValueError, match="predictions changed after candidate bundle restoration"):
        train_candidate(
            data,
            configured,
            tmp_path / "candidate",
            registry=ArchitectureRegistry((prior_descriptor(),)),
            worker_id="fixture-plugin-worker",
        )
    assert not (tmp_path / "candidate").exists()
    assert not list(tmp_path.glob(".candidate-*"))
