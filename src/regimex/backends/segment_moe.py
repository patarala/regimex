"""Train and restore Segment MoE from workspace observations."""

import copy
import hashlib
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
import torch
from pydantic import Field, JsonValue
from torch import Tensor, nn

from regimex.backends.epoch_history import HistoryRecord, ValidationRecord, validate_epoch_selection
from regimex.backends.segment_moe_features import SegmentFeatureVersion, segment_features
from regimex.backends.segment_moe_layout import (
    BLOCK_FEATURE_COUNT,
    CONTEXT_LENGTH,
    SEGMENT_FEATURE_COUNT,
)
from regimex.backends.segment_moe_network import SegmentMoE
from regimex.contracts import StrictModel
from regimex.directions import DIRECTION_NAMES, AnalysisConfig
from regimex.evaluation import probabilities_valid
from regimex.features import DoubleArray, IntArray
from regimex.model_bundles import (
    BundleMember,
    BundleRole,
    ModelBundle,
    read_bundle_json,
    verify_model_bundle,
)
from regimex.model_contracts import (
    BundleProvenance,
    FitResult,
    HistoryCallback,
    ObservationInputs,
    SelectionEvaluator,
    TrainingInputs,
    require_observation_inputs,
)
from regimex.model_registry import (
    InferenceRuntime,
    PersistenceParameters,
    TrainingPolicy,
    TrainingPolicySpec,
    TrainingRuntime,
    validate_persistence_policy,
)
from regimex.resource_limits import ResourceBudgetExceeded
from regimex.segmentation import DECODER_VERSION
from regimex.storage import digest, json_bytes, output_directory, write_json

ARCHITECTURE = "persistence"
VERSION = "0.1.0"
PREPROCESSING = "persistence-block-standardization/1.0"
_ROLES: dict[BundleRole, str] = {
    "configuration": "configuration.json",
    "history": "history.json",
    "weights": "weights.f32",
    "preprocessing": "preprocessing.json",
    "validation": "validation.json",
}
type _Moments = Annotated[
    tuple[float, ...], Field(min_length=BLOCK_FEATURE_COUNT, max_length=BLOCK_FEATURE_COUNT)
]


class BlockScale(StrictModel):
    mean: _Moments
    scale: Annotated[
        tuple[Annotated[float, Field(gt=0)], ...],
        Field(min_length=BLOCK_FEATURE_COUNT, max_length=BLOCK_FEATURE_COUNT),
    ]

    @classmethod
    def fit(cls, blocks: Tensor) -> BlockScale:
        values = blocks.to(dtype=torch.float64)
        return cls(
            mean=tuple(values.mean(dim=0).tolist()),
            scale=tuple(values.std(dim=0, correction=0).clamp_min(1e-4).tolist()),
        )

    def transform(self, blocks: Tensor) -> Tensor:
        mean = blocks.new_tensor(self.mean, dtype=torch.float64)
        scale = blocks.new_tensor(self.scale, dtype=torch.float64)
        return ((blocks - mean) / scale).clamp(-7, 7).to(dtype=blocks.dtype)


class Configuration(StrictModel):
    codec_version: Literal["direction-persistence/1.0"] = "direction-persistence/1.0"
    parameters: PersistenceParameters
    training_policy: TrainingPolicy
    training_runtime: TrainingRuntime
    analysis: AnalysisConfig
    environment: dict[str, JsonValue]


@dataclass(frozen=True)
class SegmentTable:
    spans: IntArray
    contexts: Tensor
    blocks: Tensor
    row_count: int

    @classmethod
    def prepare(
        cls,
        inputs: ObservationInputs,
        config: AnalysisConfig,
        check_budget: Callable[[], None] | None = None,
        *,
        feature_version: SegmentFeatureVersion = SegmentFeatureVersion.ORIGINAL_STEPS,
    ) -> SegmentTable:
        require_observation_inputs(inputs)
        spans, contexts, blocks = [], [], []
        offset = 0
        for rows in inputs.observations:
            if check_budget:
                check_budget()
            sequence_spans, context, block = segment_features(
                tuple(rows), config, version=feature_version
            )
            spans.append(sequence_spans + offset)
            contexts.append(context)
            blocks.append(block)
            offset += len(rows)
        return cls(
            np.concatenate(spans),
            torch.from_numpy(np.concatenate(contexts)),
            torch.from_numpy(np.concatenate(blocks)),
            offset,
        )

    def standardized(self, scale: BlockScale) -> SegmentTable:
        return replace(self, blocks=scale.transform(self.blocks))

    @torch.inference_mode()
    def predict(self, network: SegmentMoE, batch_size: int) -> DoubleArray:
        # The decoder determines availability. Uniform scores keep every original step aligned.
        result = np.full((self.row_count, 3), 1 / 3, dtype=np.float64)
        device = next(network.parameters()).device
        for start in range(0, len(self.spans), batch_size):
            stop = start + batch_size
            contexts = self.contexts[start:stop].to(device)
            blocks = self.blocks[start:stop].to(device)
            probabilities = network(contexts, blocks).to(device="cpu", dtype=torch.float64).numpy()
            for (left, right), prob in zip(self.spans[start:stop], probabilities, strict=True):
                result[left:right] = prob
        probabilities_valid(result, self.row_count)
        return result


@dataclass(frozen=True)
class SegmentMoEPredictor:
    model: SegmentMoE
    scale: BlockScale
    configuration: Configuration
    feature_version: SegmentFeatureVersion = SegmentFeatureVersion.ORIGINAL_STEPS
    architecture_id = ARCHITECTURE
    implementation_version = VERSION
    class_names = DIRECTION_NAMES

    @property
    def analysis(self) -> AnalysisConfig:
        return self.configuration.analysis

    def predict(self, inputs: ObservationInputs, *, runtime: InferenceRuntime) -> DoubleArray:
        InferenceRuntime.model_validate(runtime.model_dump())
        _check_cache_budget(self.model, runtime)
        torch.set_num_threads(runtime.threads)
        table = SegmentTable.prepare(inputs, self.analysis, feature_version=self.feature_version)
        return table.standardized(self.scale).predict(self.model, runtime.max_batch_size)


def _hash(document: dict[str, JsonValue]) -> str:
    return hashlib.sha256(json_bytes(document)).hexdigest()


def _weight_bytes(model: SegmentMoE) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values())


def _check_cache_budget(model: SegmentMoE, runtime: InferenceRuntime) -> None:
    # The bundle also stores two float64 moments per block feature.
    if _weight_bytes(model) + 2 * 8 * BLOCK_FEATURE_COUNT > runtime.model_cache_budget_bytes:
        raise ResourceBudgetExceeded("Segment MoE exceeds the inference model cache budget")


def _load_weights(model: SegmentMoE, path: Path) -> None:
    state = model.state_dict()
    size = _weight_bytes(model)
    with path.open("rb") as stream:
        data = stream.read(size + 1)
    if len(data) != size:
        raise ValueError("Segment MoE weights do not match the configured layout")
    values = np.frombuffer(data, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("Segment MoE weights contain nonfinite values")
    first = 0
    for name, template in sorted(state.items()):
        stop = first + template.numel()
        state[name] = torch.from_numpy(values[first:stop].astype(np.float32)).reshape(
            template.shape
        )
        first = stop
    model.load_state_dict(state)


@dataclass(frozen=True)
class TrainingSegments:
    contexts: Tensor
    blocks: Tensor
    targets: Tensor
    weights: Tensor

    @classmethod
    def from_table(cls, table: SegmentTable, labels: IntArray) -> TrainingSegments:
        # Prefix differences count known labels in each variable-length segment.
        observed = labels[:, None] == np.arange(len(DIRECTION_NAMES))
        prefix = np.zeros((len(labels) + 1, len(DIRECTION_NAMES)), dtype=np.int64)
        np.cumsum(observed, axis=0, out=prefix[1:])
        starts, stops = table.spans.T
        counts = prefix[stops] - prefix[starts]
        if not counts.sum(axis=0).all():
            raise ValueError("Segment MoE requires training segments for all three directions")
        labeled_steps = counts.sum(axis=1)
        keep = labeled_steps > 0
        distribution = counts[keep] / labeled_steps[keep, None]
        # Square-root weighting keeps long segments informative without drowning out short regimes.
        return cls(
            table.contexts[keep],
            table.blocks[keep],
            torch.tensor(distribution, dtype=torch.float32),
            torch.tensor(np.sqrt(labeled_steps[keep]), dtype=torch.float32),
        )


def _train_epoch(
    model: SegmentMoE,
    optimizer: torch.optim.Optimizer,
    segments: TrainingSegments,
    rng: np.random.Generator,
    batch_size: int,
    check_budget: Callable[[], None],
) -> float:
    device = next(model.parameters()).device
    order = torch.from_numpy(rng.permutation(len(segments.weights)))
    total_loss = 0.0
    for start in range(0, len(order), batch_size):
        check_budget()
        batch = order[start : start + batch_size]
        predictions = model(segments.contexts[batch].to(device), segments.blocks[batch].to(device))
        targets = segments.targets[batch].to(device)
        weights = segments.weights[batch].to(device)
        cross_entropy = -(targets * predictions.clamp_min(1e-7).log()).sum(dim=1)
        weighted_loss = cross_entropy.dot(weights)
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(weighted_loss / weights.sum())
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        total_loss += float(weighted_loss.detach())
    return total_loss / float(segments.weights.sum())


@dataclass(frozen=True)
class _Checkpoint:
    model: SegmentMoE
    seed: int
    epoch: int
    score: float


def _select_checkpoint(
    segments: TrainingSegments,
    validation: SegmentTable,
    *,
    width: int,
    device: str,
    policy: TrainingPolicy,
    evaluator: SelectionEvaluator,
    on_history: HistoryCallback | None,
    check_budget: Callable[[], None],
) -> tuple[list[dict[str, JsonValue]], _Checkpoint]:
    history: list[dict[str, JsonValue]] = []
    best: _Checkpoint | None = None
    with torch.random.fork_rng(devices=[]):
        for seed in policy.seeds:
            torch.default_generator.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = SegmentMoE(width).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=policy.learning_rate, weight_decay=policy.weight_decay
            )
            repeat_best, stale = -1.0, 0
            for epoch in range(1, policy.epochs + 1):
                model.train()
                loss = _train_epoch(
                    model, optimizer, segments, rng, policy.batch_size, check_budget
                )
                row: dict[str, JsonValue] = {"seed": seed, "epoch": epoch, "training_loss": loss}
                if epoch in (1, policy.epochs) or epoch % policy.validation_interval == 0:
                    check_budget()
                    model.eval()
                    score = evaluator(validation.predict(model, 128))
                    if not 0 <= score <= 1:
                        raise ValueError("validation score must be interval macro-F1 in [0, 1]")
                    row["validation_segment_macro_f1"] = score
                    if best is None or score > best.score:
                        best = _Checkpoint(copy.deepcopy(model), seed, epoch, score)
                    if score > repeat_best:
                        repeat_best, stale = score, 0
                    else:
                        stale += 1
                history.append(row)
                stop_reason = None
                if epoch == policy.epochs:
                    stop_reason = "epoch_limit"
                elif stale >= policy.patience:
                    stop_reason = "early_stopping"
                if on_history:
                    on_history(row | {"stop_reason": stop_reason} if stop_reason else row.copy())
                if stop_reason:
                    break
    # Every seed is evaluated at its first and last epoch.
    assert best is not None
    return history, best


class SegmentMoEBackend:
    architecture_id = ARCHITECTURE
    implementation_version = VERSION

    def fit(
        self,
        training: TrainingInputs,
        validation: ObservationInputs,
        *,
        parameters: StrictModel,
        policy: TrainingPolicySpec,
        analysis: AnalysisConfig,
        runtime: TrainingRuntime,
        evaluator: SelectionEvaluator,
    ) -> FitResult:
        if type(training) is not TrainingInputs or type(parameters) is not PersistenceParameters:
            raise TypeError("Segment MoE requires TrainingInputs and PersistenceParameters")
        require_observation_inputs(validation)
        parameters = PersistenceParameters.model_validate(parameters.model_dump())
        validate_persistence_policy(policy)
        assert isinstance(policy, TrainingPolicy)
        TrainingPolicy.model_validate(policy.model_dump())
        TrainingRuntime.model_validate(runtime.model_dump())
        AnalysisConfig.model_validate(analysis.model_dump())
        if {item.experiment_id for item in training.observations.info} & {
            item.experiment_id for item in validation.info
        }:
            raise ValueError("training and validation experiments must be disjoint")
        total = training.observations.row_count + validation.row_count
        if 2048 * total + (4 << 20) > runtime.memory_budget_bytes:
            raise ResourceBudgetExceeded("Segment MoE inputs exceed the training memory budget")
        deadline = time.monotonic() + runtime.time_budget_seconds

        def check_budget() -> None:
            if time.monotonic() >= deadline:
                raise ResourceBudgetExceeded("Segment MoE exceeded the training time budget")

        torch.set_num_threads(runtime.threads)
        train = SegmentTable.prepare(training.observations, analysis, check_budget)
        valid = SegmentTable.prepare(validation, analysis, check_budget)
        if not len(train.spans):
            raise ValueError("no observable training segments")
        scale = BlockScale.fit(train.blocks)
        segments = TrainingSegments.from_table(train.standardized(scale), training.targets)
        valid = valid.standardized(scale)
        history, checkpoint = _select_checkpoint(
            segments,
            valid,
            width=parameters.hidden_width,
            device=runtime.device,
            policy=policy,
            evaluator=evaluator,
            on_history=training.on_history,
            check_budget=check_budget,
        )
        check_budget()
        environment: dict[str, JsonValue] = {
            "python": platform.python_version(),
            "packages": {name: version(name) for name in ("regimex", "numpy", "torch")},
        }
        config = Configuration(
            parameters=parameters,
            training_policy=policy,
            training_runtime=runtime,
            analysis=analysis,
            environment=environment,
        )
        predictor = SegmentMoEPredictor(checkpoint.model.requires_grad_(False), scale, config)
        return FitResult(
            predictor,
            tuple(history),
            checkpoint.seed,
            checkpoint.score,
            {"epoch": checkpoint.epoch},
            environment,
        )

    def write_bundle(
        self,
        result: FitResult,
        root: Path,
        *,
        provenance: BundleProvenance,
        analysis: AnalysisConfig,
    ) -> str:
        predictor = result.predictor
        if not isinstance(predictor, SegmentMoEPredictor):
            raise TypeError("Segment MoE bundles require a fitted Segment MoE predictor")
        config = predictor.configuration
        if (
            analysis != config.analysis
            or _hash(analysis.model_dump(mode="json")) != provenance.analysis_profile_sha256
            or result.environment != config.environment
            or _hash(config.environment) != provenance.environment_sha256
            or provenance.training.seeds != config.training_policy.seeds
            or result.selected_seed != provenance.training.selected_seed
            or result.validation_score != provenance.training.validation_segment_macro_f1
            or set(result.selected_settings) != {"epoch"}
        ):
            raise ValueError("Segment MoE bundle provenance differs from its fitted configuration")
        validation = ValidationRecord(
            selected_seed=result.selected_seed,
            score=result.validation_score,
            selected_epoch=cast(int, result.selected_settings["epoch"]),
        )
        history = HistoryRecord.model_validate({"epochs": result.history})
        validate_epoch_selection("Segment MoE", config.training_policy, validation, history)
        with output_directory(root) as out:
            for role, doc in [
                ("configuration", config),
                ("preprocessing", predictor.scale),
                ("validation", validation),
                ("history", history),
            ]:
                write_json(out / _ROLES[cast(BundleRole, role)], doc.model_dump(mode="json"))
            with (out / _ROLES["weights"]).open("wb") as stream:
                for _, tensor in sorted(predictor.model.state_dict().items()):
                    values = tensor.detach().cpu().numpy().astype("<f4", copy=False)
                    if not np.isfinite(values).all():
                        raise ValueError("Segment MoE contains nonfinite weights")
                    stream.write(values.tobytes())
            members = tuple(
                BundleMember(
                    path=path,
                    role=role,
                    size_bytes=(out / path).stat().st_size,
                    sha256=digest(out / path),
                )
                for role, path in _ROLES.items()
            )
            bundle = ModelBundle(
                bundle_version="1.0",
                architecture_id=ARCHITECTURE,
                implementation_version=VERSION,
                interface_version="1.0",
                observation_version="2.1",
                feature_version=predictor.feature_version.value,
                preprocessing_version=PREPROCESSING,
                decoder_version=DECODER_VERSION,
                class_names=("sideways", "growth", "decline"),
                members=members,
                **provenance.model_dump(),
            )
            write_json(out / "manifest.json", bundle.model_dump(mode="json"))
            checksum = digest(out / "manifest.json")
        return checksum

    def load_bundle(
        self, root: Path, *, manifest_sha256: str, runtime: InferenceRuntime
    ) -> SegmentMoEPredictor:
        InferenceRuntime.model_validate(runtime.model_dump())
        bundle = verify_model_bundle(
            root,
            manifest_sha256=manifest_sha256,
            architecture_id=ARCHITECTURE,
            implementation_version=VERSION,
        )
        if (
            {m.role: m.path for m in bundle.members} != _ROLES
            or bundle.feature_version not in SegmentFeatureVersion
            or bundle.preprocessing_version != PREPROCESSING
            or bundle.decoder_version != DECODER_VERSION
        ):
            raise ValueError("incompatible Segment MoE bundle codec or feature layout")
        config = read_bundle_json(root / _ROLES["configuration"], Configuration, limit=128 << 10)
        scale = read_bundle_json(root / _ROLES["preprocessing"], BlockScale, limit=128 << 10)
        validation = read_bundle_json(
            root / _ROLES["validation"], ValidationRecord, limit=128 << 10
        )
        history = read_bundle_json(root / _ROLES["history"], HistoryRecord, limit=16 << 20)
        validate_persistence_policy(config.training_policy)
        validate_epoch_selection("Segment MoE", config.training_policy, validation, history)
        if (
            _hash(config.analysis.model_dump(mode="json")) != bundle.analysis_profile_sha256
            or _hash(config.environment) != bundle.environment_sha256
            or bundle.training.seeds != config.training_policy.seeds
            or bundle.training.selected_seed != validation.selected_seed
            or bundle.training.validation_segment_macro_f1 != validation.score
        ):
            raise ValueError("Segment MoE configuration differs from manifest provenance")
        with torch.random.fork_rng(devices=[]):
            model = SegmentMoE(config.parameters.hidden_width)
        _check_cache_budget(model, runtime)
        _load_weights(model, root / _ROLES["weights"])
        model.to(runtime.device)
        return SegmentMoEPredictor(
            model.eval().requires_grad_(False),
            scale,
            config,
            SegmentFeatureVersion(bundle.feature_version),
        )


def create_backend() -> SegmentMoEBackend:
    return SegmentMoEBackend()


def probe_runtime(device: str) -> None:
    if device != "cpu":
        raise ValueError("Segment MoE supports CPU execution only")
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        p = SegmentMoE(4)(
            torch.zeros(1, CONTEXT_LENGTH, SEGMENT_FEATURE_COUNT),
            torch.zeros(1, BLOCK_FEATURE_COUNT),
        )
        probabilities_valid(p.double().numpy(), 1)
