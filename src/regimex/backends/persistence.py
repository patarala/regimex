"""Train and restore a direction-persistence mixture from workspace observations."""

import copy
import hashlib
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
import torch
from pydantic import Field, JsonValue
from torch import Tensor, nn

from regimex.backends.epoch_history import HistoryRecord, ValidationRecord, validate_epoch_selection
from regimex.backends.price_persistence import Accumulation, Gate, geometry
from regimex.contracts import StrictModel
from regimex.directions import DIRECTION_NAMES, AnalysisConfig
from regimex.evaluation import probabilities_valid
from regimex.features import DoubleArray, IntArray
from regimex.model_bundles import BundleMember, BundleRole, ModelBundle, verify_model_bundle
from regimex.model_contracts import (
    BundleProvenance,
    FitResult,
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
FEATURES = "direction-persistence/1.0"
PREPROCESSING = "persistence-block-standardization/1.0"
_ROLES: dict[BundleRole, str] = {
    "configuration": "configuration.json",
    "history": "history.json",
    "weights": "weights.f32",
    "preprocessing": "preprocessing.json",
    "validation": "validation.json",
}
_BLOCK_WIDTH = 28
type _Moments = Annotated[
    tuple[float, ...], Field(min_length=_BLOCK_WIDTH, max_length=_BLOCK_WIDTH)
]


class BlockScale(StrictModel):
    mean: _Moments
    scale: Annotated[
        tuple[Annotated[float, Field(gt=0)], ...],
        Field(min_length=_BLOCK_WIDTH, max_length=_BLOCK_WIDTH),
    ]

    @classmethod
    def fit(cls, blocks: Tensor) -> BlockScale:
        values = blocks.numpy().astype(np.float64)
        return cls(mean=tuple(values.mean(0)), scale=tuple(values.std(0).clip(min=1e-4)))

    def transform(self, blocks: Tensor) -> Tensor:
        z = (blocks.numpy() - np.asarray(self.mean)) / np.asarray(self.scale)
        return torch.from_numpy(z.clip(-7, 7).astype(np.float32))


class Configuration(StrictModel):
    codec_version: Literal["direction-persistence/1.0"] = "direction-persistence/1.0"
    parameters: PersistenceParameters
    training_policy: TrainingPolicy
    training_runtime: TrainingRuntime
    analysis: AnalysisConfig
    environment: dict[str, JsonValue]


class PersistenceNetwork(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.accumulation = Accumulation()
        self.context = nn.Sequential(nn.Linear(36, width), nn.SiLU(), nn.Linear(width, 3))
        self.gate = Gate()

    @property
    def weight_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in self.state_dict().values())

    def check_cache_budget(self, runtime: InferenceRuntime) -> None:
        if self.weight_bytes + _BLOCK_WIDTH * 16 > runtime.model_cache_budget_bytes:
            raise ResourceBudgetExceeded(
                "persistence model exceeds the inference model cache budget"
            )

    def forward(self, segments: Tensor, block: Tensor) -> Tensor:
        slope, fit, simple, simple_fit = segments.unbind(-1)
        signed = slope.asinh()
        simple = simple.asinh()
        direct = torch.stack((signed, fit, simple, simple_fit), dim=-1).flatten(1)
        reflected = torch.stack((-signed, fit, -simple, simple_fit), dim=-1).flatten(1)
        # Quote-unit and sign conventions should not make one direction easier to predict.
        context = ((self.context(direct) + self.context(reflected)[:, (0, 2, 1)]) * 0.5).softmax(-1)
        weight = self.gate(block).sigmoid()
        return cast(Tensor, weight * self.accumulation(segments) + (1 - weight) * context)


@dataclass
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
    ) -> SegmentTable:
        require_observation_inputs(inputs)
        spans, contexts, blocks = [], [], []
        offset = 0
        for rows in inputs.observations:
            if check_budget:
                check_budget()
            lengths, context, block = geometry(tuple(rows), config)
            spans.append(lengths + offset)
            contexts.append(context)
            blocks.append(block)
            offset += len(rows)
        return cls(
            np.concatenate(spans),
            torch.from_numpy(np.concatenate(contexts)),
            torch.from_numpy(np.concatenate(blocks)),
            offset,
        )

    @torch.inference_mode()
    def predict(self, network: PersistenceNetwork, batch: int) -> DoubleArray:
        # The decoder determines availability. Uniform scores keep every original step aligned.
        result = np.full((self.row_count, 3), 1 / 3, dtype=np.float64)
        for start in range(0, len(self.spans), batch):
            stop = start + batch
            probabilities = (
                network(self.contexts[start:stop], self.blocks[start:stop]).double().numpy()
            )
            for (left, right), prob in zip(self.spans[start:stop], probabilities, strict=True):
                result[left:right] = prob
        probabilities_valid(result, self.row_count)
        return result


@dataclass(frozen=True)
class PersistencePredictor:
    model: PersistenceNetwork
    scale: BlockScale
    configuration: Configuration
    architecture_id = ARCHITECTURE
    implementation_version = VERSION
    class_names = DIRECTION_NAMES

    @property
    def analysis(self) -> AnalysisConfig:
        return self.configuration.analysis

    def predict(self, inputs: ObservationInputs, *, runtime: InferenceRuntime) -> DoubleArray:
        InferenceRuntime.model_validate(runtime.model_dump())
        self.model.check_cache_budget(runtime)
        torch.set_num_threads(runtime.threads)
        table = SegmentTable.prepare(inputs, self.analysis)
        table.blocks = self.scale.transform(table.blocks)
        return table.predict(self.model, runtime.max_batch_size)


def _read_json[T: StrictModel](path: Path, kind: type[T], limit: int = 128 * 1024) -> T:
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("persistence JSON member exceeds its codec size limit")
    return kind.model_validate_json(data)


def _hash(document: dict[str, JsonValue]) -> str:
    return hashlib.sha256(json_bytes(document)).hexdigest()


class PersistenceBackend:
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
            raise TypeError(
                "Direction persistence requires TrainingInputs and PersistenceParameters"
            )
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
            raise ResourceBudgetExceeded("persistence inputs exceed the training memory budget")
        deadline = time.monotonic() + runtime.time_budget_seconds

        def check_budget() -> None:
            if time.monotonic() >= deadline:
                raise ResourceBudgetExceeded("persistence exceeded the training time budget")

        train = SegmentTable.prepare(training.observations, analysis, check_budget)
        valid = SegmentTable.prepare(validation, analysis, check_budget)
        if not len(train.spans):
            raise ValueError("no observable training segments")
        labels = training.targets
        cumulative = np.vstack(
            (
                np.zeros(3, dtype=np.int64),
                np.column_stack([labels == k for k in range(3)]).cumsum(0),
            )
        )
        counts = cumulative[train.spans[:, 1]] - cumulative[train.spans[:, 0]]
        labeled = counts.sum(1)
        keep = np.flatnonzero(labeled)
        if not counts.sum(axis=0).all():
            raise ValueError(
                "direction persistence requires training segments for all three directions"
            )
        target = torch.tensor(counts[keep] / labeled[keep, None], dtype=torch.float32)
        # Square-root weighting keeps long segments informative without drowning out short regimes.
        weight = torch.tensor(np.sqrt(labeled[keep]), dtype=torch.float32)
        scale = BlockScale.fit(train.blocks)
        context = train.contexts[keep]
        blocks = scale.transform(train.blocks)[keep]
        valid.blocks = scale.transform(valid.blocks)
        torch.set_num_threads(runtime.threads)
        history: list[dict[str, JsonValue]] = []
        best_score, best_seed, best_epoch = -1.0, 0, 0
        best: PersistenceNetwork | None = None
        with torch.random.fork_rng(devices=[]):
            for seed in policy.seeds:
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                model = PersistenceNetwork(parameters.hidden_width)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=policy.learning_rate, weight_decay=policy.weight_decay
                )
                seed_best, stale = -1.0, 0
                for epoch in range(1, policy.epochs + 1):
                    order = rng.permutation(len(keep))
                    combined_loss = 0.0
                    for start in range(0, len(order), policy.batch_size):
                        check_budget()
                        batch = order[start : start + policy.batch_size]
                        p = model(context[batch], blocks[batch])
                        loss_sum = (
                            -(target[batch] * p.clamp_min(1e-7).log()).sum(1).dot(weight[batch])
                        )
                        loss = loss_sum / weight[batch].sum()
                        optimizer.zero_grad(set_to_none=True)
                        torch.autograd.backward(loss)
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                        optimizer.step()
                        combined_loss += float(loss_sum.detach())
                    row: dict[str, JsonValue] = dict(
                        seed=seed, epoch=epoch, training_loss=combined_loss / float(weight.sum())
                    )
                    if (
                        epoch == 1
                        or epoch % policy.validation_interval == 0
                        or epoch == policy.epochs
                    ):
                        check_budget()
                        score = evaluator(valid.predict(model, 128))
                        if not np.isfinite(score) or not 0 <= score <= 1:
                            raise ValueError("validation score must be interval macro-F1 in [0, 1]")
                        row["validation_segment_macro_f1"] = score
                        if score > best_score:
                            best, best_score, best_seed, best_epoch = (
                                copy.deepcopy(model),
                                score,
                                seed,
                                epoch,
                            )
                        if score > seed_best:
                            seed_best, stale = score, 0
                        else:
                            stale += 1
                    history.append(row)
                    reason = (
                        "epoch_limit"
                        if epoch == policy.epochs
                        else "early_stopping"
                        if stale >= policy.patience
                        else None
                    )
                    if training.on_history:
                        training.on_history(row | {"stop_reason": reason} if reason else row.copy())
                    if reason:
                        break
        check_budget()
        assert best is not None
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
        predictor = PersistencePredictor(best.eval().requires_grad_(False), scale, config)
        return FitResult(
            predictor, tuple(history), best_seed, best_score, {"epoch": best_epoch}, environment
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
        if not isinstance(predictor, PersistencePredictor):
            raise TypeError("persistence bundles require a fitted persistence predictor")
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
            raise ValueError("persistence bundle provenance differs from its fitted configuration")
        validation = ValidationRecord(
            selected_seed=result.selected_seed,
            score=result.validation_score,
            selected_epoch=cast(int, result.selected_settings["epoch"]),
        )
        history = HistoryRecord.model_validate({"epochs": result.history})
        validate_epoch_selection(
            "Direction persistence", config.training_policy, validation, history
        )
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
                        raise ValueError("persistence model contains nonfinite weights")
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
                feature_version=FEATURES,
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
    ) -> PersistencePredictor:
        InferenceRuntime.model_validate(runtime.model_dump())
        bundle = verify_model_bundle(
            root,
            manifest_sha256=manifest_sha256,
            architecture_id=ARCHITECTURE,
            implementation_version=VERSION,
        )
        if (
            {m.role: m.path for m in bundle.members} != _ROLES
            or bundle.feature_version != FEATURES
            or bundle.preprocessing_version != PREPROCESSING
            or bundle.decoder_version != DECODER_VERSION
        ):
            raise ValueError("incompatible persistence bundle codec or feature layout")
        config = _read_json(root / _ROLES["configuration"], Configuration)
        scale = _read_json(root / _ROLES["preprocessing"], BlockScale)
        validation = _read_json(root / _ROLES["validation"], ValidationRecord)
        history = _read_json(root / _ROLES["history"], HistoryRecord, 16 << 20)
        validate_persistence_policy(config.training_policy)
        validate_epoch_selection(
            "Direction persistence", config.training_policy, validation, history
        )
        if (
            _hash(config.analysis.model_dump(mode="json")) != bundle.analysis_profile_sha256
            or _hash(config.environment) != bundle.environment_sha256
            or bundle.training.seeds != config.training_policy.seeds
            or bundle.training.selected_seed != validation.selected_seed
            or bundle.training.validation_segment_macro_f1 != validation.score
        ):
            raise ValueError("persistence configuration differs from manifest provenance")
        with torch.random.fork_rng(devices=[]):
            model = PersistenceNetwork(config.parameters.hidden_width)
        state = model.state_dict()
        model.check_cache_budget(runtime)
        size = model.weight_bytes
        with (root / _ROLES["weights"]).open("rb") as stream:
            data = stream.read(size + 1)
        if len(data) != size:
            raise ValueError("persistence weights do not match the configured layout")
        values = np.frombuffer(data, dtype="<f4")
        if not np.isfinite(values).all():
            raise ValueError("persistence weights contain nonfinite values")
        first = 0
        for name, template in sorted(state.items()):
            stop = first + template.numel()
            state[name] = torch.from_numpy(values[first:stop].copy().reshape(tuple(template.shape)))
            first = stop
        model.load_state_dict(state)
        return PersistencePredictor(model.eval().requires_grad_(False), scale, config)


def create_backend() -> PersistenceBackend:
    return PersistenceBackend()


def probe_runtime(device: str) -> None:
    if device != "cpu":
        raise ValueError("Direction persistence supports CPU execution only")
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        p = PersistenceNetwork(4)(torch.zeros(1, 9, 4), torch.zeros(1, _BLOCK_WIDTH))
        probabilities_valid(p.double().numpy(), 1)
