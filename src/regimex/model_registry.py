"""Lightweight architecture discovery and explicit worker capability checks.

Descriptor discovery may run in the API process. Dependency imports and runtime
probes run only through the methods explicitly named ``on_worker``. A descriptor
describes an installed architecture; it does not establish worker availability.
"""

import importlib
import json
import os
import stat
from collections.abc import Callable, Collection, Iterable
from datetime import UTC, datetime
from enum import StrEnum
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal, Self, cast

from pydantic import ConfigDict, Field, JsonValue, ValidationError, model_validator

from regimex.backends import baseline_layout, tcn_layout
from regimex.contracts import MAX_STEPS, Seed, StrictModel

ENTRY_POINT_GROUP = "regimex.models"
INTERFACE_VERSION: Literal["1.0"] = "1.0"
Device = Literal["cpu"]
Optimizer = Literal["lbfgs", "adamw", "extra_trees"]
ObservationField = Literal[
    "last_price",
    "trade_count",
    "trade_volume",
    "buy_order_count",
    "buy_order_volume",
    "sell_order_count",
    "sell_order_volume",
    "best_bid",
    "best_ask",
]
ArchitectureId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
Version = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^\S+$")]
ModuleName = Annotated[str, Field(pattern=r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")]
FactoryReference = Annotated[
    str, Field(pattern=r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
]


class RegistryError(ValueError):
    """An installed architecture has an invalid or incompatible contract."""


class SummaryMode(StrEnum):
    LOCAL_AND_GLOBAL = "local_and_global"
    LOCAL = "local"
    LOCAL_AND_GLOBAL_OBSERVED = "local_and_global_observed"
    LOCAL_OBSERVED = "local_observed"

    @property
    def whole_sequence(self) -> bool:
        return self in (self.LOCAL_AND_GLOBAL, self.LOCAL_AND_GLOBAL_OBSERVED)

    @property
    def observed_only(self) -> bool:
        return self in (self.LOCAL_AND_GLOBAL_OBSERVED, self.LOCAL_OBSERVED)

    def width(self, channels: int, context_sizes: tuple[int, ...]) -> int:
        return 4 * channels * (len(context_sizes) + int(self.whole_sequence))


class LogisticParameters(StrictModel):
    """Regularization candidates preserve the existing aggregate baseline search."""

    regularization: Annotated[tuple[Annotated[float, Field(gt=0)], ...], Field(min_length=1)] = (
        0.1,
        1.0,
        10.0,
    )
    context_sizes: tuple[Annotated[int, Field(ge=2, le=MAX_STEPS)], ...] = (32, 128)

    summaries: SummaryMode = SummaryMode.LOCAL_AND_GLOBAL

    @model_validator(mode="after")
    def distinct_candidates(self) -> Self:
        if len(set(self.regularization)) != len(self.regularization):
            raise ValueError("regularization candidates must be unique")
        if not self.context_sizes and not self.summaries.whole_sequence:
            raise ValueError("local-only summaries require at least one context size")
        if len(set(self.context_sizes)) != len(self.context_sizes):
            raise ValueError("baseline context sizes must be unique")
        return self


TreeCount = Annotated[int, Field(ge=1, le=128)]
TreeDepth = Annotated[int, Field(ge=1, le=16)]


class AdaptiveParameters(StrictModel):
    feature_set: Literal["greedy", "multiscale"] = Field(
        default="greedy",
        exclude_if=lambda value: value == "greedy",
        title="Price features",
        description="Greedy splits use a fast recursive fit. Multiscale fits several "
        "segmentations jointly and estimates sustained slope across changes in price level.",
    )
    sample_weighting: Literal["per_step", "interval"] = Field(
        default="per_step",
        exclude_if=lambda value: value == "per_step",
        title="Training weights",
        description="Per step gives each labeled step equal weight. Interval weights give "
        "shorter labeled intervals more influence within their direction class.",
    )
    trees: TreeCount = Field(
        default=40,
        title="Number of trees",
        description="Each tree votes on the direction. More trees can stabilize probabilities "
        "and take longer to fit.",
    )
    max_depth: TreeDepth = Field(
        default=12,
        title="Maximum tree depth",
        description="Limits the number of splits along each path. Shallower trees group "
        "more observations together.",
    )


class PersistenceParameters(StrictModel):
    hidden_width: Annotated[int, Field(ge=4, le=128)] = Field(
        default=32,
        title="Context network width",
        description="Number of hidden units in the network that reads neighboring price segments.",
    )


class TCNParameters(StrictModel):
    side_slopes: bool = False
    channels: Annotated[int, Field(ge=4, le=256)] = 32
    dilations: Annotated[
        tuple[Annotated[int, Field(ge=1, le=MAX_STEPS)], ...], Field(min_length=1, max_length=32)
    ] = tcn_layout.DEFAULT_DILATIONS
    dropout: Annotated[float, Field(ge=0, lt=1)] = 0.15
    smoothing: Annotated[float, Field(ge=0)] = 0.05
    interval_weight_power: Annotated[float, Field(ge=0, le=1)] = 0.0

    @model_validator(mode="after")
    def distinct_dilations(self) -> Self:
        if len(set(self.dilations)) != len(self.dilations):
            raise ValueError("TCN dilations must be unique")
        return self


def configured_parameter_count(
    architecture_id: str, parameters: dict[str, JsonValue]
) -> int | None:
    """Count the coefficients or trainable tensors of a known training layout."""
    try:
        if architecture_id == "tcn":
            return tcn_layout.trainable_parameter_count(
                TCNParameters.model_validate_json(json.dumps(parameters))
            )
        if architecture_id == "persistence":
            width = PersistenceParameters.model_validate_json(json.dumps(parameters)).hidden_width
            # Accumulation (133), block mixing (721), and a 36 -> width -> 3 context head.
            return 133 + 721 + 40 * width + 3
        if architecture_id == "baseline":
            return baseline_layout.trainable_parameter_count(
                LogisticParameters.model_validate_json(json.dumps(parameters))
            )
    except ValidationError:
        # Historical runs may describe parameters from a different backend version.
        return None
    return None


class TrainingPolicy(StrictModel):
    optimizer: Literal["lbfgs", "adamw"]
    seeds: Annotated[tuple[Seed, ...], Field(min_length=1)] = (11, 23, 47)
    epochs: Annotated[int, Field(ge=1, le=1000)] = 30
    batch_size: Annotated[int, Field(ge=1, le=128)] = 16
    patience: Annotated[int, Field(ge=1)] = 5
    learning_rate: Annotated[float, Field(gt=0)] = 0.001
    weight_decay: Annotated[float, Field(ge=0)] = 0.0001
    validation_interval: Annotated[int, Field(ge=1)] = 2
    selection_metric: Literal["validation_segment_macro_f1"] = "validation_segment_macro_f1"

    @model_validator(mode="after")
    def distinct_seeds(self) -> Self:
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("training seeds must be unique")
        return self


class FixedUpdatePolicy(StrictModel):
    """Train every seed for an exact number of successful optimizer updates."""

    budget: Literal["updates"] = "updates"
    optimizer: Literal["adamw"] = "adamw"
    seeds: Annotated[tuple[Seed, ...], Field(min_length=1)] = (11, 23, 47)
    updates: Annotated[int, Field(ge=1, le=32_000)]
    validation_updates: Annotated[
        tuple[Annotated[int, Field(ge=1, le=32_000)], ...], Field(min_length=1, max_length=1000)
    ]
    batch_size: Annotated[int, Field(ge=1, le=128)] = 16
    learning_rate: Annotated[float, Field(gt=0)] = 0.001
    weight_decay: Annotated[float, Field(ge=0)] = 0.0001
    selection_metric: Literal["validation_segment_macro_f1"] = "validation_segment_macro_f1"

    @model_validator(mode="after")
    def coherent_budget(self) -> Self:
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("training seeds must be unique")
        if len(self.seeds) * self.updates > 32_000:
            raise ValueError("fixed-update history exceeds its 32000-record artifact budget")
        if (
            tuple(sorted(set(self.validation_updates))) != self.validation_updates
            or self.validation_updates[-1] != self.updates
        ):
            raise ValueError("validation updates must increase uniquely and end at the budget")
        return self


class SingleFitPolicy(StrictModel):
    """Score one fitted forest on the validation intervals."""

    budget: Literal["fit"] = "fit"
    optimizer: Literal["extra_trees"]
    seeds: tuple[Seed] = (173,)
    selection_metric: Literal["validation_segment_macro_f1"] = "validation_segment_macro_f1"


TrainingPolicySpec = TrainingPolicy | FixedUpdatePolicy | SingleFitPolicy
type TrainingBudget = Literal["epochs", "updates", "fit"]


BASELINE_DEFAULT_POLICY = TrainingPolicy(optimizer="lbfgs", seeds=(11,))


def resolve_training_policy(architecture_id: str, policy: TrainingPolicySpec) -> TrainingPolicySpec:
    """Apply native request defaults while preserving explicit choices and plugin policies."""
    if (
        architecture_id == "baseline"
        and isinstance(policy, TrainingPolicy)
        and "seeds" not in policy.model_fields_set
    ):
        return policy.model_copy(update={"seeds": BASELINE_DEFAULT_POLICY.seeds})
    return policy


def validate_baseline_policy(policy: TrainingPolicySpec) -> None:
    if not isinstance(policy, TrainingPolicy):
        raise RegistryError("baseline does not support fixed-update training")
    TrainingPolicy.model_validate(policy.model_dump())
    if policy.optimizer != "lbfgs":
        raise RegistryError("baseline supports only the lbfgs optimizer")
    if len(policy.seeds) != 1:
        raise RegistryError("deterministic baseline requires exactly one recorded seed")
    unused = (
        "epochs",
        "batch_size",
        "patience",
        "learning_rate",
        "weight_decay",
        "validation_interval",
    )
    changed = [
        name for name in unused if getattr(policy, name) != getattr(BASELINE_DEFAULT_POLICY, name)
    ]
    if changed:
        raise RegistryError("baseline does not use training policy fields: " + ", ".join(changed))


def validate_tcn_policy(policy: TrainingPolicySpec) -> None:
    if isinstance(policy, FixedUpdatePolicy):
        FixedUpdatePolicy.model_validate(policy.model_dump())
    elif isinstance(policy, TrainingPolicy):
        TrainingPolicy.model_validate(policy.model_dump())
    else:
        raise RegistryError("TCN supports only epoch or fixed-update training")
    if policy.optimizer != "adamw":
        raise RegistryError("TCN supports only the AdamW optimizer")
    if len(policy.seeds) > 32:
        raise RegistryError("TCN supports at most 32 search seeds")


class TrainingRuntime(StrictModel):
    device: Device = "cpu"
    threads: Annotated[int, Field(ge=1, le=256)] = 4
    time_budget_seconds: Annotated[int, Field(ge=1)] = 3600
    memory_budget_bytes: Annotated[int, Field(ge=1)] = 4 * 1024**3


class InferenceRuntime(StrictModel):
    device: Device = "cpu"
    threads: Annotated[int, Field(ge=1, le=256)] = 2
    max_batch_size: Annotated[int, Field(ge=1, le=4096)] = 16
    model_cache_budget_bytes: Annotated[int, Field(ge=1)] = 512 * 1024**2


_MAX_INFERENCE_RUNTIME_BYTES = 8 * 1024


def read_inference_runtime(path: Path) -> InferenceRuntime:
    """Read a bounded inference policy, following links to regular files."""
    # Opening a FIFO without O_NONBLOCK would wait for a writer before we can inspect it.
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Inference runtime JSON must be a regular file.")
        raw = stream.read(_MAX_INFERENCE_RUNTIME_BYTES + 1)
    if len(raw) > _MAX_INFERENCE_RUNTIME_BYTES:
        raise ValueError("Inference runtime JSON exceeds 8 KiB.")
    try:
        return InferenceRuntime.model_validate_json(raw)
    except ValidationError as exc:
        error = exc.errors(include_input=False, include_context=False)[0]
        field = ".".join(map(str, error["loc"])) or "JSON"
        raise ValueError(f"Invalid inference runtime ({field}): {error['msg']}") from None


class ModelSize(StrictModel):
    name: Annotated[str, Field(min_length=1)]
    parameters: dict[str, JsonValue]
    trainable_parameters: Annotated[int, Field(gt=0)]


class ArchitectureInfo(StrictModel):
    """Serializable installed metadata; worker availability may still be unknown."""

    descriptor_version: Literal["1.0"]
    architecture_id: ArchitectureId
    implementation_version: Version
    interface_version: Literal["1.0"]
    parameter_schema: dict[str, JsonValue]
    sizes: tuple[ModelSize, ...] = ()
    default_training_policy: TrainingPolicy | SingleFitPolicy
    max_training_seeds: Annotated[int, Field(ge=1)] | None = None
    max_training_epochs: Annotated[int, Field(ge=1)] | None = None
    max_training_updates: Annotated[int, Field(ge=1)] | None = None
    supported_optimizers: tuple[Optimizer, ...]
    supported_training_budgets: tuple[TrainingBudget, ...] = ("epochs",)
    required_observation_fields: tuple[ObservationField, ...]
    missing_value_policy: Literal["explicit_masks"]
    supported_devices: tuple[Device, ...]
    dependency_modules: tuple[ModuleName, ...]
    executable: bool | None
    availability_reason: str | None


class ArchitectureDescriptor(StrictModel):
    """Trusted Python descriptor; its factories must remain cheap to import.

    An executable adapter declares both a lazy backend factory and a worker
    runtime probe. The probe accepts the device, raises on unsupported execution,
    and returns None on success. It must exercise its real runtime dependencies.
    It must not train a model or alter process-global runtime settings.
    """

    model_config = ConfigDict(revalidate_instances="always")

    descriptor_version: Literal["1.0"] = "1.0"
    architecture_id: ArchitectureId
    implementation_version: Version
    interface_version: Literal["1.0"] = INTERFACE_VERSION
    parameters_model: type[StrictModel] = Field(exclude=True)
    policy_validator: Callable[[TrainingPolicySpec], None] | None = Field(
        default=None, exclude=True
    )
    trains_by_epoch: bool = True
    sizes: tuple[ModelSize, ...] = ()
    default_training_policy: TrainingPolicy | SingleFitPolicy
    supported_optimizers: Annotated[tuple[Optimizer, ...], Field(min_length=1)]
    supported_training_budgets: Annotated[tuple[TrainingBudget, ...], Field(min_length=1)] = (
        "epochs",
    )
    required_observation_fields: Annotated[tuple[ObservationField, ...], Field(min_length=1)]
    missing_value_policy: Literal["explicit_masks"] = "explicit_masks"
    supported_devices: Annotated[tuple[Device, ...], Field(min_length=1)] = ("cpu",)
    dependency_modules: tuple[ModuleName, ...] = ()
    backend_factory: FactoryReference | None = None
    worker_probe: FactoryReference | None = None
    unavailable_reason: Annotated[str, Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def coherent_descriptor(self) -> Self:
        if (
            self.parameters_model.model_config.get("extra") != "forbid"
            or self.parameters_model.model_config.get("strict") is not True
            or self.parameters_model.model_config.get("allow_inf_nan") is not False
        ):
            raise ValueError(
                "architecture parameter schemas must forbid extras and be strict/finite"
            )
        # Generate the schema during discovery, before serving an incomplete catalog.
        self.parameters_model.model_json_schema()
        for size in self.sizes:
            self.validate_parameters(size.parameters)
        for values in (
            self.supported_optimizers,
            self.supported_training_budgets,
            self.required_observation_fields,
            self.supported_devices,
            self.dependency_modules,
        ):
            if len(set(values)) != len(values):
                raise ValueError("architecture capability declarations must be unique")
        self.validate_training_policy(self.default_training_policy)
        if self.backend_factory is None:
            if self.worker_probe is not None or self.unavailable_reason is None:
                raise ValueError("an unavailable adapter needs a reason and cannot declare a probe")
        elif self.worker_probe is None or self.unavailable_reason is not None:
            raise ValueError("an adapter factory needs a worker probe and no unavailable reason")
        return self

    def validate_parameters(self, values: dict[str, JsonValue]) -> StrictModel:
        # JSON arrays are accepted as tuples without accepting coerced scalar types.
        return self.parameters_model.model_validate_json(json.dumps(values, allow_nan=False))

    def validate_training_policy(self, policy: TrainingPolicySpec) -> None:
        budget = "epochs" if isinstance(policy, TrainingPolicy) else policy.budget
        if budget not in self.supported_training_budgets:
            raise RegistryError(
                f"architecture {self.architecture_id!r} "
                f"does not support {budget!r} training budgets"
            )
        if policy.optimizer not in self.supported_optimizers:
            raise RegistryError(
                f"architecture {self.architecture_id!r} "
                f"does not support optimizer {policy.optimizer!r}"
            )

        if self.policy_validator is not None:
            self.policy_validator(policy)

    def validate_observation_fields(self, available_fields: Collection[str]) -> None:
        """Check recorded field capabilities, independently from per-row missingness."""
        missing = set(self.required_observation_fields).difference(available_fields)
        if missing:
            raise RegistryError(
                f"architecture {self.architecture_id!r} requires observation fields: "
                + ", ".join(sorted(missing))
            )

    def describe(
        self,
        *,
        max_training_seeds: int | None = None,
        max_training_epochs: int | None = None,
        max_training_updates: int | None = None,
    ) -> ArchitectureInfo:
        return ArchitectureInfo(
            descriptor_version=self.descriptor_version,
            architecture_id=self.architecture_id,
            implementation_version=self.implementation_version,
            interface_version=self.interface_version,
            parameter_schema=self.parameters_model.model_json_schema(),
            sizes=self.sizes,
            default_training_policy=self.default_training_policy,
            max_training_seeds=1
            if isinstance(self.default_training_policy, SingleFitPolicy)
            else max_training_seeds,
            max_training_epochs=max_training_epochs if self.trains_by_epoch else None,
            max_training_updates=max_training_updates
            if "updates" in self.supported_training_budgets
            else None,
            supported_optimizers=self.supported_optimizers,
            supported_training_budgets=self.supported_training_budgets,
            required_observation_fields=self.required_observation_fields,
            missing_value_policy=self.missing_value_policy,
            supported_devices=self.supported_devices,
            dependency_modules=self.dependency_modules,
            executable=None if self.backend_factory is not None else False,
            availability_reason=(
                "execution must be checked on the worker"
                if self.backend_factory is not None
                else self.unavailable_reason
            ),
        )


class WorkerCapability(StrictModel):
    architecture_id: ArchitectureId
    implementation_version: Version
    interface_version: Literal["1.0"] = INTERFACE_VERSION
    worker_id: Annotated[str, Field(min_length=1, max_length=128)]
    checked_at: datetime
    device: Device
    checked_dependency_modules: tuple[ModuleName, ...]
    executable: bool
    reasons: tuple[str, ...]

    @model_validator(mode="after")
    def coherent_result(self) -> Self:
        if self.executable == bool(self.reasons):
            raise ValueError("worker availability and its reasons disagree")
        if self.checked_at.tzinfo is None:
            raise ValueError("worker capability checks need an aware timestamp")
        return self


def _load_callable(reference: str) -> Callable[..., object]:
    module_name, attribute_path = reference.split(":", 1)
    value: object = importlib.import_module(module_name)
    for name in attribute_path.split("."):
        value = getattr(value, name)
    if not callable(value):
        raise RegistryError(f"declared factory or probe {reference!r} is not callable")
    return cast(Callable[..., object], value)


class ArchitectureRegistry:
    """Immutable startup catalog of trusted installed descriptor factories."""

    def __init__(self, descriptors: Iterable[ArchitectureDescriptor] = ()) -> None:
        entries: dict[str, ArchitectureDescriptor] = {}
        for descriptor in descriptors:
            if not isinstance(descriptor, ArchitectureDescriptor):
                raise RegistryError("an architecture factory must return ArchitectureDescriptor")
            try:
                descriptor = ArchitectureDescriptor.model_validate(descriptor)
            except ValueError as exc:
                raise RegistryError(f"invalid architecture descriptor: {exc}") from exc
            if descriptor.architecture_id in entries:
                raise RegistryError(f"duplicate architecture ID {descriptor.architecture_id!r}")
            entries[descriptor.architecture_id] = descriptor
        self._entries = MappingProxyType(entries)

    @classmethod
    def discover(cls) -> Self:
        descriptors: list[ArchitectureDescriptor] = []
        for entry_point in sorted(
            metadata.entry_points(group=ENTRY_POINT_GROUP), key=lambda point: point.name
        ):
            try:
                factory = entry_point.load()
                if not callable(factory):
                    raise RegistryError("descriptor factory is not callable")
                descriptor = factory()
                if not isinstance(descriptor, ArchitectureDescriptor):
                    raise RegistryError("factory did not return ArchitectureDescriptor")
            except Exception as exc:
                raise RegistryError(
                    f"invalid architecture entry point {entry_point.name!r}: {exc}"
                ) from exc
            descriptors.append(descriptor)
        return cls(descriptors)

    @property
    def descriptors(self) -> tuple[ArchitectureDescriptor, ...]:
        return tuple(self._entries[key] for key in sorted(self._entries))

    def get(self, architecture_id: str) -> ArchitectureDescriptor:
        try:
            return self._entries[architecture_id]
        except KeyError as exc:
            raise RegistryError(f"unknown installed architecture {architecture_id!r}") from exc

    def _check_backend_on_worker(
        self, architecture_id: str, runtime: TrainingRuntime | InferenceRuntime, *, worker_id: str
    ) -> tuple[WorkerCapability, Callable[[], object] | None, list[Exception]]:
        descriptor = self.get(architecture_id)
        checked: list[str] = []
        reasons: list[str] = []
        causes: list[Exception] = []
        factory = None
        if descriptor.backend_factory is None:
            assert descriptor.unavailable_reason is not None
            reasons.append(descriptor.unavailable_reason)
        else:
            for module_name in descriptor.dependency_modules:
                checked.append(module_name)
                try:
                    importlib.import_module(module_name)
                except Exception as exc:
                    reasons.append(f"dependency {module_name!r} cannot be imported: {exc}")
                    causes.append(exc)
            if not reasons:
                try:
                    factory = _load_callable(descriptor.backend_factory)
                    assert descriptor.worker_probe is not None
                    probe = _load_callable(descriptor.worker_probe)
                    if probe(runtime.device) is not None:
                        raise RegistryError("worker runtime probes must return None on success")
                except Exception as exc:
                    reasons.append(f"backend runtime check failed: {exc}")
                    causes.append(exc)
        capability = WorkerCapability(
            architecture_id=descriptor.architecture_id,
            implementation_version=descriptor.implementation_version,
            worker_id=worker_id,
            checked_at=datetime.now(UTC),
            device=runtime.device,
            checked_dependency_modules=tuple(checked),
            executable=not reasons,
            reasons=tuple(reasons),
        )
        return capability, factory, causes

    def check_capability_on_worker(
        self, architecture_id: str, runtime: TrainingRuntime | InferenceRuntime, *, worker_id: str
    ) -> WorkerCapability:
        capability, _, _ = self._check_backend_on_worker(
            architecture_id, runtime, worker_id=worker_id
        )
        return capability

    def load_backend_factory_on_worker(
        self, architecture_id: str, runtime: TrainingRuntime | InferenceRuntime, *, worker_id: str
    ) -> Callable[[], object]:
        capability, factory, causes = self._check_backend_on_worker(
            architecture_id, runtime, worker_id=worker_id
        )
        if not capability.executable:
            cause: Exception | None = None
            if causes:
                cause = (
                    causes[0] if len(causes) == 1 else ExceptionGroup("Dependency failures", causes)
                )
            raise RegistryError(
                f"architecture {architecture_id!r} is unavailable on worker {worker_id!r}: "
                + "; ".join(capability.reasons)
            ) from cause
        assert factory is not None
        return factory


def adaptive_descriptor() -> ArchitectureDescriptor:
    return ArchitectureDescriptor(
        architecture_id="adaptive",
        implementation_version="0.1.0",
        parameters_model=AdaptiveParameters,
        default_training_policy=SingleFitPolicy(optimizer="extra_trees"),
        trains_by_epoch=False,
        supported_optimizers=("extra_trees",),
        supported_training_budgets=("fit",),
        required_observation_fields=("last_price",),
        dependency_modules=("numpy", "sklearn"),
        backend_factory="regimex.backends.adaptive:create_backend",
        worker_probe="regimex.backends.adaptive:probe_runtime",
    )


def baseline_descriptor() -> ArchitectureDescriptor:
    """Describe the real adapter without importing its scientific dependencies."""
    return ArchitectureDescriptor(
        architecture_id="baseline",
        implementation_version="0.1.0",
        parameters_model=LogisticParameters,
        default_training_policy=BASELINE_DEFAULT_POLICY,
        policy_validator=validate_baseline_policy,
        trains_by_epoch=False,
        supported_optimizers=("lbfgs",),
        # Activity and quotes contribute when recorded; their explicit masks
        # preserve the existing baseline's documented missing-data behavior.
        required_observation_fields=("last_price",),
        dependency_modules=("numpy", "sklearn", "threadpoolctl"),
        backend_factory="regimex.backends.baseline:create_backend",
        worker_probe="regimex.backends.baseline:probe_runtime",
    )


def tcn_descriptor() -> ArchitectureDescriptor:
    """Discover the real CPU TCN backend without importing its scientific runtime."""
    sizes = []
    for width in (16, 32, 48, 64):
        parameters = TCNParameters(channels=width)
        sizes.append(
            ModelSize(
                name=f"{width} channels",
                parameters=parameters.model_dump(
                    mode="json", include={"channels", "dilations", "side_slopes"}
                ),
                trainable_parameters=tcn_layout.trainable_parameter_count(parameters),
            )
        )
    return ArchitectureDescriptor(
        architecture_id="tcn",
        implementation_version="0.2.0",
        parameters_model=TCNParameters,
        sizes=tuple(sizes),
        default_training_policy=TrainingPolicy(optimizer="adamw"),
        policy_validator=validate_tcn_policy,
        supported_optimizers=("adamw",),
        supported_training_budgets=("epochs", "updates"),
        required_observation_fields=("last_price",),
        dependency_modules=("numpy", "torch"),
        backend_factory="regimex.backends.tcn:create_backend",
        worker_probe="regimex.backends.tcn:probe_runtime",
    )


def validate_persistence_policy(policy: TrainingPolicySpec) -> None:
    if not isinstance(policy, TrainingPolicy) or policy.optimizer != "adamw":
        raise RegistryError("Direction persistence requires epoch training with AdamW")
    if len(policy.seeds) > 32:
        raise RegistryError("Direction persistence supports at most 32 search seeds")


def persistence_descriptor() -> ArchitectureDescriptor:
    return ArchitectureDescriptor(
        architecture_id="persistence",
        implementation_version="0.1.0",
        parameters_model=PersistenceParameters,
        default_training_policy=TrainingPolicy(
            optimizer="adamw",
            seeds=(11,),
            epochs=24,
            batch_size=32,
            learning_rate=0.003,
            validation_interval=1,
            patience=6,
        ),
        policy_validator=validate_persistence_policy,
        supported_optimizers=("adamw",),
        required_observation_fields=("last_price",),
        dependency_modules=("numpy", "torch"),
        backend_factory="regimex.backends.persistence:create_backend",
        worker_probe="regimex.backends.persistence:probe_runtime",
    )
