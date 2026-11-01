"""Lightweight architecture discovery and explicit worker capability checks.

Descriptor discovery may run in the API process. Dependency imports and runtime
probes run only through the methods explicitly named ``on_worker``. A descriptor
describes an installed architecture; it does not establish worker availability.
"""

import importlib
import json
from collections.abc import Callable, Collection, Iterable
from datetime import UTC, datetime
from importlib import metadata
from types import MappingProxyType
from typing import Annotated, Literal, Self, cast

from pydantic import ConfigDict, Field, JsonValue, model_validator

from regimex.contracts import MAX_STEPS, Seed, StrictModel

ENTRY_POINT_GROUP = "regimex.models"
INTERFACE_VERSION: Literal["1.0"] = "1.0"
Device = Literal["cpu"]
Optimizer = Literal["lbfgs", "adamw"]
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


class LogisticParameters(StrictModel):
    """Regularization candidates preserve the existing aggregate baseline search."""

    regularization: Annotated[tuple[Annotated[float, Field(gt=0)], ...], Field(min_length=1)] = (
        0.1,
        1.0,
        10.0,
    )
    context_sizes: tuple[Annotated[int, Field(ge=2, le=MAX_STEPS)], ...] = (32, 128)

    @model_validator(mode="after")
    def distinct_candidates(self) -> Self:
        if len(set(self.regularization)) != len(self.regularization):
            raise ValueError("regularization candidates must be unique")
        if len(set(self.context_sizes)) != len(self.context_sizes):
            raise ValueError("baseline context sizes must be unique")
        return self


class TCNParameters(StrictModel):
    channels: Annotated[int, Field(ge=4, le=256)] = 32
    dilations: Annotated[
        tuple[Annotated[int, Field(ge=1, le=MAX_STEPS)], ...], Field(min_length=1, max_length=32)
    ] = (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
    )
    dropout: Annotated[float, Field(ge=0, lt=1)] = 0.15
    smoothing: Annotated[float, Field(ge=0)] = 0.05
    interval_weight_power: Annotated[float, Field(ge=0, le=1)] = 0.0

    @model_validator(mode="after")
    def distinct_dilations(self) -> Self:
        if len(set(self.dilations)) != len(self.dilations):
            raise ValueError("TCN dilations must be unique")
        return self


class TrainingPolicy(StrictModel):
    optimizer: Optimizer
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


class ArchitectureInfo(StrictModel):
    """Serializable installed metadata; worker availability may still be unknown."""

    descriptor_version: Literal["1.0"]
    architecture_id: ArchitectureId
    implementation_version: Version
    interface_version: Literal["1.0"]
    parameter_schema: dict[str, JsonValue]
    default_training_policy: TrainingPolicy
    supported_optimizers: tuple[Optimizer, ...]
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
    default_training_policy: TrainingPolicy
    supported_optimizers: Annotated[tuple[Optimizer, ...], Field(min_length=1)]
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
        for values in (
            self.supported_optimizers,
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

    def validate_training_policy(self, policy: TrainingPolicy) -> None:
        if policy.optimizer not in self.supported_optimizers:
            raise RegistryError(
                f"architecture {self.architecture_id!r} "
                f"does not support optimizer {policy.optimizer!r}"
            )

    def validate_observation_fields(self, available_fields: Collection[str]) -> None:
        """Check recorded field capabilities, independently from per-row missingness."""
        missing = set(self.required_observation_fields).difference(available_fields)
        if missing:
            raise RegistryError(
                f"architecture {self.architecture_id!r} requires observation fields: "
                + ", ".join(sorted(missing))
            )

    def describe(self) -> ArchitectureInfo:
        return ArchitectureInfo(
            descriptor_version=self.descriptor_version,
            architecture_id=self.architecture_id,
            implementation_version=self.implementation_version,
            interface_version=self.interface_version,
            parameter_schema=self.parameters_model.model_json_schema(),
            default_training_policy=self.default_training_policy,
            supported_optimizers=self.supported_optimizers,
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

    def check_capability_on_worker(
        self, architecture_id: str, runtime: TrainingRuntime | InferenceRuntime, *, worker_id: str
    ) -> WorkerCapability:
        descriptor = self.get(architecture_id)
        checked: list[str] = []
        reasons: list[str] = []
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
            if not reasons:
                try:
                    _load_callable(descriptor.backend_factory)
                    assert descriptor.worker_probe is not None
                    probe = _load_callable(descriptor.worker_probe)
                    if probe(runtime.device) is not None:
                        raise RegistryError("worker runtime probes must return None on success")
                except Exception as exc:
                    reasons.append(f"backend runtime check failed: {exc}")
        return WorkerCapability(
            architecture_id=descriptor.architecture_id,
            implementation_version=descriptor.implementation_version,
            worker_id=worker_id,
            checked_at=datetime.now(UTC),
            device=runtime.device,
            checked_dependency_modules=tuple(checked),
            executable=not reasons,
            reasons=tuple(reasons),
        )

    def load_backend_factory_on_worker(
        self, architecture_id: str, runtime: TrainingRuntime | InferenceRuntime, *, worker_id: str
    ) -> Callable[[], object]:
        capability = self.check_capability_on_worker(architecture_id, runtime, worker_id=worker_id)
        if not capability.executable:
            raise RegistryError(
                f"architecture {architecture_id!r} is unavailable on worker {worker_id!r}: "
                + "; ".join(capability.reasons)
            )
        reference = self.get(architecture_id).backend_factory
        assert reference is not None
        return _load_callable(reference)


def baseline_descriptor() -> ArchitectureDescriptor:
    """Describe the real adapter without importing its scientific dependencies."""
    return ArchitectureDescriptor(
        architecture_id="baseline",
        implementation_version="0.1.0",
        parameters_model=LogisticParameters,
        default_training_policy=TrainingPolicy(optimizer="lbfgs", seeds=(11,)),
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
    return ArchitectureDescriptor(
        architecture_id="tcn",
        implementation_version="0.2.0",
        parameters_model=TCNParameters,
        default_training_policy=TrainingPolicy(optimizer="adamw"),
        supported_optimizers=("adamw",),
        required_observation_fields=("last_price",),
        dependency_modules=("numpy", "torch"),
        backend_factory="regimex.backends.tcn:create_backend",
        worker_probe="regimex.backends.tcn:probe_runtime",
    )
