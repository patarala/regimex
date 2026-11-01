"""Final-benchmark reservations and attested observation exposure history.

Inventory and blindness statements are attestations, not proof of unseen data.
The application persists validated revisions transactionally in PostgreSQL.
"""

import hashlib
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from regimex.contracts import Identifier, Seed, StrictModel
from regimex.storage import json_bytes

SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Partition = Literal["train", "validation", "test", "external", "development", "final-review"]


class Record(StrictModel):
    # Revalidate records at persistence boundaries, including model_copy updates.
    model_config = ConfigDict(revalidate_instances="always")


def _hash(value: object) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


class BenchmarkMember(Record):
    source_namespace: Identifier
    experiment_id: Identifier
    group_id: Identifier
    seed: Seed | None = None
    observation_hashes: Annotated[tuple[SHA256, ...], Field(min_length=1)]


def _member_keys(member: BenchmarkMember) -> set[tuple[str, ...]]:
    """Index every independence constraint without pairwise historical scans."""
    keys = {
        ("experiment", member.source_namespace, member.experiment_id),
        ("group", member.group_id),
        *(("observation", value) for value in member.observation_hashes),
    }
    if member.seed is not None:
        keys.add(("seed", str(member.seed)))
    return keys


class PartitionMembership(Record):
    partition: Partition
    member: BenchmarkMember


class DatasetHistory(Record):
    dataset_id: Identifier
    dataset_sha256: SHA256
    recorded_at: AwareDatetime
    memberships: Annotated[tuple[PartitionMembership, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def isolated_partitions(self) -> Self:
        seen: set[tuple[str, str]] = set()
        partitions: dict[tuple[str, ...], Partition] = {}
        for item in self.memberships:
            key = (item.member.source_namespace, item.member.experiment_id)
            if key in seen:
                raise ValueError("dataset contains duplicate experiment membership")
            seen.add(key)
            for identity in _member_keys(item.member):
                if identity in partitions and partitions[identity] != item.partition:
                    raise ValueError("dataset history has overlapping partitions")
                partitions[identity] = item.partition
        return self


class HistoryAttestation(Record):
    """A named assertion of inventory completeness, scoped to source namespaces."""

    source_namespaces: Annotated[tuple[Identifier, ...], Field(min_length=1)]
    attested_by: Identifier
    statement: Annotated[str, Field(min_length=1, max_length=4096)]
    at: AwareDatetime
    history_sha256: SHA256


class DirectionSemantics(Record):
    """Identity of compatible classes, interval bounds, availability and decoder.

    The profile digest binds the actual interval/availability configuration;
    annotation algorithm and reviewer identity belong to reference provenance.
    """

    version: Literal["direction-semantics/1.0"] = "direction-semantics/1.0"
    profile_sha256: SHA256
    decoder_version: Identifier
    classes: tuple[Literal["sideways"], Literal["growth"], Literal["decline"]] = (
        "sideways",
        "growth",
        "decline",
    )
    interval_bounds: Literal["[start,stop)"] = "[start,stop)"
    availability: tuple[Literal["observed"], Literal["no_trades"], Literal["insufficient_data"]] = (
        "observed",
        "no_trades",
        "insufficient_data",
    )


class ReferenceProvenance(Record):
    reference_set_sha256: SHA256
    kind: Literal["automatic", "human"]
    policy: Identifier
    semantics: DirectionSemantics
    reviewer: Identifier | None = None
    blind_review: bool = False
    independent: bool = False

    @model_validator(mode="after")
    def honest_independence(self) -> Self:
        if self.kind == "automatic" and (
            self.reviewer is not None or self.blind_review or self.independent
        ):
            raise ValueError("automatic references cannot claim independent human review")
        if self.kind == "human" and (self.reviewer is None or not self.reviewer.strip()):
            raise ValueError("human references require a reviewer")
        if self.independent and not self.blind_review:
            raise ValueError("independent human references require blind-review attestation")
        return self


class ChoiceFreeze(Record):
    at: AwareDatetime
    choices_sha256: SHA256
    model_artifact_sha256: SHA256
    training_dataset_hashes: Annotated[tuple[SHA256, ...], Field(min_length=1)]
    training_reference: ReferenceProvenance
    history_count: Annotated[int, Field(ge=0)]
    history_sha256: SHA256
    inventory_attestation: HistoryAttestation


class BenchmarkReservation(Record):
    benchmark_id: Identifier
    reserved_at: AwareDatetime
    members: Annotated[tuple[BenchmarkMember, ...], Field(min_length=1)]
    reference: ReferenceProvenance
    history_count: Annotated[int, Field(ge=0)]
    history_sha256: SHA256
    inventory_attestation: HistoryAttestation
    selection_started_at: AwareDatetime | None = None
    freeze: ChoiceFreeze | None = None
    opened_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def ordered_events(self) -> Self:
        if (
            self.inventory_attestation.history_sha256 != self.history_sha256
            or self.inventory_attestation.at > self.reserved_at
            or any(
                member.source_namespace not in self.inventory_attestation.source_namespaces
                for member in self.members
            )
        ):
            raise ValueError("reservation requires a prior complete-history attestation")
        for index, member in enumerate(self.members):
            for previous in self.members[:index]:
                same_id = (previous.source_namespace, previous.experiment_id) == (
                    member.source_namespace,
                    member.experiment_id,
                )
                if same_id or set(previous.observation_hashes) & set(member.observation_hashes):
                    raise ValueError("benchmark contains duplicate experiments or observations")
        if self.selection_started_at is not None and self.selection_started_at <= self.reserved_at:
            raise ValueError("selection must start after benchmark reservation")
        if self.freeze is not None:
            if self.selection_started_at is None or self.freeze.at <= self.selection_started_at:
                raise ValueError("choices must freeze after selection starts")
            if self.freeze.training_reference.semantics != self.reference.semantics:
                raise ValueError("training labels and final references have incompatible semantics")
            if (
                self.freeze.inventory_attestation.history_sha256 != self.freeze.history_sha256
                or self.freeze.inventory_attestation.at > self.freeze.at
            ):
                raise ValueError("freeze requires a prior complete-history attestation")
        if self.opened_at is not None and (self.freeze is None or self.opened_at <= self.freeze.at):
            raise ValueError("benchmark may open only after choices freeze")
        return self


def history_hash(history: tuple[DatasetHistory, ...]) -> str:
    return _hash([item.model_dump(mode="json") for item in history])


class BenchmarkRegistry(Record):
    version: Literal["benchmark-registry/1.0"] = "benchmark-registry/1.0"
    history: tuple[DatasetHistory, ...] = ()
    history_attestation: HistoryAttestation | None = None
    benchmarks: tuple[BenchmarkReservation, ...] = ()

    @model_validator(mode="after")
    def independent_reservations(self) -> Self:
        if len({item.dataset_sha256 for item in self.history}) != len(self.history):
            raise ValueError("dataset history is already registered")
        if len({item.benchmark_id for item in self.benchmarks}) != len(self.benchmarks):
            raise ValueError("benchmark identifier is already reserved")
        if self.history_attestation is not None:
            attestation = self.history_attestation
            namespaces = set(attestation.source_namespaces)
            if len(namespaces) != len(attestation.source_namespaces):
                raise ValueError("history attestation contains duplicate namespaces")
            if attestation.history_sha256 != history_hash(self.history) or any(
                row.member.source_namespace not in namespaces
                for item in self.history
                for row in item.memberships
            ):
                raise ValueError("history attestation does not cover the complete recorded history")
            if any(item.recorded_at > attestation.at for item in self.history):
                raise ValueError("history must be recorded before its completeness attestation")
        prior_keys = {
            identity
            for item in self.history
            for row in item.memberships
            for identity in _member_keys(row.member)
        }
        for reservation in self.benchmarks:
            self._check_history_binding(reservation.history_count, reservation.history_sha256)
            self._check_attestation(reservation.inventory_attestation, reservation.history_count)
            if any(_member_keys(member) & prior_keys for member in reservation.members):
                raise ValueError("benchmark overlaps a prior partition, seed group, or observation")
            prior_keys.update(
                identity for member in reservation.members for identity in _member_keys(member)
            )
            if reservation.freeze is not None:
                freeze = reservation.freeze
                self._check_history_binding(freeze.history_count, freeze.history_sha256)
                self._check_attestation(freeze.inventory_attestation, freeze.history_count)
                known = {item.dataset_sha256 for item in self.history[: freeze.history_count]}
                if not set(freeze.training_dataset_hashes) <= known:
                    raise ValueError("choice freeze references an unregistered training dataset")
        return self

    def _check_history_binding(self, count: int, expected: str) -> None:
        if count > len(self.history) or history_hash(self.history[:count]) != expected:
            raise ValueError("reservation or freeze history identity disagrees with the registry")

    def _check_attestation(self, attestation: HistoryAttestation, count: int) -> None:
        BenchmarkRegistry(
            history=self.history[:count], history_attestation=attestation, benchmarks=()
        )


def _replace(registry: BenchmarkRegistry, **updates: object) -> BenchmarkRegistry:
    return BenchmarkRegistry.model_validate({**registry.model_dump(), **updates})


def register_history(registry: BenchmarkRegistry, history: DatasetHistory) -> BenchmarkRegistry:
    """Register a complete dataset; overlapping historical datasets remain recorded."""
    return _replace(registry, history=(*registry.history, history), history_attestation=None)


def attest_history(
    registry: BenchmarkRegistry,
    *,
    source_namespaces: tuple[str, ...],
    attested_by: str,
    statement: str,
    at: AwareDatetime,
) -> BenchmarkRegistry:
    return _replace(
        registry,
        history_attestation=HistoryAttestation(
            source_namespaces=source_namespaces,
            attested_by=attested_by,
            statement=statement,
            at=at,
            history_sha256=history_hash(registry.history),
        ),
    )


def _require_attestation(registry: BenchmarkRegistry) -> HistoryAttestation:
    if registry.history_attestation is None:
        raise ValueError("complete historical inventory must be explicitly attested")
    return registry.history_attestation


def reserve_benchmark(
    registry: BenchmarkRegistry,
    *,
    benchmark_id: str,
    members: tuple[BenchmarkMember, ...],
    reference: ReferenceProvenance,
    at: AwareDatetime,
) -> BenchmarkRegistry:
    """Reserve metadata before selection, without loading observations or references."""
    attestation = _require_attestation(registry)
    if at < attestation.at or any(
        member.source_namespace not in attestation.source_namespaces for member in members
    ):
        raise ValueError("reservation requires a prior attestation covering its source namespaces")
    reservation = BenchmarkReservation(
        benchmark_id=benchmark_id,
        reserved_at=at,
        members=members,
        reference=reference,
        history_count=len(registry.history),
        history_sha256=history_hash(registry.history),
        inventory_attestation=attestation,
    )
    return _replace(registry, benchmarks=(*registry.benchmarks, reservation))


def _update_reservation(
    registry: BenchmarkRegistry, benchmark_id: str, **updates: object
) -> BenchmarkRegistry:
    if benchmark_id not in {item.benchmark_id for item in registry.benchmarks}:
        raise ValueError("unknown benchmark reservation")
    reservations = tuple(
        BenchmarkReservation.model_validate({**item.model_dump(), **updates})
        if item.benchmark_id == benchmark_id
        else item
        for item in registry.benchmarks
    )
    return _replace(registry, benchmarks=reservations)


def _reservation(registry: BenchmarkRegistry, benchmark_id: str) -> BenchmarkReservation:
    for item in registry.benchmarks:
        if item.benchmark_id == benchmark_id:
            return item
    raise ValueError("unknown benchmark reservation")


def record_selection_started(
    registry: BenchmarkRegistry, benchmark_id: str, *, at: AwareDatetime
) -> BenchmarkRegistry:
    """Record the boundary before callers perform any new model selection."""
    attestation = _require_attestation(registry)
    if at < attestation.at:
        raise ValueError("selection start requires a prior complete-history attestation")
    if _reservation(registry, benchmark_id).selection_started_at is not None:
        raise ValueError("selection start is already recorded")
    return _update_reservation(registry, benchmark_id, selection_started_at=at)


def freeze_choices(
    registry: BenchmarkRegistry,
    benchmark_id: str,
    *,
    at: AwareDatetime,
    choices_sha256: str,
    model_artifact_sha256: str,
    training_dataset_hashes: tuple[str, ...],
    training_reference: ReferenceProvenance,
) -> BenchmarkRegistry:
    attestation = _require_attestation(registry)
    if at < attestation.at or _reservation(registry, benchmark_id).freeze is not None:
        raise ValueError("freeze requires a current attestation and previously unfrozen choices")
    freeze = ChoiceFreeze(
        at=at,
        choices_sha256=choices_sha256,
        model_artifact_sha256=model_artifact_sha256,
        training_dataset_hashes=training_dataset_hashes,
        training_reference=training_reference,
        history_count=len(registry.history),
        history_sha256=history_hash(registry.history),
        inventory_attestation=attestation,
    )
    return _update_reservation(registry, benchmark_id, freeze=freeze)


def open_benchmark(
    registry: BenchmarkRegistry, benchmark_id: str, *, at: AwareDatetime
) -> BenchmarkRegistry:
    """Record explicit exposure before the caller reads final references or scores.

    This function returns no reference data. The application authorizes access
    and commits the exposure before the evaluator reads any final references.
    """
    attestation = _require_attestation(registry)
    if at < attestation.at or _reservation(registry, benchmark_id).opened_at is not None:
        raise ValueError(
            "opening requires a current attestation and a previously unopened benchmark"
        )
    return _update_reservation(registry, benchmark_id, opened_at=at)


class RegistryRevision(Record):
    version: Literal["benchmark-registry-revision/1.0"] = "benchmark-registry-revision/1.0"
    revision: Annotated[int, Field(ge=0)]
    previous_sha256: SHA256 | None
    registry: BenchmarkRegistry


def _preserves_history(previous: BenchmarkRegistry, current: BenchmarkRegistry) -> None:
    if current.history[: len(previous.history)] != previous.history or len(
        current.benchmarks
    ) < len(previous.benchmarks):
        raise ValueError("registry revisions cannot remove or rewrite recorded history")
    for old, new in zip(previous.benchmarks, current.benchmarks, strict=False):
        immutable = old.model_dump(exclude={"selection_started_at", "freeze", "opened_at"})
        if immutable != new.model_dump(exclude={"selection_started_at", "freeze", "opened_at"}):
            raise ValueError("registry revisions cannot rewrite benchmark reservations")
        for field in ("selection_started_at", "freeze", "opened_at"):
            value = getattr(old, field)
            if value is not None and value != getattr(new, field):
                raise ValueError(
                    "registry revisions cannot rewrite selection, freeze, or open events"
                )
