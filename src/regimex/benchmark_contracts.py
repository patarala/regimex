"""Metadata for the service's attested benchmark and final-reference gate."""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field

from regimex.benchmarks import (
    BenchmarkMember,
    BenchmarkRegistry,
    BenchmarkReservation,
    _member_keys,
)
from regimex.contracts import Identifier, StrictModel
from regimex.data_catalog import SnapshotMember, content_hash
from regimex.operations import Sha256


def benchmark_member(member: SnapshotMember) -> BenchmarkMember:
    return BenchmarkMember(
        source_namespace=member.source_namespace,
        experiment_id=member.snapshot_id,
        group_id="raw-group:"
        + content_hash(member.split_group.model_dump(mode="json", exclude={"seed"})),
        seed=member.split_group.seed,
        observation_hashes=member.observation_hashes,
    )


def dataset_identity_keys(members: tuple[SnapshotMember, ...]) -> set[tuple[str, ...]]:
    return {
        key
        for member in members
        for key in (
            *_member_keys(benchmark_member(member)),
            ("run", member.source_namespace, member.source_id, member.run_id),
        )
    }


class RegistryState(StrictModel):
    revision: Annotated[int, Field(ge=-1)]
    sha256: Sha256 | None
    registry: BenchmarkRegistry
    access_scope: Literal["service_reference_gate_only"] = "service_reference_gate_only"
    inventory_scope: Literal["attested_named_sources"] = "attested_named_sources"


class BenchmarkBinding(StrictModel):
    owner_id: Identifier
    benchmark_id: Identifier
    dataset_id: Identifier
    dataset_sha256: Sha256
    reference_set_id: Identifier
    reference_sha256: Sha256
    partition: Literal["test"] = "test"


class FrozenChoice(StrictModel):
    benchmark_id: Identifier
    owner_id: Identifier
    model_version_id: Identifier
    bundle_sha256: Sha256
    inference_policy_sha256: Sha256
    choices_sha256: Sha256


class BenchmarkView(StrictModel):
    binding: BenchmarkBinding
    reservation: BenchmarkReservation
    frozen_choice: FrozenChoice | None = None
    registry_revision: Annotated[int, Field(ge=0)]
    registry_sha256: Sha256
    access_scope: Literal["service_reference_gate_only"] = "service_reference_gate_only"


class FinalEvaluationGrant(StrictModel):
    benchmark_id: Identifier
    owner_id: Identifier
    dataset_id: Identifier
    dataset_sha256: Sha256
    reference_set_id: Identifier
    reference_sha256: Sha256
    model_version_id: Identifier
    bundle_sha256: Sha256
    inference_policy_sha256: Sha256
    choices_sha256: Sha256
    partition: Literal["test"] = "test"
    opened_at: AwareDatetime
    registry_revision: Annotated[int, Field(ge=0)]
    registry_sha256: Sha256
