"""Backend-independent envelope and integrity checks, before any weight codec runs.

An accepted envelope authenticates recorded bytes and metadata. It does not
establish that weights, preprocessing, or settings form a usable model; every
backend must validate its codec and scientific invariants before inference.
"""

import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from regimex.contracts import Identifier, Seed, StrictModel
from regimex.operations import Sha256

BundleRole = Literal["configuration", "preprocessing", "weights", "validation", "history"]
REQUIRED_ROLES = frozenset({"configuration", "preprocessing", "weights", "validation", "history"})
MAX_BUNDLE_MANIFEST_BYTES = 256 * 1024


class BundleMember(StrictModel):
    path: Annotated[str, Field(min_length=1, max_length=256)]
    role: BundleRole
    size_bytes: Annotated[int, Field(ge=1, le=512 * 1024 * 1024)]
    sha256: Sha256

    @model_validator(mode="after")
    def safe_relative_path(self) -> Self:
        if (
            self.path == "manifest.json"
            or "\\" in self.path
            or ":" in self.path
            or "\x00" in self.path
            or PurePosixPath(self.path).is_absolute()
            or any(part in {"", ".", ".."} for part in self.path.split("/"))
        ):
            raise ValueError("bundle member must have a safe relative POSIX path")
        return self


class ReferenceIdentity(StrictModel):
    reference_set_id: Identifier
    kind: Literal["automatic", "human"]
    policy: Identifier
    content_sha256: Sha256
    direction_semantics_sha256: Sha256


class TrainingEvidence(StrictModel):
    run_id: Identifier
    dataset_sha256: Sha256
    split_manifest_sha256: Sha256
    labels: ReferenceIdentity
    validation_references: ReferenceIdentity
    seeds: Annotated[tuple[Seed, ...], Field(min_length=1)]
    selected_seed: Seed
    selection_metric: Literal["decoded_segment_macro_f1/iou-0.50"]
    validation_segment_macro_f1: Annotated[float, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def compatible_references(self) -> Self:
        if len(set(self.seeds)) != len(self.seeds) or self.selected_seed not in self.seeds:
            raise ValueError("selected seed must belong to a unique recorded search")
        if (
            self.labels.direction_semantics_sha256
            != self.validation_references.direction_semantics_sha256
        ):
            raise ValueError("training and validation direction semantics differ")
        return self


class ModelBundle(StrictModel):
    bundle_version: Literal["1.0"]
    model_version_id: Identifier
    architecture_id: Identifier
    implementation_version: Identifier
    interface_version: Literal["1.0"]
    observation_version: Literal["2.1"]
    feature_version: Identifier
    preprocessing_version: Identifier
    decoder_version: Identifier
    analysis_profile: Identifier
    analysis_profile_sha256: Sha256
    class_names: tuple[Literal["sideways"], Literal["growth"], Literal["decline"]]
    training: TrainingEvidence
    # Full resolved settings and environment belong to checksummed JSON members.
    build_id: Identifier
    environment_sha256: Sha256
    members: Annotated[tuple[BundleMember, ...], Field(min_length=5, max_length=64)]

    @model_validator(mode="after")
    def complete_bundle(self) -> Self:
        if len({member.path for member in self.members}) != len(self.members):
            raise ValueError("bundle member paths must be unique")
        roles = [member.role for member in self.members]
        if set(roles) != REQUIRED_ROLES or len(roles) != len(REQUIRED_ROLES):
            raise ValueError("bundle requires exactly one member for every required role")
        if sum(member.size_bytes for member in self.members) > 1024 * 1024 * 1024:
            raise ValueError("bundle exceeds its total materialized size limit")
        return self


def verify_model_bundle(
    root: Path, *, manifest_sha256: str, architecture_id: str, implementation_version: str
) -> ModelBundle:
    """Verify recorded bytes; backend loaders still validate arrays and settings.

    The caller must retain the materialized files immutably through backend
    loading. These integrity checks do not authorize later mutable replacements.
    """
    try:
        with _directory(root) as root_descriptor:
            with _regular_file(root_descriptor, "manifest.json") as manifest_descriptor:
                if os.fstat(manifest_descriptor).st_size > MAX_BUNDLE_MANIFEST_BYTES:
                    raise ValueError("bundle manifest exceeds its input size limit")
                with os.fdopen(os.dup(manifest_descriptor), "rb") as stream:
                    manifest_bytes = stream.read(MAX_BUNDLE_MANIFEST_BYTES + 1)
            if len(manifest_bytes) > MAX_BUNDLE_MANIFEST_BYTES:
                raise ValueError("bundle manifest exceeds its input size limit")
            if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
                raise ValueError("bundle manifest differs from its committed digest")
            bundle = ModelBundle.model_validate_json(manifest_bytes)
            if (bundle.architecture_id, bundle.implementation_version) != (
                architecture_id,
                implementation_version,
            ):
                raise ValueError("bundle architecture implementation is incompatible")
            expected = {member.path for member in bundle.members} | {"manifest.json"}
            _verify_entries(root_descriptor, expected)
            for member in bundle.members:
                with _regular_file(root_descriptor, member.path) as descriptor:
                    if os.fstat(descriptor).st_size != member.size_bytes:
                        raise ValueError(f"bundle member size differs: {member.path}")
                    checksum = hashlib.sha256()
                    remaining = member.size_bytes
                    with os.fdopen(os.dup(descriptor), "rb") as stream:
                        while remaining:
                            block = stream.read(min(remaining, 1024 * 1024))
                            if not block:
                                raise ValueError(f"bundle member size differs: {member.path}")
                            remaining -= len(block)
                            checksum.update(block)
                        if stream.read(1):
                            raise ValueError(f"bundle member size differs: {member.path}")
                    if checksum.hexdigest() != member.sha256:
                        raise ValueError(f"bundle member checksum differs: {member.path}")
            return bundle
    except OSError as exc:
        raise ValueError(
            "bundle must contain readable regular files without symbolic links"
        ) from exc


@contextmanager
def _directory(path: Path | str, *, parent: int | None = None) -> Iterator[int]:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    try:
        yield descriptor
    finally:
        os.close(descriptor)


@contextmanager
def _regular_file(root_descriptor: int, path: str) -> Iterator[int]:
    """Open every component without following links, including directory swaps."""
    parent = os.dup(root_descriptor)
    try:
        parts = path.split("/")
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("bundle members must be regular files")
            yield descriptor
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def _verify_entries(root_descriptor: int, expected: set[str]) -> None:
    directories = {
        str(parent)
        for name in expected
        for parent in PurePosixPath(name).parents
        if str(parent) != "."
    }
    actual: set[str] = set()

    def visit(descriptor: int, prefix: str) -> None:
        with os.scandir(descriptor) as entries:
            for entry in entries:
                name = prefix + entry.name
                if entry.is_symlink():
                    raise ValueError("bundle cannot contain symbolic links")
                if entry.is_dir(follow_symlinks=False) and name in directories:
                    with _directory(entry.name, parent=descriptor) as child:
                        visit(child, name + "/")
                elif entry.is_file(follow_symlinks=False) and name in expected:
                    actual.add(name)
                else:
                    raise ValueError("bundle files differ from its manifest")

    visit(root_descriptor, "")
    if actual != expected:
        raise ValueError("bundle files differ from its manifest")
