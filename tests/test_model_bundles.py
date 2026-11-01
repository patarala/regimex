"""Reject corrupt/incompatible artifacts before invoking a backend or loading weights."""

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from regimex.model_bundles import (
    MAX_BUNDLE_MANIFEST_BYTES,
    BundleMember,
    ModelBundle,
    ReferenceIdentity,
    TrainingEvidence,
    verify_model_bundle,
)
from regimex.storage import digest, write_json


def bundle(root: Path) -> ModelBundle:
    root.mkdir()
    members = []
    for role in ("configuration", "preprocessing", "weights", "validation", "history"):
        path = root / f"{role}.bin"
        path.write_bytes(role.encode())
        members.append(
            BundleMember.model_validate(
                {
                    "path": path.name,
                    "role": role,
                    "size_bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
        )
    automatic = ReferenceIdentity(
        reference_set_id="automatic-1",
        kind="automatic",
        policy="automatic/1",
        content_sha256="1" * 64,
        direction_semantics_sha256="2" * 64,
    )
    human = ReferenceIdentity(
        reference_set_id="human-validation-1",
        kind="human",
        policy="human/1",
        content_sha256="3" * 64,
        direction_semantics_sha256="2" * 64,
    )
    result = ModelBundle(
        bundle_version="1.0",
        model_version_id="baseline-1",
        architecture_id="baseline",
        implementation_version="1.0",
        interface_version="1.0",
        observation_version="2.1",
        feature_version="aggregate/3.0",
        preprocessing_version="standard/1.0",
        decoder_version="duration/1.0",
        analysis_profile="direction-32/2.0",
        analysis_profile_sha256="4" * 64,
        class_names=("sideways", "growth", "decline"),
        training=TrainingEvidence(
            run_id="run-1",
            dataset_sha256="5" * 64,
            split_manifest_sha256="6" * 64,
            labels=automatic,
            validation_references=human,
            seeds=(11, 23),
            selected_seed=11,
            selection_metric="decoded_segment_macro_f1/iou-0.50",
            validation_segment_macro_f1=0.5,
        ),
        build_id="build-1",
        environment_sha256="7" * 64,
        members=tuple(members),
    )
    write_json(root / "manifest.json", result.model_dump(mode="json"))
    return result


def verify(root: Path) -> ModelBundle:
    return verify_model_bundle(
        root,
        manifest_sha256=digest(root / "manifest.json"),
        architecture_id="baseline",
        implementation_version="1.0",
    )


def test_integrity_checks_allow_independent_human_validation_provenance(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    expected = bundle(root)
    assert verify(root) == expected
    assert expected.training.labels.kind == "automatic"
    assert expected.training.validation_references.kind == "human"


@pytest.mark.parametrize("change", ["size", "checksum", "missing", "extra", "symlink"])
def test_corrupt_members_are_rejected(tmp_path: Path, change: str) -> None:
    root = tmp_path / "bundle"
    bundle(root)
    weights = root / "weights.bin"
    if change == "size":
        weights.write_bytes(b"too short")
    elif change == "checksum":
        weights.write_bytes(b"WEIGHTS")
    elif change == "missing":
        weights.unlink()
    elif change == "extra":
        (root / "surprise").write_bytes(b"unexpected")
    else:
        external = tmp_path / "external"
        weights.rename(external)
        weights.symlink_to(external)
    with pytest.raises(ValueError, match="bundle"):
        verify(root)


def test_commit_digest_and_architecture_are_verified(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    bundle(root)
    with pytest.raises(ValueError, match="committed digest"):
        verify_model_bundle(
            root, manifest_sha256="0" * 64, architecture_id="baseline", implementation_version="1.0"
        )
    with pytest.raises(ValueError, match="incompatible"):
        verify_model_bundle(
            root,
            manifest_sha256=digest(root / "manifest.json"),
            architecture_id="tcn",
            implementation_version="1.0",
        )


@pytest.mark.parametrize(
    "path",
    [
        "../weights",
        "/weights",
        "a//b",
        "a/./b",
        "a/../b",
        "a\\b",
        "C:weights",
        "manifest.json",
        "a\x00b",
    ],
)
def test_unsafe_member_paths_are_rejected(path: str) -> None:
    with pytest.raises(ValidationError, match="safe relative"):
        BundleMember(path=path, role="weights", size_bytes=1, sha256="a" * 64)


def test_incompatible_reference_semantics_and_duplicate_members_fail(tmp_path: Path) -> None:
    data = bundle(tmp_path / "bundle").model_dump(mode="json")
    data["training"]["validation_references"]["direction_semantics_sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="semantics differ"):
        ModelBundle.model_validate_json(json.dumps(data))
    data["training"]["validation_references"]["direction_semantics_sha256"] = "2" * 64
    data["members"][1]["path"] = data["members"][0]["path"]
    with pytest.raises(ValidationError, match="paths must be unique"):
        ModelBundle.model_validate_json(json.dumps(data))


def test_manifest_is_bounded_before_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "manifest.json").write_bytes(b"x" * (MAX_BUNDLE_MANIFEST_BYTES + 1))

    def unexpected_hash(data: bytes) -> None:
        pytest.fail("oversized untrusted manifest must be rejected before hashing")

    monkeypatch.setattr("regimex.model_bundles.hashlib.sha256", unexpected_hash)
    with pytest.raises(ValueError, match="input size limit"):
        verify_model_bundle(
            root, manifest_sha256="0" * 64, architecture_id="baseline", implementation_version="1.0"
        )


def test_nested_member_directories_are_supported_and_links_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    expected = bundle(root)
    nested = root / "nested"
    nested.mkdir()
    (root / "weights.bin").rename(nested / "weights.bin")
    data = expected.model_dump(mode="json")
    data["members"][2]["path"] = "nested/weights.bin"
    write_json(root / "manifest.json", data)
    assert verify(root).members[2].path == "nested/weights.bin"
    external = tmp_path / "external"
    nested.rename(external)
    nested.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic links"):
        verify(root)


@pytest.mark.parametrize("special_manifest", [False, True])
def test_special_files_are_rejected_without_blocking(
    tmp_path: Path, special_manifest: bool
) -> None:
    root = tmp_path / "bundle"
    bundle(root)
    manifest_sha256 = digest(root / "manifest.json")
    path = root / ("manifest.json" if special_manifest else "weights.bin")
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(ValueError, match="bundle"):
        verify_model_bundle(
            root,
            manifest_sha256=manifest_sha256,
            architecture_id="baseline",
            implementation_version="1.0",
        )


def test_class_order_cannot_silently_reinterpret_probabilities(tmp_path: Path) -> None:
    data = bundle(tmp_path / "bundle").model_dump(mode="json")
    data["class_names"] = ["growth", "sideways", "decline"]
    with pytest.raises(ValidationError, match="class_names"):
        ModelBundle.model_validate_json(json.dumps(data))


def test_envelope_verification_does_not_claim_a_loadable_backend_model(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    bundle(root)
    verified = verify(root)
    weights = next(member for member in verified.members if member.role == "weights")
    # Fixture bytes deliberately are not any backend's weight codec.
    assert (root / weights.path).read_bytes() == b"weights"
