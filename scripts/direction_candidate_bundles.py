"""Package every fitted forest using the observations and scores frozen at validation."""

import os
from pathlib import Path

from regimex.external_evaluations import read_evidence_file
from regimex.model_bundles import AdditionalTrainingSource, verify_model_bundle
from regimex.storage import digest, output_directory, write_json
from regimex.terminal_training import register_completed_bundle


def package_candidates(data: Path, comparison: Path, *, mixed: bool, explicit: bool = False):
    if not explicit and not os.environ.get("REGIMEX_TRAINING_TOKEN_FILE"):
        return
    import evaluate_large_direction as native
    import evaluate_mixed_direction as joint

    selection = comparison / "selection"
    frozen, freeze_sha = read_evidence_file(selection / "freeze.json")
    expected = joint.input_digests(data) if mixed else native.evidence_files(data)
    if frozen["inputs"] != expected or frozen["protocol_sha256"] != digest(
        comparison / "protocol.json"
    ):
        raise ValueError("Training receipts, weights or protocol differ from the frozen selection")
    if mixed:
        validation, validation_sha = read_evidence_file(comparison / "native-validation.json")
        if (
            validation_sha != frozen["native_validation_sha256"]
            or validation["scores"] != frozen["validation"]
        ):
            raise ValueError("Mixed candidate scores differ from the frozen validation")
    selected = frozen["challenger"]
    model = verify_model_bundle(
        selection / "bundle",
        manifest_sha256=frozen["bundle_sha256"],
        architecture_id=native.AdaptiveBackend.architecture_id,
        implementation_version=native.AdaptiveBackend.implementation_version,
    )
    candidates = joint.MIX if mixed else native.CANDIDATES
    if selected not in candidates and not (
        mixed and selected == joint.V1 and frozen["bundle_sha256"] == joint.V1_SHA
    ):
        raise ValueError("Unknown frozen challenger")
    if selected in candidates and (
        next(member.sha256 for member in model.members if member.role == "weights")
        != expected[f"fitted/{selected}.npz"]
        or model.training.validation_segment_macro_f1
        != frozen["validation"][selected]["interval_macro_f1"]
    ):
        raise ValueError("Selected bundle differs from the original fit or validation")
    register_completed_bundle(selection / "bundle", frozen["bundle_sha256"])
    dataset, refs = native.development(data)
    corpus = native.PartitionCorpus(data, "validation", dataset)
    for name in candidates:
        if name == selected:
            continue
        destination = comparison / "training-candidates" / name
        bundle = destination / "bundle"
        model_id = f"direction-{name}-{'v2' if mixed else 'v1'}"
        build_id = f"package-from-freeze-{freeze_sha}"
        binding = {
            "freeze_sha256": freeze_sha,
            "weights_sha256": expected[f"fitted/{name}.npz"],
            "receipt_sha256": expected[f"fitted/{name}.json"],
            "validation_segment_macro_f1": frozen["validation"][name]["interval_macro_f1"],
        }
        if not destination.exists():
            load_fit = joint.read_fit if mixed else native.load_fit
            fit = load_fit(data, name, dataset, refs["train"])
            additional = ()
            if mixed:
                source = AdditionalTrainingSource.model_validate(
                    fit["environment"]["additional_training_source"]
                )
                additional = (source,)
            with output_directory(destination) as stage:
                sha, _ = native.package(
                    stage,
                    name,
                    fit,
                    binding["validation_segment_macro_f1"],
                    dataset,
                    refs,
                    corpus,
                    model_version_id=model_id,
                    additional_training_sources=additional,
                    build_id=build_id,
                )
                if digest(stage / "bundle" / "weights.npz") != binding["weights_sha256"]:
                    raise ValueError(f"Packaged weights differ from the frozen fit for {name}")
                write_json(stage / "binding.json", {**binding, "manifest_sha256": sha})
        record = native.read(destination / "binding.json")
        if any(record[key] != value for key, value in binding.items()):
            raise ValueError(f"Saved bundle for {name} belongs to another comparison")
        sha = record["manifest_sha256"]
        manifest = verify_model_bundle(
            bundle,
            manifest_sha256=sha,
            architecture_id=model.architecture_id,
            implementation_version=model.implementation_version,
        )
        if (
            manifest.model_version_id != model_id
            or manifest.build_id != build_id
            or next(member.sha256 for member in manifest.members if member.role == "weights")
            != binding["weights_sha256"]
            or manifest.training.validation_segment_macro_f1
            != binding["validation_segment_macro_f1"]
        ):
            raise ValueError(f"Saved bundle no longer matches the candidate {name}")
        print(f"Packaged {name}: {bundle} ({sha})", flush=True)
        register_completed_bundle(bundle, sha)
