"""Select one direction model jointly on native and controlled validation populations."""

import argparse
from pathlib import Path
from shutil import copytree

import evaluate_large_direction as native
from check_adaptive_gap_transfer import GATES, gate
from direction_candidate_bundles import package_candidates

from regimex.backends.adaptive import AdaptiveBackend, fit_result_from_forest
from regimex.backends.adaptive_forest import DirectionContextModel
from regimex.model_bundles import AdditionalTrainingSource
from regimex.model_registry import AdaptiveParameters, SingleFitPolicy, TrainingRuntime
from regimex.storage import digest, json_bytes, output_directory, write_json, write_new_file

DATA = Path("var/verification/direction-mixed-v2")
OUTPUT = Path("var/verification/direction-mixed-evaluation-v2")
CONTROLLED = Path("var/verification/direction-mixed-transfer-v2")
V1 = "multiscale-balanced"
V1_BUNDLE = native.OUTPUT / "selection" / "bundle"
V1_SHA = "e60f2a3f4de68436a396e904b1781492df94de5531f1c9b5bf7d55989e6d5495"
MIX = {"mixed-10": 0.10, "mixed-25": 0.25, "mixed-45": 0.45}
CHALLENGERS = (*MIX, V1)
ORDER = (*CHALLENGERS, native.CONTROL)


def protocol():
    return {
        "control": native.CONTROL,
        "control_bundle_sha256": native.BUNDLE_SHA,
        "previous_native_winner": V1,
        "previous_bundle_sha256": V1_SHA,
        "latent_training_weight_fractions": MIX,
        "tie_order": list(ORDER),
        "selection": "Keep challengers with native validation interval macro-F1 at least the "
        "release score and passing every controlled validation margin. Choose the highest native "
        "interval macro-F1 among those candidates.",
        "fallback": "If none qualifies, record the challenger maximizing the smaller of the two "
        "pooled F1 differences from the release. This choice is diagnostic. A final evaluation "
        "cannot change its validation status.",
        "native_gate": native.protocol()["native_gate"],
        "native_maximum_loss": native.MARGINS,
        "controlled_margins": GATES,
        "bootstrap": native.protocol()["bootstrap"],
    }


def input_digests(data):
    paths = [
        "candidate-plan.json",
        "prior.json",
        "population.json",
        "reservation.json",
        "dataset.json",
        "latent-training/recipe.json",
        "latent-training/manifest.json",
    ]
    for part in ("train", "validation", "test"):
        paths += [f"{part}/{name}.json" for name in ("reservation", "locations", "dataset")]
        if part != "test":
            paths.append(f"{part}/references.json")
    paths += [f"fitted/{name}.{suffix}" for name in MIX for suffix in ("json", "npz")]
    return {path: digest(data / path) for path in paths}


def implementation():
    return {
        **native.implementation(),
        "joint_selection": digest(Path(__file__)),
        "controlled_gate": digest(Path(__file__).with_name("check_adaptive_gap_transfer.py")),
    }


def read_fit(data, name, dataset, labels):
    root = data / "latent-training"
    receipt = native.read(data / "fitted" / f"{name}.json")
    lineage = AdditionalTrainingSource.model_validate(receipt["additional_training_source"])
    manifest = native.read(root / "manifest.json")
    for key in (
        "source_id",
        "exports_sha256",
        "references_sha256",
        "recipe_sha256",
        "experiment_count",
        "classified_steps",
    ):
        if getattr(lineage, key) != manifest[key]:
            raise ValueError(f"Changed latent training population for {name}")
    path = data / "fitted" / f"{name}.npz"
    parameters = AdaptiveParameters(
        feature_set="multiscale", sample_weighting="interval", trees=80, max_depth=15
    )
    if (
        lineage.weight_fraction != MIX[name]
        or lineage.manifest_sha256 != digest(root / "manifest.json")
        or lineage.recipe_sha256 != digest(root / "recipe.json")
        or receipt["weights_sha256"] != digest(path)
        or receipt["training_receipt_sha256"] != digest(data / "train" / "reservation.json")
        or receipt["population_receipt_sha256"] != digest(data / "reservation.json")
        or receipt["dataset_sha256"] != dataset.content_sha256
        or receipt["references_sha256"] != labels.content_sha256
        or AdaptiveParameters.model_validate(receipt["parameters"]) != parameters
        or receipt["seed"] != 173
    ):
        raise ValueError(f"Unexpected mixed training inputs or recipe for {name}")
    return {
        "model": DirectionContextModel.load(
            path, trees=parameters.trees, max_depth=parameters.max_depth
        ),
        "parameters": parameters,
        "policy": SingleFitPolicy(optimizer="extra_trees", seeds=(receipt["seed"],)),
        "analysis": native.PROFILE,
        "runtime": TrainingRuntime(threads=receipt["threads"]),
        "environment": receipt,
    }


def load_validation_models(data, control, v1bundle):
    dataset, refs = native.development(data)
    backend = AdaptiveBackend()
    models = {
        native.CONTROL: backend.load_bundle(
            control, manifest_sha256=native.BUNDLE_SHA, runtime=native.RUNTIME
        ),
        V1: backend.load_bundle(v1bundle, manifest_sha256=V1_SHA, runtime=native.RUNTIME),
    }
    for name in MIX:
        spec = read_fit(data, name, dataset, refs["train"])
        models[name] = fit_result_from_forest(**spec, validation_score=0).predictor
    return models


def weights(data, control, v1bundle):
    return {
        **{name: digest(data / "fitted" / f"{name}.npz") for name in MIX},
        V1: digest(v1bundle / "weights.npz"),
        native.CONTROL: digest(control / "weights.npz"),
    }


def validate(data, output, control, v1bundle):
    if native.read(output / "protocol.json") != protocol() or (output / "selection").exists():
        raise ValueError("Validation is closed or selection policy changed")
    inputs = input_digests(data)
    source = implementation()
    dataset, refs = native.development(data)
    cohort = native.PartitionCorpus(data, "validation", dataset)
    models = load_validation_models(data, control, v1bundle)
    scores, _ = native.measure(cohort, refs["validation"], models)
    if inputs != input_digests(data) or source != implementation():
        raise ValueError("Training inputs or implementation changed during native validation")
    write_new_file(
        output / "native-validation.json",
        json_bytes(
            {
                "scores": scores,
                "inputs": inputs,
                "implementation": source,
                "weights_sha256": weights(data, control, v1bundle),
                "protocol_sha256": digest(output / "protocol.json"),
            }
        ),
    )


def rank(scores, controlled):
    if set(scores) != set(ORDER) or set(controlled) != set(ORDER):
        raise ValueError("Both validation populations must include every registered candidate")
    rows = {}
    eligible = []
    base = native.CONTROL
    for name in ORDER:
        native_delta = scores[name]["interval_macro_f1"] - scores[base]["interval_macro_f1"]
        controlled_delta = (
            controlled[name]["interval"]["pooled"]["macro_f1"]
            - controlled[base]["interval"]["pooled"]["macro_f1"]
        )
        checks = gate(controlled[base], controlled[name])
        passes = (
            native_delta >= 0
            and checks["pooled"]
            and checks["stability"]
            and all(checks["per_class"].values())
        )
        rows[name] = {
            "native_gain": native_delta,
            "controlled_gain": controlled_delta,
            "controlled_checks": checks,
            "eligible": passes,
            "minimum_gain": min(native_delta, controlled_delta),
        }
        if passes and name != base:
            eligible.append(name)
    winner = (
        max(eligible, key=lambda name: scores[name]["interval_macro_f1"])
        if eligible
        else max(CHALLENGERS, key=lambda name: rows[name]["minimum_gain"])
    )
    return winner, bool(eligible), rows


def checked_decision(data, output, controlled, control, v1bundle):
    native_record = native.read(output / "native-validation.json")
    transfer = native.read(controlled / "validation.json")
    if (
        native_record["protocol_sha256"] != digest(output / "protocol.json")
        or native.read(output / "protocol.json") != protocol()
        or transfer["plan_sha256"] != digest(controlled / "plan.json")
        or native_record["inputs"] != input_digests(data)
        or native_record["implementation"] != implementation()
        or weights(data, control, v1bundle) != native_record["weights_sha256"]
        or native_record["weights_sha256"] != transfer["weights_sha256"]
    ):
        raise ValueError("Validation results do not share frozen model and population identities")
    scores = native_record["scores"]
    winner, eligible, rows = rank(scores, transfer["results"])
    return {
        "challenger": winner,
        "validation_winner": winner if eligible else native.CONTROL,
        "eligible_on_validation": eligible,
        "comparison": rows,
        "validation": scores,
        "native_validation_sha256": digest(output / "native-validation.json"),
        "controlled_validation_sha256": digest(controlled / "validation.json"),
        "controlled_plan_sha256": transfer["plan_sha256"],
        "control_bundle_sha256": native.BUNDLE_SHA,
        "protocol_sha256": native_record["protocol_sha256"],
        "inputs": native_record["inputs"],
    }


def freeze(data, output, controlled, control, v1bundle):
    decision = checked_decision(data, output, controlled, control, v1bundle)
    name = decision["challenger"]
    with output_directory(output / "selection") as stage:
        if name == V1:
            copytree(v1bundle, stage / "bundle")
            sha = V1_SHA
            AdaptiveBackend().load_bundle(
                stage / "bundle", manifest_sha256=sha, runtime=native.RUNTIME
            )
            verified = {"reused_bundle_sha256": V1_SHA}
        else:
            dataset, refs = native.development(data)
            spec = read_fit(data, name, dataset, refs["train"])
            sha, verified = native.package(
                stage,
                name,
                spec,
                decision["validation"][name]["interval_macro_f1"],
                dataset,
                refs,
                native.PartitionCorpus(data, "validation", dataset),
                model_version_id=f"direction-{name}-v2",
                additional_training_sources=(
                    AdditionalTrainingSource.model_validate(
                        spec["environment"]["additional_training_source"]
                    ),
                ),
                build_id="direction-mixed-" + digest(Path(__file__)),
            )
        write_json(
            stage / "freeze.json",
            {**decision, "bundle_sha256": sha, "round_trip_on_validation": verified},
        )
    package_candidates(data, output, mixed=True)
    print(
        f"{name} selected; eligible on joint validation: {decision['eligible_on_validation']}",
        flush=True,
    )


def score(data, output, controlled, control, v1bundle):
    decision = checked_decision(data, output, controlled, control, v1bundle)
    frozen = native.read(output / "selection" / "freeze.json")
    if any(frozen[key] != value for key, value in decision.items()):
        raise ValueError(
            "Frozen mixed direction selection differs from the joint validation results"
        )
    native.score_frozen(
        data,
        output,
        control,
        frozen,
        validation_qualified=decision["eligible_on_validation"],
        evidence=input_digests,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("register", "validate", "freeze", "score", "package-candidates")
    )
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--controlled", type=Path, default=CONTROLLED)
    parser.add_argument("--control", type=Path, default=native.BUNDLE)
    parser.add_argument("--v1-bundle", type=Path, default=V1_BUNDLE)
    args = parser.parse_args()
    if args.command == "package-candidates":
        package_candidates(args.data, args.output, mixed=True, explicit=True)
    elif args.command == "register":
        write_new_file(args.output / "protocol.json", json_bytes(protocol()))
    elif args.command == "validate":
        validate(args.data, args.output, args.control, args.v1_bundle)
    else:
        operation = freeze if args.command == "freeze" else score
        operation(args.data, args.output, args.controlled, args.control, args.v1_bundle)


if __name__ == "__main__":
    main()
