"""Select a fixed mixture of two fitted forests across native and controlled validation."""

import argparse
from pathlib import Path

import evaluate_large_direction as native
import numpy as np
from check_adaptive_gap_transfer import GATES, gate
from evaluate_mixed_direction import V1, V1_BUNDLE, V1_SHA

from regimex.application.analysis_computation import snapshot_inputs
from regimex.backends.adaptive import AdaptiveBackend
from regimex.data_catalog import analysis_profile_hash, content_hash
from regimex.model_bundles import ModelBundle, ParentModelEvidence, TrainingEvidence
from regimex.model_contracts import BundleProvenance
from regimex.storage import digest, json_bytes, output_directory, write_json, write_new_file

DATA = Path("var/verification/direction-ensemble-v3")
OUTPUT = Path("var/verification/direction-ensemble-evaluation-v3")
CONTROLLED = Path("var/verification/direction-ensemble-transfer-v3")
FRACTIONS = {"blend-125": 0.125, "blend-250": 0.25, "blend-375": 0.375, "blend-500": 0.50}
PARENTS = {native.CONTROL: native.BUNDLE_SHA, V1: V1_SHA}
ORDER = (*FRACTIONS, *PARENTS)


def protocol():
    return {
        "parents": PARENTS,
        "fractions": FRACTIONS,
        "tie_order": list(FRACTIONS),
        "formula": "p = (1 - f) * release + f * multiscale-balanced; decode once",
        "selection": "Select the highest native interval macro-F1 among mixtures with strictly "
        "positive native gain against the release and passing all controlled validation margins. "
        "Stop before either final if no mixture qualifies.",
        "native_gate": native.protocol()["native_gate"],
        "native_maximum_loss": native.MARGINS,
        "controlled_margins": GATES,
        "bootstrap": native.protocol()["bootstrap"],
    }


def validation_probabilities(control, native_bundle):
    backend = AdaptiveBackend()
    greedy = backend.load_bundle(control, manifest_sha256=native.BUNDLE_SHA, runtime=native.RUNTIME)
    multiscale = backend.load_bundle(native_bundle, manifest_sha256=V1_SHA, runtime=native.RUNTIME)

    def predict(inputs):
        a = greedy.predict(inputs, runtime=native.RUNTIME)
        b = multiscale.predict(inputs, runtime=native.RUNTIME)
        return {
            native.CONTROL: a,
            V1: b,
            **{name: (1 - f) * a + f * b for name, f in FRACTIONS.items()},
        }

    return predict


def input_digests(data):
    paths = [
        "candidate-plan.json",
        "prior.json",
        "population.json",
        "reservation.json",
        "dataset.json",
    ]
    for part in ("train", "validation", "test"):
        paths.extend(f"{part}/{name}.json" for name in ("reservation", "locations", "dataset"))
        if part != "test":
            paths.append(f"{part}/references.json")
    return {path: digest(data / path) for path in paths}


def validate(data, output, control, native_bundle):
    plan = output / "protocol.json"
    if native.read(plan) != protocol() or (output / "selection").exists():
        raise ValueError("Ensemble validation is closed or its protocol changed")
    inputs = input_digests(data)
    dataset, refs = native.development(data)
    corpus = native.PartitionCorpus(data, "validation", dataset)
    scores, _ = native.measure_predictions(
        corpus, refs["validation"], ORDER, validation_probabilities(control, native_bundle)
    )
    if inputs != input_digests(data):
        raise ValueError("Registered data changed during ensemble selection")
    write_new_file(
        output / "native-validation.json",
        json_bytes(
            {
                "protocol_sha256": digest(plan),
                "parents": PARENTS,
                "fractions": FRACTIONS,
                "inputs": inputs,
                "scores": scores,
                "dataset_sha256": dataset.content_sha256,
                "metric_code_sha256": digest(Path(native.__file__)),
            }
        ),
    )


def select(scores, transfer):
    if set(scores) != set(ORDER) or set(transfer) != set(ORDER):
        raise ValueError("Validation is missing a registered mixture or parent")
    checks = {}
    for name in FRACTIONS:
        guards = gate(transfer[native.CONTROL], transfer[name])
        gain = scores[name]["interval_macro_f1"] - scores[native.CONTROL]["interval_macro_f1"]
        checks[name] = {
            "native_gain": gain,
            "controlled_checks": guards,
            "eligible": gain > 0
            and guards["pooled"]
            and guards["stability"]
            and all(guards["per_class"].values()),
        }
    eligible = [name for name in FRACTIONS if checks[name]["eligible"]]
    return max(eligible, key=lambda n: scores[n]["interval_macro_f1"]) if eligible else None, checks


def decision(data, output, controlled):
    validation = output / "native-validation.json"
    observations = native.read(validation)
    transfer_path = controlled / "validation.json"
    transfer = native.read(transfer_path)
    binding_path = controlled / "population-binding.json"
    binding = native.read(binding_path)
    if (
        native.read(output / "protocol.json") != protocol()
        or observations["protocol_sha256"] != digest(output / "protocol.json")
        or observations["inputs"] != input_digests(data)
        or observations["metric_code_sha256"] != digest(Path(native.__file__))
        or transfer["plan_sha256"] != digest(controlled / "plan.json")
        or observations["parents"] != PARENTS
        or transfer["parents"] != PARENTS
        or observations["fractions"] != FRACTIONS
        or transfer["fractions"] != FRACTIONS
        or observations["dataset_sha256"] != binding["dataset_sha256"]
        or binding["validation_sha256"] != digest(transfer_path)
        or binding["reservation_sha256"] != digest(data / "reservation.json")
        or binding["plan_sha256"] != transfer["plan_sha256"]
    ):
        raise ValueError("Controlled and native ensemble validations have incompatible identities")
    name, checks = select(observations["scores"], transfer["results"])
    return {
        "validation_winner": name or native.CONTROL,
        "challenger": name,
        "checks": checks,
        "validation": observations["scores"],
        "inputs": observations["inputs"],
        "parents": PARENTS,
        "control_bundle_sha256": native.BUNDLE_SHA,
        "native_validation_sha256": digest(validation),
        "controlled_validation_sha256": digest(transfer_path),
        "controlled_population_binding_sha256": digest(binding_path),
        "controlled_plan_sha256": transfer["plan_sha256"],
        "protocol_sha256": observations["protocol_sha256"],
    }


def package(stage, name, score, dataset, refs, corpus, control, native_bundle):
    from regimex.backends.adaptive_blend import blend_environment, pack_bundle

    models = tuple(
        ModelBundle.model_validate_json((bundle / "manifest.json").read_bytes())
        for bundle in (control, native_bundle)
    )
    fraction = FRACTIONS[name]
    main = models[1].training
    provenance = BundleProvenance(
        model_version_id=f"direction-{name}-v3",
        build_id="direction-ensemble-" + digest(Path(__file__)),
        analysis_profile="direction-32/2.2",
        analysis_profile_sha256=analysis_profile_hash(native.PROFILE),
        environment_sha256=content_hash(blend_environment(native.BUNDLE_SHA, V1_SHA, fraction)),
        training=TrainingEvidence(
            run_id=f"direction-{name}-v3",
            dataset_sha256=main.dataset_sha256,
            split_manifest_sha256=main.split_manifest_sha256,
            labels=main.labels,
            validation_references=refs["validation"].identity(),
            validation_dataset_sha256=dataset.content_sha256,
            selected_seed=main.selected_seed,
            seeds=main.seeds,
            selection_metric=main.selection_metric,
            validation_segment_macro_f1=score,
            parent_models=tuple(
                ParentModelEvidence.from_bundle(model, sha)
                for model, sha in zip(models, PARENTS.values(), strict=True)
            ),
        ),
    )
    dest = stage / "bundle"
    sha = pack_bundle(
        dest,
        greedy_bundle=control,
        greedy_sha256=native.BUNDLE_SHA,
        multiscale_bundle=native_bundle,
        multiscale_sha256=V1_SHA,
        fraction=fraction,
        provenance=provenance,
    )
    restored = AdaptiveBackend().load_bundle(dest, manifest_sha256=sha, runtime=native.RUNTIME)
    pure = validation_probabilities(control, native_bundle)
    verified = {}
    for member in corpus.members:
        key = f"{member.step_count}x{len(member.instrument_ids)}"
        if key not in verified:
            inputs = snapshot_inputs(corpus.snapshot(member).export)
            np.testing.assert_array_equal(
                restored.predict(inputs, runtime=native.RUNTIME), pure(inputs)[name]
            )
            verified[key] = {"export_sha256": member.export_sha256, "rows": inputs.row_count}
    return sha, verified


def freeze(data, output, controlled, control, native_bundle):
    result = decision(data, output, controlled)
    name = result["challenger"]
    with output_directory(output / "selection") as stage:
        if name:
            dataset, refs = native.development(data)
            result["bundle_sha256"], result["round_trip_on_validation"] = package(
                stage,
                name,
                result["validation"][name]["interval_macro_f1"],
                dataset,
                refs,
                native.PartitionCorpus(data, "validation", dataset),
                control,
                native_bundle,
            )
        write_json(stage / "freeze.json", result)
    print(f"Joint validation choice: {result['validation_winner']}", flush=True)


def score(data, output, controlled, control):
    result = decision(data, output, controlled)
    if result["challenger"] is None:
        raise ValueError("No ensemble qualified on validation; final populations remain reserved")
    frozen = native.read(output / "selection" / "freeze.json")
    if any(frozen[key] != value for key, value in result.items()):
        raise ValueError("Final ensemble differs from joint validation")
    native.score_frozen(
        data, output, control, frozen, validation_qualified=True, evidence=input_digests
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "validate", "freeze", "score"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--controlled", type=Path, default=CONTROLLED)
    parser.add_argument("--control", type=Path, default=native.BUNDLE)
    parser.add_argument("--v1-bundle", type=Path, default=V1_BUNDLE)
    args = parser.parse_args()
    if args.command == "register":
        write_new_file(args.output / "protocol.json", json_bytes(protocol()))
    elif args.command == "validate":
        validate(args.data, args.output, args.control, args.v1_bundle)
    elif args.command == "freeze":
        freeze(args.data, args.output, args.controlled, args.control, args.v1_bundle)
    else:
        score(args.data, args.output, args.controlled, args.control)


if __name__ == "__main__":
    main()
