"""Select a direction forest on fresh validation experiments and confirm its intervals once."""

import argparse
from collections import Counter
from importlib import import_module
from pathlib import Path

import numpy as np
from build_exposure_index import build
from compare_direction_context import (
    load_evaluation,
    matrix,
    open_pool,
    probability_metrics,
    targets,
)
from compare_profile_21 import RUNTIME, Corpus, sha256, software, write
from compare_tcn_widths import PROFILE, generate
from compare_temporal_architectures import (
    DRAW_SEED,
    DRAWS,
    LENGTHS,
    experiment_counts,
    read,
    summary,
)
from confirm_tcn_widths import audit_population
from confirm_training_populations import contrast
from fit_pause_targets import DATASET

from regimex.admin_contracts import GenerationSpec, RawGenerationPlan
from regimex.backends.adaptive import AdaptiveBackend
from regimex.data_catalog import DatasetManifest, content_hash
from regimex.storage import json_bytes, output_directory, write_new_file

CONTROL = "adaptive-0.1.0"
BUNDLE_SHA = "8bf057911823ed1d149320de2d765f91937ddc4b7205c4356951d208a52b48a0"
MODULES = {"local": "local_adaptive_forest", "pieces": "adaptive_piece_evidence"}
CANDIDATES = (CONTROL, *MODULES)
POOLS = {"validation": (1_296_873_000, 20), "holdout": (3_244_126_000, 40)}
MARGINS = {"point_macro_f1": 0.02, "class_f1": 0.03, "length_f1": 0.03, "short_recall": 0.03}
VERIFICATION = Path("var/verification")
TRAINING = VERIFICATION / "training-populations/development/mixed-192"
BUNDLE = Path("var/release/adaptive-direction-0.1.0/bundle")


def specs(pool):
    seed, count = POOLS[pool]
    return tuple(
        GenerationSpec(
            dataset_id=f"adaptive-intervals-{pool}-{steps}",
            plan=RawGenerationPlan(seed=seed + steps, experiment_count=count, step_count=steps),
        )
        for steps in LENGTHS
    )


def protocol():
    return {
        "cohorts": {pool: [s.model_dump(mode="json") for s in specs(pool)] for pool in POOLS},
        "control": CONTROL,
        "bundle_sha256": BUNDLE_SHA,
        "training_dataset_sha256": DATASET,
        "candidates": list(CANDIDATES),
        "profile": PROFILE.model_dump(mode="json"),
        "runtime": RUNTIME.model_dump(mode="json"),
        "selection": "Highest validation interval macro-F1 at IoU 0.5, first candidate on ties.",
        "qualification": "A new winner must have a positive paired 95% lower bound for the "
        "pooled interval F1 gain and satisfy all regression margins against the release. "
        "Short sideways intervals are shorter than 64 steps and require positive support.",
        "maximum_loss": MARGINS,
        "bootstrap": {"draws": DRAWS, "seed": DRAW_SEED, "unit": "whole experiment within length"},
    }


def prior_exposure(previous, transfer, prior, additional):
    receipt = read(previous / "reservation.json")
    transfer_plan = read(transfer / "plan.json")
    external = sha256(prior)
    if transfer_plan["prior_sha256"] != external or transfer_plan[
        "native_reservation_sha256"
    ] != sha256(previous / "reservation.json"):
        raise ValueError("Previous transfer plan and native reservation have different histories")
    datasets = []
    for pool in POOLS:
        path = previous / pool / "dataset.json"
        manifest = DatasetManifest.model_validate_json(path.read_bytes())
        record = receipt["pools"][pool]
        if (
            record["dataset_sha256"] != manifest.content_sha256
            or record["external_inventory_sha256"] != external
        ):
            raise ValueError("Previous native population differs from its reservation")
        datasets.append(path)
    prior_index = build([*datasets, *additional], [prior])
    historical_groups = transfer_plan["protocol"]["groups"]
    prior_index["excluded_seeds"] = sorted(
        set(prior_index["excluded_seeds"]).union(historical_groups)
    )
    return prior_index


def reserve(output, training, prior, previous, transfer, additional=()):
    development = Corpus(training, expected_digest=DATASET)
    inventory = prior_exposure(previous, transfer, prior, additional)
    with output_directory(output) as stage:
        write(stage / "plan.json", protocol())
        write(stage / "prior.json", inventory)
        pools = {}
        first = None
        for pool, (_, count) in POOLS.items():
            directory = stage / pool
            directory.mkdir()
            catalog = generate(directory, specs(pool), f"adaptive-intervals-{pool}").dataset
            expected = {
                (size, instruments): count // 2 for size in LENGTHS for instruments in (1, 2)
            }
            members = Counter((m.step_count, len(m.instrument_ids)) for m in catalog.members)
            if members != expected:
                raise ValueError(
                    "Generated cohort differs from the registered length and instrument counts"
                )
            audit = audit_population(development.dataset, catalog, stage / "prior.json")
            if first is not None:
                audit_population(first, catalog, None)
            first = catalog
            pools[pool] = {**audit, "locations_sha256": sha256(directory / "locations.json")}
        write(
            stage / "reservation.json",
            {
                "plan_sha256": sha256(stage / "plan.json"),
                "prior_sha256": sha256(stage / "prior.json"),
                "training_locations_sha256": sha256(training / "locations.json"),
                "previous_reservation_sha256": sha256(previous / "reservation.json"),
                "previous_transfer_sha256": sha256(transfer / "plan.json"),
                "pools": pools,
            },
        )


def checked_reservation(output):
    receipt = read(output / "reservation.json")
    if (
        receipt["plan_sha256"] != content_hash(protocol())
        or sha256(output / "plan.json") != receipt["plan_sha256"]
        or sha256(output / "prior.json") != receipt["prior_sha256"]
    ):
        raise ValueError("Training history or the registered evaluation policy changed")
    return receipt


def fingerprint():
    # Imported evaluation helpers and both feature extractors are part of the frozen method.
    from ast import Import, ImportFrom, parse, walk

    directory = Path(__file__).resolve().parent
    pending = [Path(__file__).stem, *MODULES.values()]
    hashes = {}
    while pending:
        name = pending.pop()
        if name in hashes:
            continue
        source = directory / (name + ".py")
        hashes[name] = sha256(source)
        for node in walk(parse(source.read_bytes())):
            if isinstance(node, Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ImportFrom) and node.module:
                imports = [node.module]
            else:
                continue
            pending.extend(module for module in imports if (directory / (module + ".py")).is_file())
    return {**software(), "scripts": hashes}


def train(output, training):
    receipt = checked_reservation(output)
    if receipt["training_locations_sha256"] != sha256(training / "locations.json"):
        raise ValueError("Training export locations changed after reservation")
    identity = fingerprint()
    data = load_evaluation(Corpus(training, expected_digest=DATASET), ("train",))
    y = targets(data)
    y.setflags(write=False)
    with output_directory(output / "fitted") as stage:
        weights = {}
        for name, source in MODULES.items():
            module = import_module(source)
            x = matrix(module, data.inputs)
            model = module.fit(x, y)
            path = stage / (name + ".npz")
            model.save(path)
            restored = module.DirectionContextModel.load(path)
            np.testing.assert_allclose(
                restored.predict(x), model.predict(x), rtol=1e-12, atol=1e-12
            )
            weights[name] = sha256(path)
            print(f"Fitted {name} on {len(y)} rows with {x.shape[1]} features", flush=True)
            del x, model, restored
        if fingerprint() != identity or checked_reservation(output) != receipt:
            raise ValueError("Fit inputs changed during training")
        write(
            stage / "fit.json",
            {
                "reservation_sha256": sha256(output / "reservation.json"),
                "software": identity,
                "weights": weights,
                "labels_sha256": content_hash(y.tolist()),
                "rows": len(y),
                "classifiable_rows": int((y >= 0).sum()),
            },
        )


def check_fit(output):
    checked_reservation(output)
    folder = output / "fitted"
    fit_record = read(folder / "fit.json")
    if (
        fit_record["reservation_sha256"] != sha256(output / "reservation.json")
        or fit_record["software"] != fingerprint()
        or fit_record["weights"] != {name: sha256(folder / (name + ".npz")) for name in MODULES}
    ):
        raise ValueError("Frozen methods, fitted weights or reserved population changed")
    return fit_record


def load_predictors(output, names, control_bundle=BUNDLE):
    """Load the requested models once and expose only the observation input boundary."""
    names = tuple(dict.fromkeys(names))
    if not names or set(names) - set(CANDIDATES):
        raise ValueError("Unregistered direction classifier")
    control = None
    if CONTROL in names:
        control = AdaptiveBackend().load_bundle(
            control_bundle, manifest_sha256=BUNDLE_SHA, runtime=RUNTIME
        )
        if control.analysis != PROFILE:
            raise ValueError("Released forest uses a different analysis profile")
    learned = {}
    for name in names:
        if name in MODULES:
            module = import_module(MODULES[name])
            learned[name] = (
                module,
                module.DirectionContextModel.load(output / "fitted" / (name + ".npz")),
            )

    def predict(inputs):
        result = {CONTROL: control.predict(inputs, runtime=RUNTIME)} if control is not None else {}
        for name, (module, model) in learned.items():
            result[name] = model.predict(matrix(module, inputs))
        return result

    return predict


def measure(output, evaluation, names, control_bundle):
    probabilities = load_predictors(output, names, control_bundle)(evaluation.inputs)
    y = targets(evaluation)
    scores, groups = {}, {}
    for name, values in probabilities.items():
        group = experiment_counts(evaluation, values)
        _, lengths, counts = group
        groups[name] = group
        scores[name] = {
            **summary(counts),
            "by_length": {str(size): summary(counts[lengths == size]) for size in LENGTHS},
            "probabilities": probability_metrics(y, values),
        }
    return scores, groups


def choose(scores):
    return max(CANDIDATES, key=lambda name: scores[name]["interval_macro_f1"])


def freeze(output, control_bundle):
    if (output / "freeze.json").exists() or (output / "opened.json").exists():
        raise ValueError("Validation selection has already closed")
    receipt = checked_reservation(output)
    record = check_fit(output)
    validation = open_pool(output, receipt, "validation")
    scores, _ = measure(output, validation, CANDIDATES, control_bundle)
    if record != check_fit(output):
        raise ValueError("Fitted evidence changed during selection")
    selection = {
        "fit_sha256": sha256(output / "fitted/fit.json"),
        "winner": choose(scores),
        "scores": scores,
    }
    write_new_file(output / "freeze.json", json_bytes(selection))
    print(f"Selected {selection['winner']}", flush=True)


def read_freeze(output):
    check_fit(output)
    selection = read(output / "freeze.json")
    if selection["fit_sha256"] != sha256(output / "fitted/fit.json") or selection[
        "winner"
    ] != choose(selection["scores"]):
        raise ValueError("Frozen winner differs from the fitted validation evidence")
    return selection


def qualification(control, winner, paired):
    reference = control["short_sideways_recall"]
    candidate = winner["short_sideways_recall"]
    return {
        "positive_interval_gain": paired["interval_macro_f1"]["interval_95pct"][0] > 0,
        "point_f1": winner["point_macro_f1"] is not None
        and control["point_macro_f1"] is not None
        and winner["point_macro_f1"] >= control["point_macro_f1"] - MARGINS["point_macro_f1"],
        "class_f1": all(
            winner["interval_per_class"][k]["f1"] >= v["f1"] - MARGINS["class_f1"]
            for k, v in control["interval_per_class"].items()
        ),
        "length_f1": all(
            winner["by_length"][k]["interval_macro_f1"]
            >= v["interval_macro_f1"] - MARGINS["length_f1"]
            for k, v in control["by_length"].items()
        ),
        "short_sideways_recall": reference is not None
        and candidate is not None
        and candidate >= reference - MARGINS["short_recall"],
    }


def score(output, control_bundle):
    selection = read_freeze(output)
    winner = selection["winner"]
    if winner == CONTROL:
        raise ValueError("The released model won validation; the final population remains reserved")
    opening = {"freeze_sha256": sha256(output / "freeze.json")}
    write_new_file(output / "opened.json", json_bytes(opening))
    holdout = open_pool(output, checked_reservation(output), "holdout")
    scores, groups = measure(output, holdout, (CONTROL, winner), control_bundle)
    paired = contrast(groups[CONTROL], groups[winner])
    checks = qualification(scores[CONTROL], scores[winner], paired)
    if read_freeze(output) != selection or opening["freeze_sha256"] != sha256(
        output / "freeze.json"
    ):
        raise ValueError("Frozen winner changed during confirmation")
    write_new_file(
        output / "results.json",
        json_bytes(
            {
                **opening,
                "scores": scores,
                "winner": winner,
                "paired_vs_control": paired,
                "checks": checks,
                "qualified": all(checks.values()),
                "experiment_counts": {
                    name: dict(zip(ids, rows.tolist(), strict=True))
                    for name, (ids, _, rows) in groups.items()
                },
            }
        ),
    )
    print(f"Confirmed {winner}: {checks}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase in ("reserve", "train", "freeze", "score"):
        command = phases.add_parser(phase)
        command.add_argument("output", type=Path)
        if phase in {"reserve", "train"}:
            command.add_argument("--training", type=Path, default=TRAINING)
        if phase == "reserve":
            command.add_argument(
                "--prior", type=Path, default=VERIFICATION / "direction-context-prior.json"
            )
            command.add_argument(
                "--previous", type=Path, default=VERIFICATION / "direction-context"
            )
            command.add_argument(
                "--previous-transfer",
                type=Path,
                default=VERIFICATION / "direction-context-transfer",
            )
            command.add_argument("--known-dataset", type=Path, action="append", default=[])
        if phase in {"freeze", "score"}:
            command.add_argument("--control-bundle", type=Path, default=BUNDLE)
    args = parser.parse_args()
    if args.phase == "reserve":
        reserve(
            args.output,
            args.training,
            args.prior,
            args.previous,
            args.previous_transfer,
            args.known_dataset,
        )
    elif args.phase == "train":
        train(args.output, args.training)
    elif args.phase == "freeze":
        freeze(args.output, args.control_bundle)
    else:
        score(args.output, args.control_bundle)


if __name__ == "__main__":
    main()
