"""Score frozen pause classifiers on independent paths with analytically defined directions."""

import argparse
import json
from pathlib import Path

import numpy as np
from check_pause_target_transfer import examples
from compare_profile_21 import RUNTIME
from confirm_trend_pauses import CONFIG, STRESS
from refine_model_intervals import refine_decoded

from regimex.directions import DIRECTION_NAMES
from regimex.evaluation import _metrics, _segment_counts
from regimex.storage import digest, json_bytes, write_new_file

METHODS = ("control", "candidate", "hybrid")
BATCHES = 3
DRAW_OFFSET = 1400
DRAWS = 2000
DRAW_SEED = 467385
PAUSES = ("plateau-32", "plateau-48")


def protocol():
    dependencies = (
        "check_local_pause_transfer.py",
        "check_pause_target_transfer.py",
        "confirm_trend_pauses.py",
        "refine_model_intervals.py",
        "refine_trend_pauses.py",
    )
    return {
        "cases": {
            shape: [DRAW_OFFSET + index * BATCHES + b for b in range(BATCHES)]
            for index, shape in enumerate(STRESS)
        },
        "design": "32 independent paths per seed group, with both signs and two noise levels.",
        "arms": list(METHODS),
        "same_direction_pauses": list(PAUSES),
        "primary": "IoU-matched 32- and 48-step interruptions in same-direction trends.",
        "controls": "False sideways intervals and affected paths on a continuing slowdown.",
        "secondary": "Pooled interval macro-F1 and raw sideways wins within true pauses.",
        "bootstrap": {"draws": DRAWS, "seed": DRAW_SEED, "strata": "shape", "unit": "path"},
        "files": {name: digest(Path(__file__).with_name(name)) for name in dependencies},
    }


def evaluate(data, models):
    probabilities = {
        name: model.predict(data.inputs, runtime=RUNTIME) for name, model in models.items()
    }
    decoded = {name: data._decode(values)[1] for name, values in probabilities.items()}
    decoded["hybrid"] = tuple(
        refine_decoded(rows, intervals, CONFIG)
        for rows, intervals in zip(data.inputs.observations, decoded["control"], strict=True)
    )
    counts = []
    signal = []
    start = 0
    sideways = DIRECTION_NAMES.index("sideways")
    for idx, (ref, info) in enumerate(zip(data.references, data.inputs.info, strict=True)):
        counts.append([_segment_counts(ref, decoded[name][idx]) for name in METHODS])
        pauses = [r for r in ref if r.direction == "sideways" and r.stop - r.start < 64]
        for pause in pauses:
            wins = {}
            for name, values in probabilities.items():
                window = values[start + pause.start : start + pause.stop]
                wins[name] = int(np.count_nonzero(window.argmax(axis=1) == sideways))
            signal.append({"width": pause.stop - pause.start, "winning_steps": wins})
        start += info.step_count
    return np.array(counts), signal


def metric_rows(groups):
    totals = sum(val.sum(axis=0) for val in groups.values())
    holds = sum(groups[shape].sum(axis=0)[:, 0] for shape in PAUSES)
    slowdowns = groups["slowdown"][:, :, 0, 1]
    return {
        method: {
            "interval_macro_f1": _metrics(totals[i])["macro_f1"],
            "same_direction_matches": int(holds[i, 0]),
            "same_direction_references": int(holds[i, 0] + holds[i, 2]),
            "slowdown_false_sideways": int(slowdowns[:, i].sum()),
            "slowdowns_with_false_sideways": int(np.count_nonzero(slowdowns[:, i])),
        }
        for i, method in enumerate(METHODS)
    }


def paired(groups):
    rng = np.random.default_rng(DRAW_SEED)
    pooled = np.zeros((DRAWS, len(METHODS), 3, 3), dtype=np.int64)
    pause_counts = np.zeros((DRAWS, len(METHODS), 3), dtype=np.int64)
    for shape, rows in groups.items():
        draws = rng.integers(len(rows), size=(DRAWS, len(rows)))
        sample = rows[draws].sum(axis=1)
        pooled += sample
        if shape in PAUSES:
            pause_counts += sample[:, :, 0]
    tp, fp, fn = pooled.transpose(3, 0, 1, 2)
    denominator = 2 * tp + fp + fn
    f1 = np.divide(
        2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0
    ).mean(axis=2)
    recall = pause_counts[:, :, 0] / (pause_counts[:, :, 0] + pause_counts[:, :, 2])
    return {
        method: {
            key: np.quantile(metric[:, i] - metric[:, 0], [0.025, 0.975]).tolist()
            for key, metric in (("interval_macro_f1", f1), ("same_direction_recall", recall))
        }
        for i, method in enumerate(METHODS)
        if method != "control"
    }


def score(output, training, populations, previous):
    from fit_local_pause_features import load_predictors

    models = load_predictors(training, populations, previous)
    if set(models) != {"control", "candidate"}:
        raise ValueError("Both trained models are required")
    counts = {}
    signals = {}
    for shape, batches in protocol()["cases"].items():
        grouped, hits = [], []
        for index in batches:
            rows, wins = evaluate(examples(shape, index), models)
            grouped.append(rows)
            hits.extend(wins)
        counts[shape] = np.concatenate(grouped)
        signals[shape] = hits
    write_new_file(
        output / "results.json",
        json_bytes(
            {
                "plan_sha256": digest(output / "plan.json"),
                "training_freeze_sha256": digest(training / "fitted" / "freeze.json"),
                "methods": list(METHODS),
                "results": metric_rows(counts),
                "paired_95pct_vs_control": paired(counts),
                "raw_signals": signals,
                "path_counts": {shape: rows.tolist() for shape, rows in counts.items()},
            }
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    phases.add_parser("reserve").add_argument("output", type=Path)
    scoring = phases.add_parser("score")
    scoring.add_argument("output", type=Path)
    for name in ("training", "populations", "previous"):
        scoring.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    plan = args.output / "plan.json"
    if args.phase == "reserve":
        write_new_file(plan, json_bytes(protocol()))
    elif json.loads(plan.read_bytes()) != protocol():
        raise ValueError("The implementation or design differs from the reservation")
    else:
        score(args.output, args.training, args.populations, args.previous)
