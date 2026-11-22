"""Compare a frozen pause annotator with known directions on previously unused price paths."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from check_annotation_penalties import FAMILIES, INSTRUMENT, NOISE, PathGrid, case

from regimex.contracts import Observation
from regimex.directions import AnalysisConfig, Direction, DirectionInterval, summarize_interval
from regimex.evaluation import _metrics, _segment_counts
from regimex.segmentation import annotate
from regimex.storage import digest, json_bytes, write_new_file

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = Path(__file__).with_name("refine_trend_pauses.py")
STRESS = (
    "flat",
    "level-shift",
    "steady-trend",
    "slowdown",
    "reversal",
    "plateau-32",
    "plateau-48",
    "pause-before-reversal",
)
SEED = 731984062
REPEATS = 16
DRAWS = 2000
METHODS = ("current", "lower-penalty", "refined")
CONFIG = AnalysisConfig(version="2.2")


def protocol():
    files = (
        "scripts/check_annotation_penalties.py",
        "scripts/confirm_trend_pauses.py",
        "src/regimex/directions.py",
        "src/regimex/segmentation.py",
        "src/regimex/evaluation.py",
        "uv.lock",
    )
    return {
        "inputs": {path: digest(ROOT / path) for path in files},
        "numpy": np.__version__,
        "seed": SEED,
        "repeats_per_sign_and_noise": REPEATS,
        "families": {"replication": list(FAMILIES), "stress": list(STRESS)},
        "methods": list(METHODS),
        "profile": CONFIG.model_dump(mode="json"),
        "primary": "Interval macro-F1 and recall of pauses shorter than 64 steps.",
        "controls": (
            "False trends on flat and shifted paths, false sideways intervals on continuous "
            "trends, slowdowns and reversals."
        ),
        "bootstrap": {
            "draws": DRAWS,
            "seed": 148021,
            "unit": "whole path",
            "strata": "suite and shape",
        },
    }


def stress_case(family, sign, sigma, rng):
    size = int(rng.choice([224, 320]))
    middle = int(rng.integers(size * 35 // 100, size * 65 // 100))
    width = {"plateau-32": 32, "plateau-48": 48, "pause-before-reversal": 40}.get(family, 48)
    start, stop = middle - width // 2, middle + width // 2
    up = Direction.GROWTH if sign == 1 else Direction.DECLINE
    down = Direction.DECLINE if sign == 1 else Direction.GROWTH
    flat = Direction.SIDEWAYS
    slope = sign * rng.uniform(0.00012, 0.00029)
    rate = np.full(size, slope)
    shift = np.zeros(size)
    if family in ("flat", "level-shift"):
        rate.fill(0)
        if family == "level-shift":
            shift[middle:] = sign * rng.uniform(0.007, 0.02)
        spans = [(0, size, flat)]
    elif family == "reversal":
        rate[middle:] *= -rng.uniform(0.8, 1.25)
        spans = [(0, middle, up), (middle, size, down)]
    elif family.startswith("plateau-") or family == "pause-before-reversal":
        rate[start:stop] = 0
        reverse = family == "pause-before-reversal"
        rate[stop:] *= (-1 if reverse else 1) * rng.uniform(0.8, 1.25)
        spans = [(0, start, up), (start, stop, flat), (stop, size, down if reverse else up)]
    else:
        if family == "slowdown":
            # Even the slower section sustains over 0.5% movement across 48 steps.
            rate.fill(sign * 0.00034)
            rate[start:stop] = sign * 0.00016
        spans = [(0, size, up)]
    clean = np.r_[0.0, np.cumsum(rate[1:])] + shift
    phi = rng.uniform(0.45, 0.82)
    noise = rng.standard_t(5, size=size) * np.sqrt(3 / 5)
    for i in range(1, size):
        noise[i] = phi * noise[i - 1] + np.sqrt(1 - phi**2) * noise[i]
    tick = rng.choice([0.01, 0.05])
    prices = np.round(100 * np.exp(clean + sigma * noise) / tick) * tick
    rows = tuple(
        Observation(
            instrument_id=INSTRUMENT.instrument_id,
            step=i,
            last_price=price,
            trade_count=1,
            trade_volume=1,
            buy_order_count=None,
            buy_order_volume=None,
            sell_order_count=None,
            sell_order_volume=None,
            best_bid=None,
            best_ask=None,
        )
        for i, price in enumerate(prices)
    )
    references = tuple(
        DirectionInterval(
            instrument_id=INSTRUMENT.instrument_id,
            start=a,
            stop=b,
            status="observed",
            direction=direction,
            statistics=summarize_interval(rows, a, b),
        )
        for a, b, direction in spans
    )
    return PathGrid("path", rows, step_count=size), references


def groups():
    for suite_index, (suite, families) in enumerate(
        (("replication", FAMILIES), ("stress", STRESS))
    ):
        for family_index, family in enumerate(families):
            cases = []
            for noise_index, sigma in enumerate(
                NOISE if suite == "replication" else (0.0005, 0.0014)
            ):
                for sign in (-1, 1):
                    for repeat in range(REPEATS):
                        seed = np.random.SeedSequence(
                            [SEED, suite_index, family_index, noise_index, sign + 1, repeat]
                        )
                        builder = case if suite == "replication" else stress_case
                        grid, reference = builder(family, sign, sigma, np.random.default_rng(seed))
                        # The prior generator uses the shape name as its experiment identifier.
                        cases.append((replace(grid, experiment_id="path"), reference))
            yield suite, family, cases


def scores(groups_by_shape):
    pools = np.sum([x.sum(axis=0) for x in groups_by_shape.values()], axis=0)
    pause_groups = [
        x for name, x in groups_by_shape.items() if "plateau" in name or "pause" in name
    ]
    pauses = np.sum([x.sum(axis=0) for x in pause_groups], axis=0)[:, 0]
    result = {}
    for i, method in enumerate(METHODS):
        steady = [x for key, x in groups_by_shape.items() if key in {"flat", "level-shift"}]
        trends = [
            x
            for key, x in groups_by_shape.items()
            if key in {"slow-trend", "steady-trend", "slowdown", "reversal"}
        ]
        matched, _, missed = pauses[i].tolist()
        result[method] = {
            "interval_macro_f1": _metrics(pools[i])["macro_f1"],
            "short_sideways_matched": matched,
            "short_sideways_references": matched + missed,
            "false_trends_on_level_paths": int(sum(x[:, i, 1:, 1].sum() for x in steady)),
            "false_sideways_on_trends": int(sum(x[:, i, 0, 1].sum() for x in trends)),
        }
    return result


def paired_bounds(groups_by_shape):
    # Resample the same paths for both methods, retaining each shape's frequency.
    rng = np.random.default_rng(148021)
    draws = np.zeros((DRAWS, len(METHODS), 3, 3), dtype=np.int64)
    pauses = np.zeros((DRAWS, len(METHODS), 3), dtype=np.int64)
    for name, counts in groups_by_shape.items():
        indexes = rng.integers(len(counts), size=(DRAWS, len(counts)))
        sampled = counts[indexes].sum(axis=1)
        draws += sampled
        if "plateau" in name or "pause" in name:
            pauses += sampled[:, :, 0]
    tp, fp, fn = draws.transpose(3, 0, 1, 2)
    denom = 2 * tp + fp + fn
    macro = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0).mean(
        axis=2
    )
    recall = pauses[:, :, 0] / (pauses[:, :, 0] + pauses[:, :, 2])
    return {
        key: np.quantile(value[:, 2] - value[:, 0], [0.025, 0.975]).tolist()
        for key, value in (("interval_macro_f1", macro), ("short_sideways_recall", recall))
    }


def score(folder):
    from refine_trend_pauses import annotate_pauses

    receipt = json.loads((folder / "freeze.json").read_bytes())
    if receipt != {"protocol": protocol(), "candidate_sha256": digest(CANDIDATE)}:
        raise ValueError("The design or candidate differs from the freeze")
    suites = {"replication": {}, "stress": {}}
    lower_penalty = CONFIG.model_copy(update={"boundary_penalty": 4.0})
    for suite, family, cases in groups():
        pairs = []
        for grid, expected in cases:
            variants = (
                annotate(grid, CONFIG),
                annotate(grid, lower_penalty),
                annotate_pauses(grid, CONFIG),
            )
            pairs.append([_segment_counts(expected, variant.intervals) for variant in variants])
        suites[suite][family] = np.asarray(pairs, dtype=np.int64)
    output = {"freeze_sha256": digest(folder / "freeze.json"), "suites": {}}
    for suite, counts in suites.items():
        output["suites"][suite] = {
            "scores": scores(counts),
            "paired_95pct_refined_minus_current": paired_bounds(counts),
            "counts_by_shape": {
                name: {method: rows[:, i].sum(axis=0).tolist() for i, method in enumerate(METHODS)}
                for name, rows in counts.items()
            },
            "counts_by_path": {name: rows.tolist() for name, rows in counts.items()},
        }
    write_new_file(folder / "results.json", json_bytes(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("reserve", "freeze", "score"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan = args.output / "plan.json"
    if args.phase == "reserve":
        write_new_file(plan, json_bytes(protocol()))
    elif json.loads(plan.read_bytes()) != protocol():
        raise ValueError("The evaluation plan differs from its reservation")
    elif args.phase == "freeze":
        write_new_file(
            args.output / "freeze.json",
            json_bytes({"protocol": protocol(), "candidate_sha256": digest(CANDIDATE)}),
        )
    else:
        score(args.output)
