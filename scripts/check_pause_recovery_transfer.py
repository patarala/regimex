"""Check a frozen pause policy on irregular plateaus downstream of the released TCN."""

import argparse
import json
from pathlib import Path

import numpy as np
from check_pause_overlay_on_tcn import BUNDLES, RUNTIME, observed_inputs
from trend_pause_cases import MARKETS, STEPS, Bounds, native_schedule, observe

from regimex.backends.tcn import TCNBackend
from regimex.directions import Direction
from regimex.evaluation import matched_interval_indices
from regimex.segmentation import decode
from regimex.storage import digest, json_bytes, write_new_file
from regimex.trend_pauses import scan_trend_pauses

FIRST_SEED = 1_962_100_000
GROUPS = 96
MODEL = "tcn-0.1.0"
SHAPES = ("pause", "slowdown", "pullback", "steady")


def plan():
    return {
        "groups": [FIRST_SEED + 2 * i for i in range(GROUPS)],
        "source_sha256": {
            p.name: digest(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("check_pause_overlay_on_tcn.py"),
                Path(__file__).with_name("trend_pause_cases.py"),
            )
        },
        "model_sha256": BUNDLES[MODEL],
        "runtime": RUNTIME.model_dump(mode="json"),
        "veto": {
            "minimum_recall_difference": -0.02,
            "maximum_extra_marks_per_1000_negative_steps": 0.1,
        },
    }


def paths(seed, index):
    rng = np.random.default_rng([seed, 2207])
    direction = (Direction.GROWTH, Direction.DECLINE)[index % 2]
    sign = 1 if direction == Direction.GROWTH else -1
    tick = (0.01, 0.05, 0.20)[index % 3]
    step = np.arange(STEPS)
    rates = (
        sign
        * rng.uniform(0.00024, 0.00054)
        * (1 + 0.32 * np.sin(step / rng.uniform(48, 96) + rng.uniform(-np.pi, np.pi)))
    )
    rates[0] = 0
    starts = (int(rng.integers(74, 166)), int(rng.integers(290, 370)))
    windows = [Bounds(s, s + int(rng.integers(32, 64))) for s in starts]
    noise = rng.standard_t(5, STEPS) * rng.uniform(0.00028, 0.00090)
    for i in range(1, STEPS):
        noise[i] += 0.45 * noise[i - 1]
    # Isolated price deviations are shared by every shape, including the controls.
    outliers = rng.choice(STEPS, 4, replace=False)
    noise[outliers] += rng.choice((-1, 1), 4) * 0.0025
    fractions = (0.0, rng.uniform(0.08, 0.28), -rng.uniform(0.12, 0.40), 1.0)
    for offset, (market, probability) in enumerate(MARKETS.items()):
        schedule = native_schedule(seed + offset, probability, tick)
        # Alternate sides at executions, so inactive bins do not advance the bounce.
        executions = np.cumsum([bool(row.trade_count) for row in schedule])
        bounce = (2 * (executions % 2) - 1) * 0.00028
        for shape, fraction in zip(SHAPES, fractions, strict=True):
            increments = rates.copy()
            for start, stop in windows:
                increments[start + 1 : stop] *= fraction
            prices = tick * np.rint(100 * np.exp(np.cumsum(increments) + noise + bounce) / tick)
            yield (
                market,
                shape,
                windows if shape == "pause" else [],
                direction,
                observe(schedule, prices),
            )


def score(folder, comparison, bundle):
    from compare_pause_recovery import CONTROL, predict, read_freeze

    reserved = json.loads((folder / "plan.json").read_bytes())
    if reserved != plan():
        raise ValueError("Transfer plan differs from its reservation")
    frozen = read_freeze(comparison)
    cohorts = json.loads((comparison / "plan.json").read_bytes())["protocol"]["cohorts"]
    if cohorts["transfer"] != reserved["groups"]:
        raise ValueError("The comparison reserved different transfer groups")
    winner = frozen["winner"]
    methods = list(dict.fromkeys((CONTROL, winner)))
    model = TCNBackend().load_bundle(bundle, manifest_sha256=BUNDLES[MODEL], runtime=RUNTIME)
    receipt = {
        "plan_sha256": digest(folder / "plan.json"),
        "freeze_sha256": digest(comparison / "freeze.json"),
        "methods": methods,
    }
    write_new_file(folder / "opened.json", json_bytes(receipt))
    records = []
    for index, seed in enumerate(reserved["groups"]):
        panel = list(paths(seed, index))
        inputs = observed_inputs([case[-1] for case in panel])
        probabilities = model.predict(inputs, runtime=RUNTIME).reshape(len(panel), STEPS, 3)
        for case, rows, values in zip(panel, inputs.observations, probabilities, strict=True):
            market, shape, expected, direction, _ = case
            parents = decode(values, rows, model.analysis)
            scan = scan_trend_pauses(rows, parents, model.analysis)
            for method in methods:
                marks = predict(scan, method)
                eligible = [m for m in marks if m.direction == direction]
                matches = matched_interval_indices(expected, eligible)
                records.append(
                    {
                        "group": seed,
                        "market": market,
                        "shape": shape,
                        "method": method,
                        "expected": len(expected),
                        "matched": len(matches),
                        "found": len(marks),
                    }
                )
    summary = {}
    for method in methods:
        subset = [r for r in records if r["method"] == method]
        negative = [r for r in subset if r["shape"] != "pause"]
        matched = sum(r["matched"] for r in subset)
        expected = sum(r["expected"] for r in subset)
        found = sum(r["found"] for r in subset)
        false = sum(r["found"] for r in negative)
        summary[method] = {
            "expected": expected,
            "matched": matched,
            "recall": matched / expected,
            "precision": matched / found if found else None,
            "negative_steps": len(negative) * STEPS,
            "false": false,
            "false_per_1000": 1000 * false / (len(negative) * STEPS),
            "false_by_shape": {
                shape: sum(r["found"] for r in negative if r["shape"] == shape)
                for shape in SHAPES
                if shape != "pause"
            },
        }
    control, selected = summary[CONTROL], summary[winner]
    passed = (
        selected["recall"] - control["recall"] >= reserved["veto"]["minimum_recall_difference"]
        and selected["false_per_1000"] - control["false_per_1000"]
        <= reserved["veto"]["maximum_extra_marks_per_1000_negative_steps"]
    )
    if (
        digest(comparison / "freeze.json") != receipt["freeze_sha256"]
        or digest(folder / "plan.json") != receipt["plan_sha256"]
        or reserved != plan()
        or frozen != read_freeze(comparison)
    ):
        raise ValueError("The transfer plan or selected detector changed during scoring")
    write_new_file(
        folder / "results.json",
        json_bytes({**receipt, "passed": passed, "summary": summary, "rows": records}),
    )
    print(json.dumps({"passed": passed, **summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    phases.add_parser("reserve").add_argument("output", type=Path)
    replay = phases.add_parser("score")
    replay.add_argument("output", type=Path)
    replay.add_argument("comparison", type=Path)
    replay.add_argument("--bundle", type=Path, default=Path("release/models/tcn"))
    args = parser.parse_args()
    if args.phase == "reserve":
        write_new_file(args.output / "plan.json", json_bytes(plan()))
    else:
        score(args.output, args.comparison, args.bundle)
