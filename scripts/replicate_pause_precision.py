"""Confirm one stricter pause hypothesis on an independent cohort after the first comparison."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from compare_trend_pauses import protocol as original_protocol
from confirm_pause_precision import (
    CONTROL,
    group_draws,
    measure_group,
    paired_intervals,
    passes_final,
    read_freeze,
    summarize,
)
from trend_pause_cases import MARKETS, STEPS

from regimex.storage import digest, json_bytes, write_new_file

CANDIDATE = "shape/shape-6"
GROUPS = 384
FIRST_SEED = 2_062_810_000
TCN_COHORTS = (2_087_600_000, 2_071_700_000)


def schedule_seeds(groups):
    return [seed + offset for seed in groups for offset in range(len(MARKETS))]


def reserve_protocol(previous):
    frozen = read_freeze(previous)
    paths = json.loads((previous / "plan.json").read_bytes())
    results = json.loads((previous / "results.json").read_bytes())
    if (
        results["freeze_sha256"] != digest(previous / "freeze.json")
        or CANDIDATE in frozen["finalists"]
        or results["adopted"]
    ):
        raise ValueError("Expected a completed comparison with this candidate still untested")
    excluded = set()
    for p, sha in paths["inventories"].items():
        path = Path(p)
        if digest(path) != sha:
            raise ValueError(f"Historical seed inventory changed: {path}")
        excluded.update(json.loads(path.read_bytes())["excluded_seeds"])
    for grouped in (original_protocol()["seeds"], paths["protocol"]["seeds"]):
        for seeds in grouped.values():
            excluded.update(schedule_seeds(seeds))
    for start in TCN_COHORTS:
        excluded.update(schedule_seeds([start + 2 * i for i in range(64)]))
    seeds = [FIRST_SEED + 2 * i for i in range(GROUPS)]
    schedules = schedule_seeds(seeds)
    if len(schedules) != len(set(schedules)) or not excluded.isdisjoint(schedules):
        raise ValueError("Fresh price-path groups overlap a previous schedule")
    return {
        "methods": [CONTROL, CANDIDATE],
        "group_seeds": seeds,
        "schedule_offsets": paths["protocol"]["schedule_offsets"],
        "excluded_tcn_groups": {"first_seeds": list(TCN_COHORTS), "groups_each": 64, "stride": 2},
        "prior_schedule_seeds_checked": len(excluded),
        "script_sha256": digest(Path(__file__)),
        "prior_results_sha256": digest(previous / "results.json"),
        "prior_freeze_sha256": results["freeze_sha256"],
        "candidates": frozen["candidates"],
        "bootstrap": paths["protocol"]["bootstrap"],
        "adoption": "Recall >= 0.8, false/1000 on slowdowns <= 0.5, higher pooled "
        "precision and a one-sided paired 95% upper false/1000 difference below zero.",
    }


def confirm(folder, previous, workers):
    plan_file = folder / "plan.json"
    plan = json.loads(plan_file.read_bytes())
    if plan != reserve_protocol(previous):
        raise ValueError("The reserved comparison changed")
    sha = digest(plan_file)
    write_new_file(folder / "opened.json", json_bytes({"plan_sha256": sha}))

    groups = enumerate(plan["group_seeds"])
    run = partial(measure_group, requested=plan["methods"])
    if workers == 1:
        rows = [row for group in groups for row in run(group)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = [row for batch in pool.map(run, groups) for row in batch]
    if sha != digest(plan_file) or plan != reserve_protocol(previous):
        raise ValueError("The cohort or implementations changed while scoring")

    results, draws = {}, {}
    for method in plan["methods"]:
        subset = [r for r in rows if r["method"] == method]
        draw = group_draws(subset)
        draws[method] = draw
        slowdowns = {}
        for level in ("near-flat", "moderate"):
            slow = [r for r in subset if "slowdown" in r["shape"] and r["severity"] == level]
            slowdowns[level] = {"false": sum(r["found"] for r in slow), "steps": len(slow) * STEPS}
        results[method] = {
            "overall": summarize(subset),
            "market": {m: summarize([r for r in subset if r["market"] == m]) for m in MARKETS},
            "slowdowns": slowdowns,
            "upper_false_per_1000": float(np.quantile(draw["false_per_1000"], 0.95)),
            "recall_95pct": np.quantile(draw["recall"], [0.025, 0.975]).tolist(),
        }
    paired = paired_intervals(draws[CONTROL], draws[CANDIDATE])
    adopted = passes_final(results[CONTROL]["overall"], results[CANDIDATE]["overall"], paired)
    write_new_file(
        folder / "results.json",
        json_bytes(
            {
                "plan_sha256": sha,
                "adopted": adopted,
                "results": results,
                "paired": paired,
                "rows": rows,
            }
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("reserve", "confirm"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.phase == "reserve":
        write_new_file(args.output / "plan.json", json_bytes(reserve_protocol(args.previous)))
    else:
        confirm(args.output, args.previous, args.workers)


if __name__ == "__main__":
    main()
