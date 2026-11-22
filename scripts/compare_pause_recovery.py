"""Compare recovery of price pauses against the report's shape-6 detector."""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from compare_trend_pauses import seeds as original_seeds
from confirm_pause_precision import (
    ROOT,
    candidate_module,
    group_draws,
    paired_intervals,
    summarize,
)
from confirm_pause_precision import (
    protocol as precision_protocol,
)
from replicate_pause_precision import FIRST_SEED, GROUPS, TCN_COHORTS, schedule_seeds
from trend_pause_cases import CONFIG, MARKETS, cases

from regimex.evaluation import matched_interval_indices
from regimex.pause_shape_confirmation import confirm_pauses
from regimex.storage import digest, json_bytes, write_new_file
from regimex.trend_pauses import scan_trend_pauses

CONTROL = "control"
FAMILIES = {"robust": "robust_pause_fit", "localize": "localize_pause"}
COHORTS = {
    "development": (2_011_300_000, 128),
    "final": (1_992_300_000, 320),
    "transfer": (1_962_100_000, 96),
}
MAX_FALSE = 0.50
DEV_FALSE_MARGIN = 0.04
FINAL_FALSE_MARGIN = 0.08
MAX_PRECISION_LOSS = 0.01


def groups(start, count):
    return [start + 2 * i for i in range(count)]


def protocol():
    previous = precision_protocol()
    sources = dict(previous["source_sha256"])
    for path in (
        "scripts/compare_pause_recovery.py",
        "scripts/replicate_pause_precision.py",
        "src/regimex/pause_shape_confirmation.py",
    ):
        sources[path] = digest(ROOT / path)
    return {
        "source_sha256": sources,
        "python": previous["python"],
        "numpy": previous["numpy"],
        "cohorts": {phase: groups(*span) for phase, span in COHORTS.items()},
        "schedule_offsets": previous["schedule_offsets"],
        "bootstrap": previous["bootstrap"],
        "analysis": previous["analysis"],
        "control": "shape-6",
        "families": FAMILIES,
        "development": "Maximize recall among candidates exceeding control recall with "
        f"slowdown false/1000 <= {MAX_FALSE} and <= control + {DEV_FALSE_MARGIN}. Break ties "
        "by lower false/1000, then grid order. If none qualify, score one diagnostic with "
        "smallest excess over the tighter false budget, then use the same tie order.",
        "confirmation": "Require development qualification, paired lower one-sided 95% "
        f"recall gain > 0, slowdown false/1000 <= {MAX_FALSE}, paired upper one-sided 95% "
        f"false/1000 difference <= {FINAL_FALSE_MARGIN} and pooled precision loss <= "
        f"{MAX_PRECISION_LOSS}. External transfer veto is evaluated separately.",
    }


def reserve(folder, inventories):
    previous = precision_protocol()
    old_groups = groups(FIRST_SEED, GROUPS)
    for phase, seeds in previous["seeds"].items():
        old_groups.extend(seeds)
        old_groups.extend(original_seeds(phase))
    for start in TCN_COHORTS:
        old_groups.extend(groups(start, 64))
    excluded = set(schedule_seeds(old_groups))
    fingerprints = {}
    for path in inventories:
        excluded.update(json.loads(path.read_bytes())["excluded_seeds"])
        fingerprints[path.as_posix()] = digest(path)
    plan = protocol()
    scheduled = [s for cohort in plan["cohorts"].values() for s in schedule_seeds(cohort)]
    if len(set(scheduled)) != len(scheduled) or not excluded.isdisjoint(scheduled):
        raise ValueError("Reserved schedule seeds overlap another cohort")
    write_new_file(
        folder / "plan.json",
        json_bytes(
            {"protocol": plan, "excluded_seeds_checked": len(excluded), "inventories": fingerprints}
        ),
    )


def methods():
    return [
        CONTROL,
        *(
            f"{family}/{key}"
            for family, module in FAMILIES.items()
            for key in candidate_module(module).SETTINGS
        ),
    ]


def predict(scan, method):
    if method == CONTROL:
        return confirm_pauses(scan, "shape-6")
    family, setting = method.split("/", 1)
    return candidate_module(FAMILIES[family]).detect(scan, setting)


def measure_group(index_seed, requested):
    index, seed = index_seed
    rows = []
    for case in cases(seed, index):
        scan = scan_trend_pauses(case.rows, case.parents, CONFIG)
        for method in requested:
            found = predict(scan, method)
            matches = matched_interval_indices(case.expected, found)
            rows.append(
                {
                    "group": seed,
                    "market": case.market,
                    "shape": case.shape,
                    "method": method,
                    "expected": len(case.expected),
                    "found": len(found),
                    "matched": len(matches),
                    "later_matched": sum(i > 0 for i in matches),
                    "onset_errors": [
                        abs(found[j].start - case.expected[i].start) for i, j in matches.items()
                    ],
                    "end_errors": [
                        abs(found[j].stop - case.expected[i].stop) for i, j in matches.items()
                    ],
                }
            )
    return rows


def evaluate(phase, requested, workers):
    items = enumerate(groups(*COHORTS[phase]))
    run = partial(measure_group, requested=requested)
    if workers == 1:
        batches = map(run, items)
        return [row for batch in batches for row in batch]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [row for batch in pool.map(run, items) for row in batch]


def metrics(rows):
    stats = summarize(rows)
    for key in ("onset_errors", "end_errors"):
        errors = [e for r in rows for e in r[key]]
        stats["median_absolute_" + key] = float(np.median(errors)) if errors else None
    return stats


def choose(options, scores):
    baseline = scores[CONTROL]
    limit = min(MAX_FALSE, baseline["false_per_1000"] + DEV_FALSE_MARGIN)
    candidates = [m for m in options if m != CONTROL]
    eligible = [
        m
        for m in candidates
        if scores[m]["recall"] > baseline["recall"] and scores[m]["false_per_1000"] <= limit
    ]
    winner = min(
        eligible or candidates,
        key=lambda m: (
            max(0, scores[m]["false_per_1000"] - limit),
            -scores[m]["recall"],
            scores[m]["false_per_1000"],
        ),
    )
    return {"winner": winner, "qualified": bool(eligible)}


def passes_final(qualified, baseline, candidate, paired):
    return (
        qualified
        and paired["lower_recall"] > 0
        and candidate["false_per_1000"] <= MAX_FALSE
        and paired["upper_false_per_1000"] <= FINAL_FALSE_MARGIN
        and candidate["precision"] is not None
        and baseline["precision"] is not None
        and candidate["precision"] >= baseline["precision"] - MAX_PRECISION_LOSS
    )


def fingerprint(folder):
    path = folder / "plan.json"
    if json.loads(path.read_bytes())["protocol"] != protocol():
        raise ValueError("The reserved protocol or source changed")
    return {
        "plan_sha256": digest(path),
        "candidate_sha256": {
            name: digest(Path(candidate_module(name).__file__)) for name in FAMILIES.values()
        },
        "methods": methods(),
    }


def develop(folder, workers):
    version = fingerprint(folder)
    if (folder / "freeze.json").exists() or (folder / "opened.json").exists():
        raise FileExistsError("Development already recorded")
    by_method = defaultdict(list)
    for row in evaluate("development", version["methods"], workers):
        by_method[row["method"]].append(row)
    scores = {m: metrics(rows) for m, rows in by_method.items()}
    if version != fingerprint(folder):
        raise ValueError("Candidate sources changed while scoring development")
    write_new_file(
        folder / "freeze.json",
        json_bytes(
            {
                **version,
                **choose(version["methods"], scores),
                "scores": scores,
            }
        ),
    )


def read_freeze(folder):
    frozen = json.loads((folder / "freeze.json").read_bytes())
    expected = fingerprint(folder)
    expected.update(choose(expected["methods"], frozen["scores"]))
    if any(frozen[k] != v for k, v in expected.items()):
        raise ValueError("The frozen candidate or development selection changed")
    return frozen


def confirm(folder, workers):
    frozen = read_freeze(folder)
    winner = frozen["winner"]
    sha = digest(folder / "freeze.json")
    write_new_file(folder / "opened.json", json_bytes({"freeze_sha256": sha}))
    by_method = defaultdict(list)
    for row in evaluate("final", [CONTROL, winner], workers):
        by_method[row["method"]].append(row)
    scores = {m: metrics(rows) for m, rows in by_method.items()}
    by_market = {
        m: {market: metrics([r for r in rows if r["market"] == market]) for market in MARKETS}
        for m, rows in by_method.items()
    }
    control, selected = (group_draws(by_method[m]) for m in (CONTROL, winner))
    paired = paired_intervals(control, selected)
    paired["lower_recall"] = float(np.quantile(selected["recall"] - control["recall"], 0.05))
    if digest(folder / "freeze.json") != sha or read_freeze(folder) != frozen:
        raise ValueError("The freeze changed while scoring confirmation")
    write_new_file(
        folder / "results.json",
        json_bytes(
            {
                "freeze_sha256": sha,
                "winner": winner,
                "scores": scores,
                "by_market": by_market,
                "paired": paired,
                "passed_primary": passes_final(
                    frozen["qualified"], scores[CONTROL], scores[winner], paired
                ),
                "rows": by_method,
            }
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("reserve", "develop", "confirm"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--exclude-seeds", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.phase == "reserve":
        reserve(args.output, args.exclude_seeds)
    elif args.phase == "develop":
        develop(args.output, args.workers)
    else:
        confirm(args.output, args.workers)


if __name__ == "__main__":
    main()
