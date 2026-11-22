"""Select a price-pause filter on development groups and confirm it once on unseen groups."""

import argparse
import json
import platform
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from compare_trend_pauses import counts
from compare_trend_pauses import protocol as previous_protocol
from trend_pause_cases import CONFIG, MARKETS, STEPS, cases

from regimex.evaluation import matched_interval_indices
from regimex.storage import digest, json_bytes, write_new_file
from regimex.trend_pauses import scan_trend_pauses

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {"shape": "pause_shape_confirmation", "flank": "pause_flank_confirmation"}
CONTROL = "report"
START = {"development": 2_113_500_000, "final": 2_104_700_000}
GROUPS = 128
BUDGET = 0.5
MIN_RECALL = 0.8
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 72851
# The simple alternative changes only the two thresholds already used by reports.
THRESHOLDS = {
    f"threshold/{gain}/{limit}": (gain, limit)
    for gain in (4, 8, 12, 16)
    for limit in (0.125, 0.25, 0.375, 0.5)
    if (gain, limit) != (4, 0.5)
}


def seeds(phase):
    return [START[phase] + 2 * i for i in range(GROUPS)]


def protocol():
    previous = previous_protocol()
    paths = set(previous["source_sha256"]) | {
        "scripts/confirm_pause_precision.py",
        "src/regimex/trend_pauses.py",
        "src/regimex/contracts.py",
        "src/regimex/observation_availability.py",
        "src/regimex/storage.py",
        ".python-version",
    }
    return {
        "source_sha256": {p: digest(ROOT / p) for p in sorted(paths)},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "seeds": {phase: seeds(phase) for phase in START},
        "schedule_offsets": previous["schedule_offsets"],
        "analysis": CONFIG.model_dump(mode="json"),
        "control": {"gain": 4, "absolute_limit": 0.5, "multiple": True, "tick_size": None},
        "alternatives": {**THRESHOLDS, **FAMILIES},
        "development": "Require one-sided 95% upper slowdown false/1000 <= 0.5 and "
        "recall >= 0.8. Maximize recall, then minimize false/1000, then use grid order. "
        "Retain control when no alternative qualifies.",
        "confirmation": "Keep one selected alternative and the threshold-only rule with the "
        "highest recall within the error budget (or the smallest upper bound if none qualify). "
        "Adopt the selection only if false/1000 <= 0.5, recall >= 0.8, "
        "one-sided paired 95% upper false/1000 difference < 0 and pooled precision increases.",
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "seed group"},
    }


def reserve(folder, exclusions):
    plan = protocol()
    old = previous_protocol()
    prior = {
        seed + offset
        for group_seeds in old["seeds"].values()
        for seed in group_seeds
        for offset in old["schedule_offsets"].values()
    }
    inventories = {}
    for path in exclusions:
        document = json.loads(path.read_bytes())
        prior.update(document["excluded_seeds"])
        inventories[path.as_posix()] = digest(path)
    new = [
        seed + offset
        for group_seeds in plan["seeds"].values()
        for seed in group_seeds
        for offset in plan["schedule_offsets"].values()
    ]
    if len(set(new)) != len(new) or not prior.isdisjoint(new):
        raise ValueError("Reserved schedule seeds overlap")
    write_new_file(
        folder / "plan.json",
        json_bytes(
            {
                "protocol": plan,
                "prior_seeds_checked": len(prior),
                "inventories": inventories,
            }
        ),
    )


def candidate_module(name):
    # A confirmed implementation may ship in the package using exactly the scored bytes.
    packaged = f"regimex.{name}"
    return import_module(packaged if find_spec(packaged) is not None else name)


def candidate_sources():
    return {name: digest(Path(candidate_module(name).__file__)) for name in FAMILIES.values()}


def methods():
    return [
        CONTROL,
        *THRESHOLDS,
        *(
            f"{family}/{setting}"
            for family, name in FAMILIES.items()
            for setting in candidate_module(name).SETTINGS
        ),
    ]


def predict(scan, requested):
    predictions = {}
    prepared = {}
    for method in requested:
        if method == CONTROL or method in THRESHOLDS:
            gain, limit = THRESHOLDS.get(method, (4, 0.5))
            found = scan.select(
                gain_multiplier=gain, absolute_multiplier=limit, absolute_limit=True, multiple=True
            )
        else:
            family, setting = method.split("/", 1)
            if family not in prepared:
                prepared[family] = candidate_module(FAMILIES[family]).prepare(scan)
            found = prepared[family].select(setting)
        predictions[method] = list(found)
    return predictions


def measure_group(index_seed, requested):
    index, seed = index_seed
    rows = []
    for case in cases(seed, index):
        scan = scan_trend_pauses(case.rows, case.parents, CONFIG)
        for method, found in predict(scan, requested).items():
            matched = matched_interval_indices(list(case.expected), found)
            rows.append(
                {
                    "group": seed,
                    "market": case.market,
                    "shape": case.shape,
                    "severity": case.severity,
                    "method": method,
                    "found": len(found),
                    "matched": len(matched),
                    "expected": len(case.expected),
                    "later_matched": sum(index > 0 for index in matched),
                }
            )
    return rows


def evaluate(phase, requested, workers):
    run = partial(measure_group, requested=requested)
    groups = list(enumerate(seeds(phase)))
    if workers == 1:
        return [row for group in groups for row in run(group)]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return [row for batch in executor.map(run, groups) for row in batch]


def summarize(rows):
    stats = counts(rows)
    found = sum(r["found"] for r in rows)
    stats.update(
        found=found,
        precision=stats["matched"] / found if found else None,
        recall=stats["matched"] / stats["expected"],
        false_per_1000=1000 * stats["slowdown_false"] / stats["slowdown_steps"],
        all_three=sum(r["expected"] == 3 and r["matched"] == 3 for r in rows),
        three_paths=sum(r["expected"] == 3 for r in rows),
    )
    return stats


def group_draws(rows):
    # Draw the same groups for all methods and pool counts before taking ratios.
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    columns = ("matched", "found", "expected", "slowdown_false", "slowdown_steps")
    summaries = (summarize(grouped[group]) for group in sorted(grouped))
    group_counts = np.array([[stats[name] for name in columns] for stats in summaries])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(len(group_counts), size=(BOOTSTRAP_DRAWS, len(group_counts)))
    matched, found, expected, false, steps = group_counts[sampled].sum(axis=1).T
    return {
        "precision": np.divide(matched, found, out=np.full(len(found), np.nan), where=found > 0),
        "recall": matched / expected,
        "false_per_1000": 1000 * false / steps,
    }


def development_summary(rows):
    result = {}
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    for method, subset in by_method.items():
        stats = summarize(subset)
        stats["upper_false_per_1000"] = float(
            np.quantile(group_draws(subset)["false_per_1000"], 0.95)
        )
        result[method] = stats
    return result


def choose(options, summary):
    eligible = [
        method
        for method in options
        if method != CONTROL
        and summary[method]["upper_false_per_1000"] <= BUDGET
        and summary[method]["recall"] >= MIN_RECALL
    ]
    return min(
        eligible,
        key=lambda m: (-summary[m]["recall"], summary[m]["false_per_1000"]),
        default=CONTROL,
    )


def plan_hash(folder):
    plan = folder / "plan.json"
    if json.loads(plan.read_bytes())["protocol"] != json.loads(json_bytes(protocol())):
        raise ValueError("The reserved protocol has changed")
    return digest(plan)


def develop(folder, workers):
    target = folder / "development.json"
    if target.exists() or (folder / "freeze.json").exists() or (folder / "opened.json").exists():
        raise FileExistsError("Development results have already been recorded")
    plan = plan_hash(folder)
    sources = candidate_sources()
    grid = methods()
    rows = evaluate("development", grid, workers)
    if plan != plan_hash(folder) or sources != candidate_sources():
        raise ValueError("Source files changed during development scoring")
    write_new_file(
        target,
        json_bytes(
            {
                "plan_sha256": plan,
                "candidates": sources,
                "methods": grid,
                "summary": development_summary(rows),
            }
        ),
    )


def selection(folder):
    path = folder / "development.json"
    development = json.loads(path.read_bytes())
    if (
        development["plan_sha256"] != plan_hash(folder)
        or development["methods"] != methods()
        or development["candidates"] != candidate_sources()
    ):
        raise ValueError("Development source files or settings have changed")
    selected = choose(development["methods"], development["summary"])
    stats = development["summary"]
    simple = min(
        THRESHOLDS,
        key=lambda m: (
            max(0.0, stats[m]["upper_false_per_1000"] - BUDGET),
            -stats[m]["recall"],
            stats[m]["false_per_1000"],
        ),
    )
    return {
        "plan_sha256": development["plan_sha256"],
        "development_sha256": digest(path),
        "candidates": development["candidates"],
        "selected": selected,
        "simple": simple,
        "finalists": list(dict.fromkeys((CONTROL, simple, selected))),
    }


def read_freeze(folder):
    frozen = json.loads((folder / "freeze.json").read_bytes())
    if frozen != selection(folder):
        raise ValueError("The freeze differs from the development selection")
    return frozen


def paired_intervals(control, candidate):
    differences = {}
    for key in control:
        delta = candidate[key] - control[key]
        finite = delta[np.isfinite(delta)]
        differences[key] = np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None
    differences["upper_false_per_1000"] = float(
        np.quantile(candidate["false_per_1000"] - control["false_per_1000"], 0.95)
    )
    return differences


def passes_final(control, candidate, paired):
    return (
        candidate["false_per_1000"] <= BUDGET
        and candidate["recall"] >= MIN_RECALL
        and paired["upper_false_per_1000"] < 0
        and candidate["precision"] is not None
        and candidate["precision"] > control["precision"]
    )


def confirm(folder, workers):
    frozen = read_freeze(folder)
    fingerprint = digest(folder / "freeze.json")
    write_new_file(folder / "opened.json", json_bytes({"freeze_sha256": fingerprint}))
    rows = evaluate("final", frozen["finalists"], workers)
    if read_freeze(folder) != frozen:
        raise ValueError("The freeze changed during confirmation")
    results, paired = {}, {}
    control_draws = group_draws([r for r in rows if r["method"] == CONTROL])
    for method in frozen["finalists"]:
        subset = [r for r in rows if r["method"] == method]
        by_market = {m: summarize([r for r in subset if r["market"] == m]) for m in MARKETS}
        severity = {}
        for level in ("near-flat", "moderate"):
            slow = [r for r in subset if r["severity"] == level and "slowdown" in r["shape"]]
            severity[level] = {"false": sum(r["found"] for r in slow), "steps": len(slow) * STEPS}
        results[method] = {"overall": summarize(subset), "market": by_market, "slowdowns": severity}
        if method != CONTROL:
            paired[method] = paired_intervals(control_draws, group_draws(subset))
    winner = frozen["selected"]
    adopted = winner != CONTROL and passes_final(
        results[CONTROL]["overall"], results[winner]["overall"], paired[winner]
    )
    write_new_file(
        folder / "results.json",
        json_bytes(
            {
                "freeze_sha256": fingerprint,
                "selected": winner,
                "adopted": adopted,
                "results": results,
                "paired": paired,
                "rows": rows,
            }
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("reserve", "develop", "freeze", "confirm"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--exclude-seeds", action="append", type=Path, default=[])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    folder = args.output
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.phase == "reserve":
        reserve(folder, args.exclude_seeds)
    elif args.phase == "develop":
        develop(folder, args.workers)
    elif args.phase == "freeze":
        frozen = selection(folder)
        write_new_file(folder / "freeze.json", json_bytes(frozen))
        print(json.dumps(frozen, indent=2))
    else:
        confirm(folder, args.workers)


if __name__ == "__main__":
    main()
