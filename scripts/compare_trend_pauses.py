"""Calibrate trend pause detectors, freeze the report choice, then score new groups."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from refine_trend_pauses import _pause
from trend_pause_cases import CONFIG, MARKETS, STEPS, Bounds, cases

from regimex.evaluation import matched_interval_indices
from regimex.segmentation import _log_prices
from regimex.storage import digest, json_bytes, write_new_file

ROOT = Path(__file__).resolve().parents[1]
DETECTOR = ROOT / "src/regimex/trend_pauses.py"
STARTS = {"development": 2_137_810_000, "final": 2_149_620_000}
GROUPS = {"development": 32, "final": 64}
BUDGET = 0.5  # Maximum detections per 1,000 steps on slowdown controls.
GAINS = (*range(2, 65, 2), 72, 80, 96, 128, 160, 192, 256, 512, 1024)
ABSOLUTE_GAINS = (*range(2, 17), 20, 24, 32, 48, 64, 96, 128, 256)
LIMITS = (0.5, 1.0, 2.0)


@dataclass(frozen=True)
class Method:
    name: str
    gain: int = 4
    limit: float | None = None
    known_tick: bool = False


def grid():
    return (
        [Method("original")]
        + [Method(f"single-{gain}", gain) for gain in GAINS]
        + [Method(f"relative-{gain}", gain) for gain in GAINS]
        + [
            Method(f"absolute-{gain}-{limit}", gain, limit)
            for gain in ABSOLUTE_GAINS
            for limit in LIMITS
        ]
    )


def seeds(phase):
    return [STARTS[phase] + 2 * group for group in range(GROUPS[phase])]


def protocol():
    paths = (
        "scripts/compare_trend_pauses.py",
        "scripts/refine_trend_pauses.py",
        "scripts/trend_pause_cases.py",
        "src/regimex/directions.py",
        "src/regimex/evaluation.py",
        "src/regimex/exports.py",
        "src/regimex/market.py",
        "src/regimex/market_config.py",
        "src/regimex/segmentation.py",
        "uv.lock",
    )
    return {
        "source_sha256": {path: digest(ROOT / path) for path in paths},
        "numpy": np.__version__,
        "seeds": {phase: seeds(phase) for phase in STARTS},
        "schedule_offsets": {name: idx for idx, name in enumerate(MARKETS)},
        "config": CONFIG.model_dump(mode="json"),
        "grid": [asdict(method) for method in grid()],
        "development_budget_per_1k_slowdown_steps": BUDGET,
        "selection": "Highest matched pause count within the budget, then fewest slowdown "
        "detections, then fewest other unmatched detections, then grid order. If no setting "
        "passes, report the least-false setting and withhold release.",
        "bootstrap": {"draws": 2000, "seed": 19702, "unit": "whole paired group"},
    }


def reserve(folder, exclude):
    previous = set()
    for path in exclude:
        document = json.loads(path.read_bytes())
        if "excluded_seeds" in document:
            previous.update(document["excluded_seeds"])
        else:
            previous.update(row["split_group"]["seed"] for row in document["members"])
    scheduled = [
        seed + offset for phase in STARTS for seed in seeds(phase) for offset in range(len(MARKETS))
    ]
    if len(scheduled) != len(set(scheduled)) or not previous.isdisjoint(scheduled):
        raise ValueError(
            "Native schedule seeds overlap between groups or with supplied prior seeds"
        )
    write_new_file(
        folder / "plan.json",
        json_bytes({"protocol": protocol(), "prior_seeds_checked": len(previous)}),
    )


def original(case):
    found = []
    for parent in case.parents:
        if parent.direction is None:
            continue
        prices = _log_prices(case.rows[parent.start : parent.stop], CONFIG)
        bounds = _pause(prices, parent.direction, CONFIG)
        if bounds:
            found.append(Bounds(parent.start + bounds[0], parent.start + bounds[1]))
    return found


def evaluate(phase, methods):
    from regimex.trend_pauses import scan_trend_pauses

    rows = []
    for index, seed in enumerate(seeds(phase)):
        for case in cases(seed, index):
            scans = {}
            for method in methods:
                if method.name == "original":
                    found = original(case)
                else:
                    tick = case.tick if method.known_tick else None
                    if tick not in scans:
                        scans[tick] = scan_trend_pauses(
                            case.rows, case.parents, CONFIG, tick_size=tick
                        )
                    found = scans[tick].select(
                        multiple=not method.name.startswith("single-"),
                        gain_multiplier=method.gain,
                        absolute_limit=method.limit is not None,
                        absolute_multiplier=method.limit if method.limit is not None else 1.0,
                    )
                matched = matched_interval_indices(list(case.expected), list(found))
                rows.append(
                    {
                        "group": seed,
                        "market": case.market,
                        "shape": case.shape,
                        "severity": case.severity,
                        "method": method.name,
                        "tick": case.tick,
                        "active": case.active_steps,
                        "expected": len(case.expected),
                        "matched": len(matched),
                        "found": len(found),
                        "later_matched": sum(idx >= 1 for idx in matched),
                    }
                )
    return rows


def counts(rows):
    pos = [row for row in rows if row["expected"]]
    slow = [row for row in rows if "slowdown" in row["shape"]]
    steady = [row for row in rows if row["shape"] == "steady"]
    return {
        "matched": sum(r["matched"] for r in pos),
        "expected": sum(r["expected"] for r in pos),
        "extra_on_positive": sum(r["found"] - r["matched"] for r in pos),
        "slowdown_false": sum(r["found"] for r in slow),
        "slowdown_steps": len(slow) * STEPS,
        "steady_false": sum(r["found"] for r in steady),
        "steady_steps": len(steady) * STEPS,
        "later_matched": sum(r["later_matched"] for r in pos),
        "later_expected": sum(max(r["expected"] - 1, 0) for r in pos),
    }


def slowdown_rate(stats):
    return 1000 * stats["slowdown_false"] / stats["slowdown_steps"]


def choose(options, stats):
    def order(option):
        metrics = stats[option.name]
        rate = slowdown_rate(metrics)
        return (
            max(rate - BUDGET, 0),
            -metrics["matched"],
            rate,
            metrics["extra_on_positive"] + metrics["steady_false"],
        )

    return min(options, key=order)


def calibrate(folder):
    options = grid()
    detector_hash = digest(DETECTOR)
    rows = evaluate("development", options)
    if detector_hash != digest(DETECTOR):
        raise ValueError("Detector changed during calibration")
    stats = {m.name: counts([r for r in rows if r["method"] == m.name]) for m in options}
    single = choose([m for m in options if m.name.startswith("single")], stats)
    relative = choose([m for m in options if m.name.startswith("relative")], stats)
    absolute = choose([m for m in options if m.limit is not None], stats)
    candidate = choose([relative, absolute], stats)
    settings = {
        "plan_sha256": digest(folder / "plan.json"),
        "detector_sha256": digest(DETECTOR),
        "development": stats,
        "methods": [asdict(m) for m in (options[0], single, relative, absolute)],
        "release": candidate.name if slowdown_rate(stats[candidate.name]) <= BUDGET else None,
    }
    write_new_file(folder / "freeze.json", json_bytes(settings))
    print(json.dumps({key: settings[key] for key in ("methods", "release")}, indent=2), flush=True)


def paired_intervals(rows, methods):
    """Resample both schedule profiles and all price shapes of a group together."""
    rng = np.random.default_rng(19702)
    groups = sorted({r["group"] for r in rows})
    samples = rng.integers(len(groups), size=(2000, len(groups)))
    draws = {}
    for method in methods:
        groups_stats = [
            counts([r for r in rows if r["group"] == group and r["method"] == method.name])
            for group in groups
        ]
        metrics = [
            [
                stats["matched"] / stats["expected"],
                slowdown_rate(stats),
                stats["later_matched"] / stats["later_expected"],
            ]
            for stats in groups_stats
        ]
        draws[method.name] = np.array(metrics)[samples].mean(axis=1)
    return {
        f"{right.name} minus {left.name}": dict(
            zip(
                ("recall", "slowdown_false_per_1000", "later_recall"),
                np.quantile(
                    draws[right.name] - draws[left.name], [0.025, 0.975], axis=0
                ).T.tolist(),
                strict=True,
            )
        )
        for index, left in enumerate(methods)
        for right in methods[index + 1 :]
    }


def score(folder):
    freeze = json.loads((folder / "freeze.json").read_bytes())
    if freeze["plan_sha256"] != digest(folder / "plan.json") or freeze["detector_sha256"] != digest(
        DETECTOR
    ):
        raise ValueError("The plan or detector differs from the frozen version")
    methods = [Method(**m) for m in freeze["methods"]]
    absolute = methods[-1]
    methods.append(Method("absolute-known-tick", absolute.gain, absolute.limit, True))
    write_new_file(
        folder / "opened.json", json_bytes({"freeze_sha256": digest(folder / "freeze.json")})
    )
    rows = evaluate("final", methods)
    result = {
        "freeze_sha256": digest(folder / "freeze.json"),
        "summary": {m.name: counts([r for r in rows if r["method"] == m.name]) for m in methods},
        "paired_95pct": paired_intervals(rows, methods),
        "rows": rows,
    }
    write_new_file(folder / "results.json", json_bytes(result))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("reserve", "calibrate", "score"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--exclude-seeds", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.phase == "reserve":
        reserve(args.output, args.exclude_seeds)
    elif json.loads((args.output / "plan.json").read_bytes())["protocol"] != protocol():
        raise ValueError("The protocol differs from the reservation")
    elif args.phase == "calibrate":
        calibrate(args.output)
    else:
        score(args.output)
