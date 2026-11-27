"""Compare the frozen interval classifier and its predecessor on paired incomplete exports."""

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from check_direction_context_transfer import MARKETS, PROFILE, STEPS, inputs_for, references
from check_direction_context_transfer import paths as original_paths

from regimex.data_catalog import DatasetManifest
from regimex.directions import DIRECTION_NAMES, Direction
from regimex.evaluation import _metrics, _segment_counts, dense_labels, matched_interval_indices
from regimex.segmentation import decode
from regimex.storage import digest, json_bytes, write_new_file

GROUPS = tuple(range(3_683_710_400, 3_683_710_432))
BASE_SHAPES = ("steady", "level-shift", "reversal")
SHORT_SHAPES = ("short-plateau", "short-pause-before-reversal")
SHAPES = (*BASE_SHAPES, *SHORT_SHAPES, "slow-continuation")
WIDTHS = (32, 48, 64)
BANDS = ("1-32", "33-64", "65-128", "129+")
GATES = {"pooled_f1_loss": 0.02, "class_f1_loss": 0.04, "increase_in_changed_fraction": 0.005}


def read(path):
    return json.loads(path.read_bytes())


def protocol():
    return {
        "sources": {
            name: digest(Path(__file__).with_name(name))
            for name in (Path(__file__).name, "check_direction_context_transfer.py")
        },
        "groups": list(GROUPS),
        "shapes": SHAPES,
        "markets": MARKETS,
        "steps": STEPS,
        "profile": PROFILE.model_dump(mode="json"),
        "short_pause_steps": [32, 63],
        "slow_continuation": "40% of the preceding drift during the short middle span.",
        "gap_widths": WIDTHS,
        "locations": (
            "Place the gap on either side of a reversal. For short pauses use either a "
            "neighboring gap or a distant gap. Preserve the full pause in both exports."
        ),
        "distance_bands": BANDS,
        "gates": GATES,
        "quality": "Pooled interval macro-F1 and each direction's F1 at IoU 0.5 across both views.",
        "stability": "Changed direction among steps classified in both exports.",
    }


def seed_audit(comparison, reservation):
    prior = comparison / "prior.json"
    inventory = read(prior)
    native = set()
    for pool, receipt in reservation["pools"].items():
        manifest = DatasetManifest.for_modeling((comparison / pool / "dataset.json").read_bytes())
        if (
            manifest.content_sha256 != receipt["dataset_sha256"]
            or digest(prior) != receipt["external_inventory_sha256"]
        ):
            raise ValueError("Native membership or its exposure inventory changed")
        native.update(member.split_group.seed for member in manifest.members)
    if set(GROUPS) & (native | set(inventory["excluded_seeds"])):
        raise ValueError("Transfer groups overlap previously used or reserved groups")
    return {
        "native_seed_count": len(native),
        "prior_seed_count": len(inventory["excluded_seeds"]),
        "prior_sha256": digest(prior),
    }


def reserve(output, comparison):
    from compare_adaptive_intervals import checked_reservation

    if (comparison / "freeze.json").exists():
        raise ValueError("Reserve transfer groups before validation selection")
    design = {
        "protocol": protocol(),
        "native_reservation_sha256": digest(comparison / "reservation.json"),
        "disjoint_from": seed_audit(comparison, checked_reservation(comparison)),
    }
    write_new_file(output / "plan.json", json_bytes(design))


def short_schedules(seed, index):
    # A separate random stream shares the new shapes' noise and geometry across both markets.
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
    start = int(rng.integers(182, 209))
    stop = start + int(rng.integers(32, 64))
    step = np.arange(STEPS)
    sign = (-1, 1)[index % 2]
    rate = sign * rng.uniform(0.00024, 0.00046)
    modulation = 1 + 0.30 * np.sin(step / 47 + rng.uniform(-np.pi, np.pi))
    noise = rng.normal(0, (0.00045, 0.0015)[index // 8 % 2], STEPS)
    for i in range(1, STEPS):
        noise[i] = 0.6 * noise[i - 1] + 0.8 * noise[i]
    tick = (0.01, 0.20)[index // 4 % 2]
    trend = DIRECTION_NAMES.index(Direction.GROWTH if sign > 0 else Direction.DECLINE)
    opposite = DIRECTION_NAMES.index(Direction.DECLINE if sign > 0 else Direction.GROWTH)
    for shape in (*SHORT_SHAPES, "slow-continuation"):
        truth = np.full(STEPS, trend)
        increments = rate * modulation
        increments[0] = 0
        if shape == "slow-continuation":
            increments[start:stop] *= 0.4
        else:
            increments[start:stop] = 0
            truth[start:stop] = DIRECTION_NAMES.index(Direction.SIDEWAYS)
            if shape == "short-pause-before-reversal":
                increments[stop:] *= -1
                truth[stop:] = opposite
        prices = np.round(tick * np.rint(100 * np.exp(np.cumsum(increments) + noise) / tick), 2)
        yield shape, truth, prices, (start, stop)


def gap_for(shape, index, bounds):
    width = WIDTHS[(index // 2) % len(WIDTHS)]
    start, stop = bounds
    if shape == "reversal" or (shape in SHORT_SHAPES and index % 4 < 2):
        left = start - width if index // 4 % 2 == 0 else stop
    else:
        left = (109, 304, 355)[(index // 3) % 3]
    return left, left + width


def paired(case, bounds):
    left, right = bounds
    rows = list(case.rows)
    for i in range(left, right):
        rows[i] = rows[i].model_copy(
            update=dict(last_price=None, price_state="unknown", trade_count=None, trade_volume=None)
        )
    return case, replace(case, view="masked", rows=tuple(rows)), bounds


def pairs(seed, index):
    schedules = list(short_schedules(seed, index))
    for case in original_paths(seed, index):
        if case.view != "complete" or case.shape not in BASE_SHAPES:
            continue
        boundary = int(np.argmax(case.directions != case.directions[0]))
        yield paired(case, gap_for(case.shape, index, (boundary, boundary)))
        if case.shape != "steady":
            continue
        for shape, truth, prices, bounds in schedules:
            rows = []
            last = 100.0
            for row in case.rows:
                if row.trade_count:
                    last = float(prices[row.step])
                rows.append(row.model_copy(update={"last_price": last}))
            full = replace(case, shape=shape, rows=tuple(rows), directions=truth)
            yield paired(full, gap_for(shape, index, bounds))


def changed_steps(before, after, bounds):
    left, right = bounds
    common = (before >= 0) & (after >= 0)
    step = np.arange(len(before))
    distance = np.maximum(left - step, step - right + 1)
    changed = before != after
    result = {}
    for name, bottom, top in zip(BANDS, (1, 33, 65, 129), (32, 64, 128, len(step)), strict=True):
        within = common & (distance >= bottom) & (distance <= top)
        result[name] = np.array([np.sum(changed & within), np.sum(within)])
    return result


def gate(base, candidate):
    metrics = [m["interval"]["pooled"] for m in (base, candidate)]
    original, selected = metrics
    return {
        "pooled": selected["macro_f1"] >= original["macro_f1"] - GATES["pooled_f1_loss"] - 1e-12,
        "per_class": {
            name: selected["per_class"][name]["f1"]
            >= original["per_class"][name]["f1"] - GATES["class_f1_loss"] - 1e-12
            for name in DIRECTION_NAMES
        },
        "stability": candidate["changed"]["pooled"]["fraction"]
        <= base["changed"]["pooled"]["fraction"] + GATES["increase_in_changed_fraction"] + 1e-12,
    }


class Totals:
    def __init__(self):
        self.intervals = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))
        self.short = defaultdict(lambda: np.zeros(2, dtype=np.int64))
        self.changes = defaultdict(lambda: np.zeros(2, dtype=np.int64))

    def add_view(self, case, truth, predicted):
        counts = _segment_counts(truth, predicted)
        for key in ("pooled", case.view, case.market, case.shape):
            self.intervals[key] += counts
        if case.shape in SHORT_SHAPES:
            expected = [r for r in truth if r.direction == Direction.SIDEWAYS]
            if len(expected) != 1 or not 32 <= expected[0].stop - expected[0].start <= 63:
                raise ValueError("The scheduled short plateau must remain fully observable")
            actual = [r for r in predicted if r.direction == Direction.SIDEWAYS]
            matched = len(matched_interval_indices(expected, actual))
            self.short[case.view] += [matched, 1]

    def summary(self):
        def rates(counts, keys):
            return {
                name: {keys[0]: int(a), keys[1]: int(b), "fraction": float(a / b) if b else None}
                for name, (a, b) in counts.items()
            }

        return {
            "interval": {key: _metrics(value) for key, value in self.intervals.items()},
            "designed_short_pauses": rates(self.short, ("matched", "references")),
            "changed": rates(self.changes, ("changed", "common")),
        }


def score(output, comparison, bundle):
    from compare_adaptive_intervals import (
        CONTROL,
        checked_reservation,
        load_predictors,
        read_freeze,
    )

    freeze = read_freeze(comparison)
    winner = freeze["winner"]
    if winner == CONTROL:
        raise ValueError("The release won validation; the transfer groups remain reserved")
    plan = output / "plan.json"
    design = read(plan)
    if (
        json_bytes(design["protocol"]) != json_bytes(protocol())
        or design["native_reservation_sha256"] != digest(comparison / "reservation.json")
        or design["disjoint_from"] != seed_audit(comparison, checked_reservation(comparison))
    ):
        raise ValueError("Transfer design or population changed")
    opening = {
        "plan_sha256": digest(plan),
        "freeze_sha256": digest(comparison / "freeze.json"),
        "winner": winner,
    }
    names = (CONTROL, winner)
    predict = load_predictors(comparison, names, bundle)
    write_new_file(output / "opened.json", json_bytes(opening))
    totals = {name: Totals() for name in names}
    for index, seed in enumerate(GROUPS):
        cases = list(pairs(seed, index))
        views = [view for full, masked, _ in cases for view in (full, masked)]
        scores = predict(inputs_for(views))
        truth = [references(view) for view in views]
        for name in names:
            probabilities = scores[name].reshape(len(views), STEPS, len(DIRECTION_NAMES))
            for i, (full, masked, bounds) in enumerate(cases):
                decoded = []
                for j, case in enumerate((full, masked), start=2 * i):
                    intervals = decode(probabilities[j], case.rows, PROFILE)
                    totals[name].add_view(case, truth[j], intervals)
                    decoded.append(dense_labels(intervals, STEPS))
                for band, count in changed_steps(*decoded, bounds).items():
                    totals[name].changes[band] += count
                    totals[name].changes["pooled"] += count
    results = {name: total.summary() for name, total in totals.items()}
    checks = gate(results[CONTROL], results[winner])
    if (
        read_freeze(comparison) != freeze
        or digest(comparison / "freeze.json") != opening["freeze_sha256"]
        or digest(plan) != opening["plan_sha256"]
        or json_bytes(design["protocol"]) != json_bytes(protocol())
    ):
        raise ValueError("Frozen comparison changed during replay")
    result = {
        **opening,
        "groups": len(GROUPS),
        "paired_paths": len(GROUPS) * len(SHAPES) * len(MARKETS),
        "results": results,
        "checks": checks,
        "passed": checks["stability"] and checks["pooled"] and all(checks["per_class"].values()),
    }
    write_new_file(output / "results.json", json_bytes(result))
    print(f"{winner}: transfer checks passed={result['passed']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase in ("reserve", "score"):
        cmd = phases.add_parser(phase)
        cmd.add_argument("output", type=Path)
        cmd.add_argument("comparison", type=Path)
        if phase == "score":
            cmd.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "reserve":
        reserve(args.output, args.comparison)
    else:
        score(args.output, args.comparison, args.control)
