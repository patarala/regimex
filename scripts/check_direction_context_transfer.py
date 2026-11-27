"""Replay the validation winner on paired price paths whose directions are known by construction."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from compare_direction_context import (
    CONTROL,
    PROFILE,
    checked_reservation,
    read,
    read_freeze,
)
from compare_direction_context import protocol as native_protocol

from regimex.data_catalog import DatasetManifest, content_hash
from regimex.directions import DIRECTION_NAMES, Direction, observation_blocks
from regimex.evaluation import _metrics, _segment_counts, dense_labels
from regimex.exports import ExportInstrument, ExportObservation, InstrumentUnits, ModelTimeGrid
from regimex.model_contracts import ObservationInputs, PredictionSequenceInfo
from regimex.segmentation import _interval, decode
from regimex.storage import digest, json_bytes, write_new_file

STEPS = 448
GROUPS = tuple(range(3_091_830_400, 3_091_830_432))
SHAPES = ("steady", "level-shift", "plateau", "reversal", "pause-before-reversal")
MARKETS = {"dense": 0.85, "sparse": 0.22}
VIEWS = ("complete", "masked")
NO_TRADES = (56, 96)
MISSING = (338, 378)
GATE = {"pooled": -0.03, "per_class": -0.06}
INSTRUMENT = ExportInstrument(
    instrument_id="BTC/USD",
    base_asset="BTC",
    quote_asset="USD",
    initial_price=100.0,
    units=InstrumentUnits(
        price_unit="quote_asset_per_base_asset",
        volume_unit="base_asset",
        price_decimal_places=2,
        volume_decimal_places=0,
    ),
)
GRID = ModelTimeGrid(
    origin=0,
    bin_width=1,
    unit="step",
    completed_horizon=STEPS,
    step_count=STEPS,
    terminal_bin="full_bins_only",
    interval_bounds="[start,stop)",
)


def protocol():
    return {
        "script_sha256": digest(Path(__file__)),
        "groups": list(GROUPS),
        "shapes": list(SHAPES),
        "markets": MARKETS,
        "views": list(VIEWS),
        "steps": STEPS,
        "ticks": [0.01, 0.20],
        "log_noise_sd": [0.0004, 0.0015],
        "drift": ["constant", "sinusoidally varying magnitude with constant sign"],
        "profile": PROFILE.model_dump(mode="json"),
        "truth": (
            "The scheduled drift sets direction. A level change with zero drift on both sides "
            "is sideways. Clip each truth interval at unavailable blocks."
        ),
        "no_execution_window": list(NO_TRADES),
        "missing_window_on_masked_view": list(MISSING),
        "comparison": "Validation winner versus mixed-192. No transfer paths enter selection.",
        "gate": GATE,
        "metric": "Pooled interval macro-F1 and per-direction F1 at IoU 0.5 over both views.",
        "replay": "Count changed predictions only on steps with observed truth in both views.",
        "native_cohorts_sha256": content_hash(native_protocol()["cohorts"]),
    }


def reserve(folder, comparison, prior):
    if (comparison / "freeze.json").exists():
        raise ValueError("Reserve transfer paths before freezing the validation winner")
    inventory = read(prior)
    if set(GROUPS).intersection(inventory["excluded_seeds"]):
        raise ValueError("Transfer groups repeat previously used seeds")
    prior_digest = digest(prior)
    native_seeds = seed_audit(comparison, checked_reservation(comparison), prior_digest)
    write_new_file(
        folder / "plan.json",
        json_bytes(
            {
                "protocol": protocol(),
                "prior_sha256": prior_digest,
                "prior_groups": len(inventory["excluded_seeds"]),
                "disjoint_native_groups": native_seeds,
                "native_reservation_sha256": digest(comparison / "reservation.json"),
            }
        ),
    )


@dataclass(frozen=True)
class PathView:
    shape: str
    market: str
    view: str
    rows: tuple[ExportObservation, ...]
    directions: np.ndarray


def paths(seed, index):
    rng = np.random.default_rng(seed)
    step = np.arange(STEPS)
    sign = (-1, 1)[index % 2]
    rates = np.full(STEPS, sign * rng.uniform(0.00020, 0.00042))
    if index // 2 % 2:
        rates *= 1 + 0.48 * np.sin(step / rng.uniform(37, 65) + rng.uniform(-np.pi, np.pi))
    tick = (0.01, 0.20)[index // 4 % 2]
    sigma = (0.0004, 0.0015)[index // 8 % 2]
    noise = sigma * np.sqrt(3 / 5) * rng.standard_t(5, STEPS)
    for i in range(1, STEPS):
        noise[i] = 0.6 * noise[i - 1] + 0.8 * noise[i]
    level_change = sign * rng.uniform(0.009, 0.021)
    left, right = int(rng.integers(146, 183)), int(rng.integers(240, 291))
    middle = (left + right) // 2
    # Shared uniforms make the sparse executions a subset of the dense executions.
    uniforms = rng.random(STEPS)
    trend = DIRECTION_NAMES.index(Direction.GROWTH if sign > 0 else Direction.DECLINE)
    opposite = DIRECTION_NAMES.index(Direction.DECLINE if sign > 0 else Direction.GROWTH)
    sideways = DIRECTION_NAMES.index(Direction.SIDEWAYS)
    for shape in SHAPES:
        direction = np.full(STEPS, trend, dtype=np.int64)
        increment = rates.copy()
        shift = np.zeros(STEPS)
        if shape == "level-shift":
            direction[:] = sideways
            increment[:] = 0
            shift[middle:] = level_change
        if shape in {"plateau", "pause-before-reversal"}:
            direction[left:right] = sideways
            increment[left + 1 : right] = 0
        if shape in {"reversal", "pause-before-reversal"}:
            boundary = middle if shape == "reversal" else right
            direction[boundary:] = opposite
            increment[boundary:] *= -1
        increment[0] = 0
        latent = np.cumsum(increment) + shift
        prices = np.round(tick * np.rint(100 * np.exp(latent + noise) / tick), 2)
        for market, probability in MARKETS.items():
            active = uniforms < probability
            # Limit accidental inactivity so only the prescribed window can split a truth interval.
            active[::24] = True
            active[slice(*NO_TRADES)] = False
            # Keep controlled gaps separate from chance inactivity at their edges.
            active[[0, NO_TRADES[0] - 1, NO_TRADES[1], MISSING[0] - 1, MISSING[1]]] = True
            observed = []
            last = 100.0
            for i, executed in enumerate(active):
                if executed:
                    last = float(prices[i])
                observed.append(
                    ExportObservation(
                        instrument_id=INSTRUMENT.instrument_id,
                        step=i,
                        last_price=last,
                        price_state="observed" if executed else "carried",
                        trade_count=int(executed),
                        trade_volume=float(executed),
                        buy_order_count=None,
                        buy_order_volume=None,
                        sell_order_count=None,
                        sell_order_volume=None,
                        best_bid=None,
                        best_ask=None,
                        ask_state="unknown",
                        bid_state="unknown",
                    )
                )
            complete = tuple(observed)
            masked = list(complete)
            for i in range(*MISSING):
                masked[i] = complete[i].model_copy(
                    update=dict(
                        last_price=None, price_state="unknown", trade_count=None, trade_volume=None
                    )
                )
            yield PathView(shape, market, "complete", complete, direction)
            yield PathView(shape, market, "masked", tuple(masked), direction)


def references(case):
    # Define movement from the scheduled drift before observing executions or hiding prices.
    bounds = np.r_[0, np.flatnonzero(np.diff(case.directions)) + 1, STEPS]
    result = []
    for start, stop, status in observation_blocks(case.rows, PROFILE):
        if status != "observed":
            result.append(_interval(case.rows, start, stop, None, status))
            continue
        for a, b in pairwise(bounds):
            left, right = max(start, int(a)), min(stop, int(b))
            if left < right:
                direction = Direction(DIRECTION_NAMES[case.directions[left]])
                result.append(_interval(case.rows, left, right, direction, status))
    return tuple(result)


def inputs_for(cases):
    return ObservationInputs(
        tuple(
            PredictionSequenceInfo(
                experiment_id=f"path-{i}", instrument_id=INSTRUMENT.instrument_id, step_count=STEPS
            )
            for i in range(len(cases))
        ),
        tuple(case.rows for case in cases),
        (GRID,) * len(cases),
        (INSTRUMENT,) * len(cases),
    )


def gate(control, candidate):
    return {
        "pooled": candidate["macro_f1"] - control["macro_f1"] >= GATE["pooled"] - 1e-12,
        "per_class": {
            name: candidate["per_class"][name]["f1"] - control["per_class"][name]["f1"]
            >= GATE["per_class"] - 1e-12
            for name in DIRECTION_NAMES
        },
    }


def seed_audit(comparison, reservation, prior_digest):
    seen = set()
    for pool, record in reservation["pools"].items():
        manifest = DatasetManifest.for_modeling((comparison / pool / "dataset.json").read_bytes())
        if (
            manifest.content_sha256 != record["dataset_sha256"]
            or record["external_inventory_sha256"] != prior_digest
        ):
            raise ValueError(
                "Native cohort or prior inventory differs from the transfer reservation"
            )
        seen.update(member.split_group.seed for member in manifest.members)
    if seen.intersection(GROUPS):
        raise ValueError("Transfer groups overlap the native evaluation")
    return len(seen)


def score(folder, comparison):
    from compare_direction_context import load_predictors

    design = read(folder / "plan.json")
    if design["protocol"] != protocol() or design["native_reservation_sha256"] != digest(
        comparison / "reservation.json"
    ):
        raise ValueError("The reserved transfer design changed")
    native = checked_reservation(comparison)
    native_seeds = seed_audit(comparison, native, design["prior_sha256"])
    if native_seeds != design["disjoint_native_groups"]:
        raise ValueError("Native evaluation membership changed after reservation")
    selected = read_freeze(comparison)
    winner = selected["winner"]
    receipt = {
        "transfer_plan_sha256": digest(folder / "plan.json"),
        "native_freeze_sha256": digest(comparison / "freeze.json"),
        "disjoint_native_groups": native_seeds,
        "winner": winner,
    }
    methods = tuple(dict.fromkeys((CONTROL, winner)))
    predict, _ = load_predictors(comparison, methods)
    write_new_file(folder / "opened.json", json_bytes(receipt))
    totals = defaultdict(lambda: np.zeros((3, 3), dtype=np.int64))
    moves = defaultdict(lambda: np.zeros(2, dtype=np.int64))
    for index, seed in enumerate(GROUPS):
        cases = list(paths(seed, index))
        inputs = inputs_for(cases)
        expected = [references(case) for case in cases]
        probabilities_by_method = predict(inputs)
        for name in methods:
            probabilities = probabilities_by_method[name].reshape(
                len(cases), STEPS, len(DIRECTION_NAMES)
            )
            full = {}
            for case, truth, prob in zip(cases, expected, probabilities, strict=True):
                decoded = decode(prob, case.rows, PROFILE)
                counts = _segment_counts(truth, decoded)
                for key in ("pooled", case.view, case.market, case.shape):
                    totals[name, key] += counts
                labels = dense_labels(truth, STEPS)
                directions = dense_labels(decoded, STEPS)
                path_key = (case.shape, case.market)
                if case.view == "complete":
                    full[path_key] = labels, directions
                else:
                    old_labels, old = full[path_key]
                    common = (labels >= 0) & (old_labels >= 0)
                    moves[name] += [
                        int((directions[common] != old[common]).sum()),
                        int(common.sum()),
                    ]
    scores = {
        name: {k: _metrics(totals[name, k]) for k in ("pooled", *VIEWS, *MARKETS, *SHAPES)}
        for name in methods
    }
    checks = gate(scores[CONTROL]["pooled"], scores[winner]["pooled"])
    if (
        digest(comparison / "freeze.json") != receipt["native_freeze_sha256"]
        or read_freeze(comparison) != selected
        or digest(folder / "plan.json") != receipt["transfer_plan_sha256"]
        or design["protocol"] != protocol()
    ):
        raise ValueError("Transfer design or fitted validation winner changed during scoring")
    result = {
        **receipt,
        "scores": scores,
        "groups": len(GROUPS),
        "paired_paths": len(GROUPS) * len(SHAPES) * len(MARKETS),
        "changed_directions_on_common_steps": {
            name: {"changed": int(pair[0]), "compared": int(pair[1])}
            for name, pair in moves.items()
        },
        "checks": checks,
        "passed": checks["pooled"] and all(checks["per_class"].values()),
    }
    write_new_file(folder / "results.json", json_bytes(result))
    print(
        f"{winner}, transfer passed={result['passed']}, "
        f"macro-F1={scores[winner]['pooled']['macro_f1']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    for phase in ("reserve", "score"):
        cmd = phases.add_parser(phase)
        cmd.add_argument("output", type=Path)
        cmd.add_argument("comparison", type=Path)
        if phase == "reserve":
            cmd.add_argument("--prior", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "reserve":
        reserve(args.output, args.comparison, args.prior)
    else:
        score(args.output, args.comparison)
