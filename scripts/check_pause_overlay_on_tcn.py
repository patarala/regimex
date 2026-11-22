"""Replay the frozen pause policies downstream of a saved TCN and its own decoder."""

import argparse
import json
from pathlib import Path

from compare_trend_pauses import counts, slowdown_rate
from trend_pause_cases import STEPS, Bounds, cases

from regimex.backends.tcn import TCNBackend
from regimex.directions import Direction
from regimex.evaluation import matched_interval_indices
from regimex.exports import (
    ExportInstrument,
    ExportPayloadV2,
    InstrumentUnits,
    ModelTimeGrid,
    seal_export,
)
from regimex.model_contracts import ObservationInputs
from regimex.model_registry import InferenceRuntime
from regimex.prediction_inputs import PredictionSequenceInfo
from regimex.segmentation import decode
from regimex.storage import digest, json_bytes, write_new_file
from regimex.trend_pauses import scan_trend_pauses

GROUPS = 64
FIRST_SEED = 2_071_700_000
METHODS = ("report", "shape/shape-5", "shape/shape-6")
RUNTIME = InferenceRuntime(threads=2, max_batch_size=8)
BUNDLES = {
    "tcn-0.1.0": "fa222148503cb4b5e3626834dfee7e04baa8939f792b9708109ce34866716506",
    "mixed-192": "5635e71056afbeb335459fb611e85f7d58edd5ff26016850860f318a76f161dd",
}


def protocol():
    return {
        "script_sha256": digest(Path(__file__)),
        "group_seeds": [FIRST_SEED + 2 * i for i in range(GROUPS)],
        "bundle_sha256": BUNDLES,
        "methods": list(METHODS),
        "runtime": RUNTIME.model_dump(mode="json"),
        "matching": "One-to-one IoU >= 0.5. A pause highlight must carry the correct parent "
        "direction. A model sideways interval is counted separately and in the combined view.",
    }


def observed_export(series_id, rows):
    instrument_id = rows[0].instrument_id
    base, quote = instrument_id.split("/")
    instrument = ExportInstrument(
        instrument_id=instrument_id,
        base_asset=base,
        quote_asset=quote,
        initial_price=100,
        units=InstrumentUnits(
            price_unit="quote_asset_per_base_asset",
            volume_unit="base_asset",
            price_decimal_places=2,
            volume_decimal_places=2,
        ),
    )
    # Tick multiplication in the synthetic path can leave binary roundoff after two decimals.
    # Serialize to the declared precision before either the TCN or detector sees the series.
    serialized = tuple(
        row.model_copy(update={"last_price": round(row.last_price, 2)}) for row in rows
    )
    payload = ExportPayloadV2.model_validate(
        {
            "schema_version": "2.1",
            "source": {
                "source_id": "controlled-price-paths",
                "source_version": "1",
                "configuration_id": "pause-comparison",
                "run_id": series_id,
                "snapshot_id": series_id,
                "state_revision": "1",
                "run_status": "completed",
                "consistency": "atomic_completed_snapshot",
            },
            "producer": {"name": "regimex-price-paths", "version": "1"},
            "time_grid": ModelTimeGrid(
                origin=0,
                unit="step",
                bin_width=1,
                completed_horizon=STEPS,
                step_count=STEPS,
                interval_bounds="[start,stop)",
                terminal_bin="full_bins_only",
            ),
            "capabilities": {
                "execution_prices": "complete",
                "trade_aggregates": "complete",
                "accepted_order_aggregates": "complete",
                "price_policy": "initial_then_carry",
                "trade_count_policy": "one_per_partial_execution",
                "order_count_policy": "accepted_submissions_including_immediate_fills",
                "rejected_order_policy": "excluded",
                "historical_quotes": {
                    "mode": "unavailable",
                    "coverage": "unavailable",
                    "provenance": "Only execution prices are recorded on this path.",
                    "complete_event_history_sha256": None,
                    "reconstruction_policy": None,
                },
            },
            "instruments": (instrument,),
            "observations": serialized,
        }
    )
    return seal_export(payload)


def observed_inputs(series):
    exports = [observed_export(f"path-{i}", rows) for i, rows in enumerate(series)]
    return ObservationInputs(
        tuple(
            PredictionSequenceInfo(
                experiment_id=e.content_sha256,
                instrument_id=e.instruments[0].instrument_id,
                step_count=STEPS,
            )
            for e in exports
        ),
        tuple(e.observations for e in exports),
        tuple(e.time_grid for e in exports),
        tuple(e.instruments[0] for e in exports),
    )


def evaluate_group(index, seed, model, methods):
    from confirm_pause_precision import predict

    panel = list(cases(seed, index))
    inputs = observed_inputs([case.rows for case in panel])
    probabilities = model.predict(inputs, runtime=RUNTIME).reshape(len(panel), STEPS, 3)
    records = []
    for case, rows, values in zip(panel, inputs.observations, probabilities, strict=True):
        decoded = decode(values, rows, model.analysis)
        scan = scan_trend_pauses(rows, decoded, model.analysis)
        overlays = predict(scan, methods)
        sideways = [
            Bounds(seg.start, seg.stop) for seg in decoded if seg.direction == Direction.SIDEWAYS
        ]
        truth_direction = next(seg.direction for seg in case.parents if seg.direction is not None)
        correct_trends = [seg for seg in decoded if seg.direction == truth_direction]
        views = {"TCN sideways": (sideways, len(sideways))}
        for name, marks in overlays.items():
            eligible = [
                mark
                for mark in marks
                if any(seg.start <= mark.start and mark.stop <= seg.stop for seg in correct_trends)
            ]
            views[name] = eligible, len(marks)
            views[f"{name} + sideways"] = eligible + sideways, len(marks) + len(sideways)
        for name, (eligible, total) in views.items():
            matches = matched_interval_indices(list(case.expected), eligible)
            records.append(
                {
                    "group": seed,
                    "market": case.market,
                    "method": name,
                    "shape": case.shape,
                    "found": total,
                    "expected": len(case.expected),
                    "matched": len(matches),
                    "later_matched": sum(idx >= 1 for idx in matches),
                }
            )
    return records


def score(folder, comparison, bundle, name):
    from confirm_pause_precision import read_freeze

    if json.loads((folder / "plan.json").read_bytes()) != protocol():
        raise ValueError("TCN replay differs from the reservation")
    freeze = read_freeze(comparison)
    if freeze["selected"] != "shape/shape-5":
        raise ValueError("Expected the first confirmation to have selected the one-fifth slope")
    model = TCNBackend().load_bundle(bundle, manifest_sha256=BUNDLES[name], runtime=RUNTIME)
    receipt = {
        "plan_sha256": digest(folder / "plan.json"),
        "comparison_freeze_sha256": digest(comparison / "freeze.json"),
        "bundle_sha256": BUNDLES[name],
    }
    write_new_file(folder / name / "opened.json", json_bytes(receipt))
    records = []
    for index, seed in enumerate(protocol()["group_seeds"]):
        records.extend(evaluate_group(index, seed, model, METHODS))
    summary = {}
    for method in dict.fromkeys(row["method"] for row in records):
        stats = counts([row for row in records if row["method"] == method])
        summary[method] = {**stats, "false_per_1000_slowdown_steps": slowdown_rate(stats)}
    write_new_file(
        folder / name / "results.json", json_bytes({**receipt, "summary": summary, "rows": records})
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    sub.add_parser("reserve").add_argument("output", type=Path)
    score_args = sub.add_parser("score")
    score_args.add_argument("output", type=Path)
    score_args.add_argument("comparison", type=Path)
    score_args.add_argument("name", choices=BUNDLES)
    score_args.add_argument("bundle", type=Path)
    args = parser.parse_args()
    if args.phase == "reserve":
        write_new_file(args.output / "plan.json", json_bytes(protocol()))
    else:
        score(args.output, args.comparison, args.bundle, args.name)
