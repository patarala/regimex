"""Apply the frozen validation winner to a simulation export on its original time grid."""

import argparse
from pathlib import Path

from compare_direction_context import (
    CONTROL,
    DEFAULT_BUNDLES,
    PROFILE,
    checked_reservation,
    load_predictors,
    read_freeze,
)

from regimex.application.analysis_computation import snapshot_inputs
from regimex.directions import DIRECTION_NAMES
from regimex.evaluation import probabilities_valid
from regimex.exports import load_export_v2
from regimex.segmentation import decode
from regimex.storage import digest, json_bytes, write_new_file


def predict_export(comparison: Path, export_file: Path, destination: Path, control: Path) -> None:
    checked_reservation(comparison)
    freeze = read_freeze(comparison)
    winner = freeze["winner"]
    freeze_digest = digest(comparison / "freeze.json")
    predict, _ = load_predictors(comparison, (winner,), {CONTROL: control})
    export = load_export_v2(export_file)
    inputs = snapshot_inputs(export)
    scores = predict(inputs)[winner]
    probabilities_valid(scores, inputs.row_count)
    instruments = []
    offset = 0
    for instrument, observations in zip(inputs.instruments, inputs.observations, strict=True):
        values = scores[offset : offset + len(observations)]
        intervals = decode(values, observations, PROFILE)
        instruments.append(
            {
                "instrument_id": instrument.instrument_id,
                "intervals": [item.model_dump(mode="json") for item in intervals],
                "probabilities": values.tolist(),
            }
        )
        offset += len(observations)
    write_new_file(
        destination,
        json_bytes(
            {
                "model": winner,
                "freeze_sha256": freeze_digest,
                "export_sha256": export.content_sha256,
                "time_grid": export.time_grid.model_dump(mode="json"),
                "analysis_profile": PROFILE.model_dump(mode="json"),
                "class_names": list(DIRECTION_NAMES),
                "instruments": instruments,
            }
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path, help="Directory containing freeze.json")
    parser.add_argument("export", type=Path, help="Simulation export (experiment-v2 JSON)")
    parser.add_argument("output", type=Path, help="New JSON file for intervals and probabilities")
    parser.add_argument("--control", type=Path, default=DEFAULT_BUNDLES[CONTROL])
    args = parser.parse_args()
    predict_export(args.comparison, args.export, args.output, args.control)
