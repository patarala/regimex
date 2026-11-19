"""Score the first tcn-0.2.0 benchmark experiment with the release's own evaluator."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np

from regimex import evaluation
from regimex.application.training import observation_inputs
from regimex.data_catalog import LabeledSnapshot, ValidatedAnnotations
from regimex.directions import AnalysisConfig, DirectionAnnotations
from regimex.exports import ExportV2
from regimex.segmentation import decode


def check(directory: Path) -> None:
    script = directory / "reproduce.py"
    with script.open("rb") as stream:
        recipe_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if recipe_sha256 != "8cd477c8c153ab29e3f0090b457184851e5d27f66b3498e3861476c2a96fd905":
        raise ValueError("Expected the published tcn-0.2.0 scoring script.")
    benchmark = directory / "data/benchmark"
    entry = json.loads((benchmark / "payload-index.json").read_bytes())["ordered_final_payloads"][0]
    export = ExportV2.model_validate_json((benchmark / entry["snapshot_path"]).read_bytes())
    labels = DirectionAnnotations.model_validate_json(
        (benchmark / entry["annotation_path"]).read_bytes()
    )
    profile = AnalysisConfig.model_validate(labels.policy_config)
    sample = LabeledSnapshot(
        source_namespace=entry["binding"]["source_namespace"],
        snapshot_id=export.source.snapshot_id,
        export=export,
        annotations=ValidatedAnnotations(
            experiment_id=labels.experiment_id,
            policy=labels.policy,
            analysis_profile=profile,
            reference_kind=labels.reference_kind,
            reviewer=labels.reviewer,
            intervals=labels.intervals,
        ),
    )
    inputs = observation_inputs((sample,))
    saved = np.load(directory / "tcn-probabilities.npy", mmap_mode="r", allow_pickle=False)
    # These are the scientific bindings the release CLI sets inside its score command.
    recipe = runpy.run_path(
        str(script),
        init_globals={
            "np": np,
            "evaluation": evaluation,
            "probabilities_valid": evaluation.probabilities_valid,
            "dense_labels": evaluation.dense_labels,
            "report": evaluation.report,
            "decode": decode,
        },
    )
    result = recipe["score_probabilities"](
        inputs, recipe["references_for"]((sample,)), profile, saved[: inputs.row_count]
    )
    rows = json.loads((directory / "tcn-metrics.json").read_bytes())["experiment_statistics"]
    expected = [row for row in rows if row["experiment_id"] == inputs.info[0].experiment_id]
    if len(expected) != 1 or result["pooled"]["experiment_statistics"] != expected:
        raise ValueError("The first experiment's recomputed statistics differ from the release.")
    print(f"Verified {len(inputs.info)} sequences and {inputs.row_count} saved predictions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_directory", type=Path)
    check(parser.parse_args().evaluation_directory)
