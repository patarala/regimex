"""Fit an offline TCN with refined training targets and unchanged validation references."""

import argparse
import hashlib
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from compare_profile_21 import RUNTIME, Corpus, sha256, software, write
from compare_tcn_widths import POLICY, PROFILE, training_data
from compare_temporal_architectures import evaluator, read
from compare_training_populations import per_seed_history
from refine_trend_pauses import PauseAnnotations, annotate_pauses

from regimex.application.training import _references, observation_inputs
from regimex.backends.tcn import TCNBackend, TCNPredictor
from regimex.backends.tcn_update_records import UpdateHistoryRecord
from regimex.data_catalog import DatasetManifest, LabeledSnapshot, SnapshotGrid
from regimex.evaluation import dense_labels
from regimex.model_contracts import TrainingInputs
from regimex.model_registry import TCNParameters, TrainingRuntime
from regimex.storage import output_directory

DATASET = "4ff45ca06aa65f3a5fddebb4045ecbb69617f286d4702d5ec67106e42c62e8f7"
ANNOTATOR = "49ddef90bf02af7595319006ea8d5e3070bf474c46e72cb435f406a3ba6d5c97"
PARAMETERS = TCNParameters(channels=48)
TRAIN_RUNTIME = TrainingRuntime(threads=2)


def training_protocol():
    folder = Path(__file__).parent
    names = (
        "fit_pause_targets.py",
        "refine_trend_pauses.py",
        "compare_training_populations.py",
        "compare_temporal_architectures.py",
        "compare_tcn_widths.py",
    )
    files = {name: sha256(folder / name) for name in names}
    if files["refine_trend_pauses.py"] != ANNOTATOR:
        raise ValueError("The pause annotator differs from the frozen candidate")
    return {
        "training_dataset": DATASET,
        "control": "mixed-192",
        "candidate_train_policy": PauseAnnotations.policy,
        "validation_profile": PROFILE.model_dump(mode="json"),
        "parameters": PARAMETERS.model_dump(mode="json"),
        "updates": POLICY.model_dump(mode="json"),
        "runtime": TRAIN_RUNTIME.model_dump(mode="json"),
        "selection": "Highest original validation interval macro-F1 across seeds and checkpoints.",
        "environment": {**software(), "scripts": files},
    }


def refined_targets(dataset: DatasetManifest, samples: tuple[LabeledSnapshot, ...]) -> np.ndarray:
    expected = [
        (m.source_namespace, m.snapshot_id, m.export_sha256)
        for m in dataset.members
        if m.partition == "train"
    ]
    actual = [(s.source_namespace, s.snapshot_id, s.export.content_sha256) for s in samples]
    if actual != expected or not samples:
        raise ValueError("Refinement requires the complete ordered training partition")
    labels = []
    for sample in samples:
        grid = SnapshotGrid.from_export(sample.export)
        intervals = annotate_pauses(grid, PROFILE).intervals
        for instrument in grid.instruments:
            matched = tuple(i for i in intervals if i.instrument_id == instrument.instrument_id)
            labels.append(dense_labels(matched, grid.step_count))
    return np.concatenate(labels)


def load_control(populations: Path, manifest_sha256: str):
    entry = read(populations / "fitted/freeze.json")["candidates"]["mixed-192"]
    if entry["dataset_sha256"] != DATASET or entry["bundle_sha256"] != manifest_sha256:
        raise ValueError("The original training run differs from the reserved control")
    bundle = populations / "fitted/mixed-192/bundle"
    predictor = TCNBackend().load_bundle(bundle, manifest_sha256=manifest_sha256, runtime=RUNTIME)
    config = predictor.configuration
    if (
        config.parameters != PARAMETERS
        or config.training_policy != POLICY
        or config.training_runtime != TRAIN_RUNTIME
        or config.analysis != PROFILE
    ):
        raise ValueError("The original TCN configuration differs from the registered comparison")
    return predictor, entry


def checked_receipt(output: Path):
    receipt = read(output / "reservation.json")
    if receipt["training_protocol"] != training_protocol():
        raise ValueError("The training implementation or policy changed after reservation")
    return receipt


def load_weights(folder: Path, control: TCNPredictor):
    model = deepcopy(control.model)
    with np.load(folder / "weights.npz", allow_pickle=False) as weights:
        model.load_state_dict(
            {key: torch.from_numpy(weights[key]) for key in weights.files}, strict=True
        )
    return replace(control, model=model)


def load_predictors(output: Path, populations: Path):
    receipt = checked_receipt(output)
    folder = output / "fitted"
    frozen = read(folder / "freeze.json")
    if (
        frozen["reservation_sha256"] != sha256(output / "reservation.json")
        or frozen["control_sha256"] != receipt["control_sha256"]
        or frozen["weights_sha256"] != sha256(folder / "weights.npz")
        or frozen["history_sha256"] != sha256(folder / "history.json")
    ):
        raise ValueError("The frozen weights, training history or reservation changed")
    control, _ = load_control(populations, frozen["control_sha256"])
    return {"original": control, "refined": load_weights(folder, control)}


def check_preprocessing(control: TCNPredictor, fitted: TCNPredictor) -> None:
    for key in ("fill", "mean", "scale"):
        np.testing.assert_array_equal(
            getattr(control.preprocessor, key), getattr(fitted.preprocessor, key)
        )


def fit(output: Path, populations: Path) -> None:
    receipt = checked_receipt(output)
    reservation_digest = sha256(output / "reservation.json")
    control, entry = load_control(populations, receipt["control_sha256"])
    bundle = populations / "fitted/mixed-192/bundle"
    corpus = Corpus(populations / "development/mixed-192", expected_digest=DATASET)
    data = training_data(corpus)
    if (
        data.labels.content_sha256 != entry["training_labels_sha256"]
        or data.validation_references.content_sha256 != entry["validation_reference_sha256"]
    ):
        raise ValueError("The original training or validation references changed")
    training = observation_inputs(data.train)
    validation = evaluator(data.validation)
    original = np.concatenate(
        [
            dense_labels(intervals, info.step_count)
            for intervals, info in zip(_references(data.train), training.info, strict=True)
        ]
    )
    refined = refined_targets(corpus.dataset, data.train)
    if not np.array_equal(original < 0, refined < 0):
        raise ValueError("Refinement changed the training availability mask")
    backend = TCNBackend()
    with output_directory(output / "fitted") as dest:
        frozen = {}
        for name, targets in (("original", original), ("refined", refined)):
            print(f"Fitting {name} targets", flush=True)
            result = backend.fit(
                TrainingInputs(training, targets),
                validation.inputs,
                parameters=PARAMETERS,
                policy=POLICY,
                analysis=PROFILE,
                runtime=TRAIN_RUNTIME,
                evaluator=validation,
                legacy_regression=control.preprocessor.legacy_regression,
            )
            fitted = result.predictor
            check_preprocessing(control, fitted)
            history = UpdateHistoryRecord.model_validate({"updates": result.history})
            if name == "original":
                if history != UpdateHistoryRecord.model_validate_json(
                    (bundle / "history.json").read_bytes()
                ):
                    raise ValueError("Control replay did not reproduce the complete native history")
                for key, value in fitted.model.state_dict().items():
                    if not torch.equal(value, control.model.state_dict()[key]):
                        raise ValueError("Control replay did not reproduce the native weights")
                frozen["original"] = {
                    "per_seed": per_seed_history(bundle / "history.json", POLICY),
                    "exact_replay": True,
                }
                continue
            path = dest / "history.json"
            write(path, history.model_dump(mode="json"))
            seeds = per_seed_history(path, POLICY)
            np.savez(
                dest / "weights.npz",
                **{k: v.detach().cpu().numpy() for k, v in fitted.model.state_dict().items()},
            )
            restored = load_weights(dest, control)
            fitted_probabilities = fitted.predict(validation.inputs, runtime=RUNTIME)
            np.testing.assert_array_equal(
                restored.predict(validation.inputs, runtime=RUNTIME), fitted_probabilities
            )
            if validation(fitted_probabilities) != result.validation_score:
                raise ValueError("Reloaded weights did not reproduce the selected validation score")
            frozen["refined"] = {
                "per_seed": seeds,
                "mean_validation_interval_f1": mean(x["interval_macro_f1"] for x in seeds),
                "selected_seed": result.selected_seed,
                "selected_update": result.selected_settings["update"],
                "selected_validation_interval_f1": result.validation_score,
            }
        if reservation_digest != sha256(output / "reservation.json") or receipt != checked_receipt(
            output
        ):
            raise ValueError("The reservation changed during training")
        write(
            dest / "freeze.json",
            {
                **frozen,
                "control_sha256": entry["bundle_sha256"],
                "reservation_sha256": reservation_digest,
                "weights_sha256": sha256(dest / "weights.npz"),
                "history_sha256": sha256(dest / "history.json"),
                "original_training_references": data.labels.content_sha256,
                "validation_references": data.validation_references.content_sha256,
                "refined_target_sha256": hashlib.sha256(
                    refined.astype("<i8").tobytes()
                ).hexdigest(),
                "rows": len(refined),
                "changed_targets": int(np.count_nonzero(refined != original)),
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Reserved pause-target comparison")
    parser.add_argument(
        "--populations", type=Path, required=True, help="Completed training-population comparison"
    )
    args = parser.parse_args()
    fit(args.output, args.populations)
