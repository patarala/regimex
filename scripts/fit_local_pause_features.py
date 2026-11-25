"""Fit a TCN with extra local-slope contrasts on the frozen refined training targets."""

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from compare_profile_21 import RUNTIME, Corpus, sha256, write
from compare_tcn_widths import POLICY, training_data
from compare_temporal_architectures import evaluator, read
from compare_training_populations import per_seed_history
from fit_pause_targets import (
    DATASET,
    PARAMETERS,
    TRAIN_RUNTIME,
    refined_targets,
)
from fit_pause_targets import load_predictors as load_target_models
from fit_pause_targets import training_protocol as target_protocol
from local_pause_features import PAUSE_FEATURE_NAMES, local_pause_features

from regimex.application.training import observation_inputs
from regimex.backends.tcn import TCNPredictor, prepare_observations
from regimex.backends.tcn_layout import TEMPORAL_NAMES
from regimex.backends.tcn_update_records import UpdateHistoryRecord
from regimex.features import log_price_channels
from regimex.prediction_inputs import PreparedPredictionInputs
from regimex.storage import output_directory
from regimex.temporal import SELECTION_BATCH_SIZE, TCNClassifier, TemporalPreprocessor, predict_tcn
from regimex.temporal_updates import fit_temporal_updates


def training_protocol():
    return {
        "control_training": target_protocol(),
        "control": "refined",
        "additional_features": list(PAUSE_FEATURE_NAMES),
        "missing_values": "Training median and scale. Pass missingness masks through.",
        "initialization": "Preserve native weights and RNG stream. Zero-initialize added weights.",
        "scripts": {
            p.name: sha256(p)
            for p in (Path(__file__), Path(__file__).with_name("local_pause_features.py"))
        },
    }


def extra_preprocessor(training):
    if (
        training.ndim != 2
        or training.shape[1] != len(PAUSE_FEATURE_NAMES)
        or not len(training)
        or np.isinf(training).any()
    ):
        raise ValueError("Invalid local features for train-only preprocessing")
    fills = []
    for column in training.T:
        known = column[np.isfinite(column)]
        fills.append(np.median(known) if len(known) else 0.0)
    fill = np.array(fills)
    complete = np.where(np.isnan(training), fill, training)
    scale = complete.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return TemporalPreprocessor(fill, complete.mean(axis=0), scale)


def features(inputs, native_preprocessor):
    native = prepare_observations(inputs, native_preprocessor)
    extras = np.concatenate(
        [local_pause_features(log_price_channels(rows)[:, 0]) for rows in inputs.observations]
    )
    return native, extras


def extend(raw, extras, preprocessor):
    return PreparedPredictionInputs(
        np.column_stack((raw.values, preprocessor.transform(extras), np.isnan(extras))),
        raw.offsets,
        raw.info,
    )


def build(width):
    native_width = len(TEMPORAL_NAMES)
    if width != native_width + 2 * len(PAUSE_FEATURE_NAMES):
        raise ValueError("Unexpected model input width")
    model = TCNClassifier(
        native_width, PARAMETERS.channels, PARAMETERS.dropout, PARAMETERS.dilations
    )
    # Keep the native initialization and the next dropout draw identical for the same seed.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        expanded = torch.nn.Conv1d(width, PARAMETERS.channels, 1)
        expanded.weight.zero_()
        expanded.weight[:, :native_width].copy_(model.input.weight)
        expanded.bias.copy_(model.input.bias)
        model.input = expanded
    return model


@dataclass(frozen=True)
class Predictor:
    model: TCNClassifier
    native_preprocessor: TemporalPreprocessor
    extra_preprocessor: TemporalPreprocessor

    def predict(self, inputs, *, runtime):
        torch.set_num_threads(runtime.threads)
        torch.use_deterministic_algorithms(True)
        raw, extras = features(inputs, self.native_preprocessor)
        prepared = extend(raw, extras, self.extra_preprocessor)
        return predict_tcn(self.model, prepared, batch_size=runtime.max_batch_size)


def check_receipt(output, previous):
    receipt = read(output / "reservation.json")
    prior = previous / "fitted" / "freeze.json"
    if (
        receipt["training_protocol"] != training_protocol()
        or receipt["previous_freeze_sha256"] != sha256(prior)
        or receipt["control_weights_sha256"] != read(prior)["weights_sha256"]
    ):
        raise ValueError("The reservation differs from the registered training code or control")
    return receipt


def read_candidate(folder: Path, control: TCNPredictor) -> Predictor:
    width = len(PAUSE_FEATURE_NAMES)
    with np.load(folder / "preprocessing.npz", allow_pickle=False) as stored:
        arrays = [stored[k] for k in ("fill", "mean", "scale")]
    if any(
        a.shape != (width,) or a.dtype != np.float64 or not np.isfinite(a).all() for a in arrays
    ) or np.any(arrays[2] <= 0):
        raise ValueError("Invalid local feature preprocessing")
    model = build(len(TEMPORAL_NAMES) + 2 * width)
    with np.load(folder / "weights.npz", allow_pickle=False) as stored:
        model.load_state_dict({k: torch.from_numpy(stored[k]) for k in stored.files}, strict=True)
    model.eval().requires_grad_(False)
    return Predictor(model, control.preprocessor, TemporalPreprocessor(*arrays))


def load_predictors(output, populations, previous):
    check_receipt(output, previous)
    folder = output / "fitted"
    frozen = read(folder / "freeze.json")
    if frozen["reservation_sha256"] != sha256(output / "reservation.json") or any(
        frozen["artifacts"][n] != sha256(folder / n)
        for n in ("weights.npz", "preprocessing.npz", "history.json")
    ):
        raise ValueError("Frozen model, preprocessing, history or reservation changed")
    control = load_target_models(previous, populations)["refined"]
    return {"control": control, "candidate": read_candidate(folder, control)}


def fit(output, populations, previous):
    receipt = check_receipt(output, previous)
    reservation_digest = sha256(output / "reservation.json")
    control = load_target_models(previous, populations)["refined"]
    prior = read(previous / "fitted/freeze.json")
    corpus = Corpus(populations / "development/mixed-192", expected_digest=DATASET)
    data = training_data(corpus)
    if data.validation_references.content_sha256 != prior["validation_references"]:
        raise ValueError("The validation reference differs from the control fit")
    targets = refined_targets(corpus.dataset, data.train)
    if (
        hashlib.sha256(targets.astype("<i8").tobytes()).hexdigest()
        != prior["refined_target_sha256"]
    ):
        raise ValueError("The ordered training targets differ from the frozen control")
    validation = evaluator(data.validation)
    train, train_extras = features(observation_inputs(data.train), control.preprocessor)
    valid, valid_extras = features(validation.inputs, control.preprocessor)
    preprocessing = extra_preprocessor(train_extras)
    training = extend(train, train_extras, preprocessing)
    validating = extend(valid, valid_extras, preprocessing)
    with output_directory(output / "fitted") as folder:
        model, history, seed = fit_temporal_updates(
            training,
            targets,
            validating,
            build=build,
            interval_weight_power=PARAMETERS.interval_weight_power,
            smoothing=PARAMETERS.smoothing,
            policy=POLICY,
            runtime=TRAIN_RUNTIME,
            scorer=validation,
        )
        write(
            folder / "history.json",
            UpdateHistoryRecord.model_validate({"updates": tuple(history)}).model_dump(mode="json"),
        )
        np.savez(
            folder / "weights.npz", **{k: v.detach().numpy() for k, v in model.state_dict().items()}
        )
        np.savez(
            folder / "preprocessing.npz",
            fill=preprocessing.fill,
            mean=preprocessing.mean,
            scale=preprocessing.scale,
        )
        restored = read_candidate(folder, control)
        selection_runtime = RUNTIME.model_copy(update={"max_batch_size": SELECTION_BATCH_SIZE})
        proba = predict_tcn(model, validating, batch_size=SELECTION_BATCH_SIZE)
        np.testing.assert_array_equal(
            restored.predict(validation.inputs, runtime=selection_runtime), proba
        )
        per_seed = per_seed_history(folder / "history.json", POLICY)
        best = max(per_seed, key=lambda row: row["interval_macro_f1"])
        if best["seed"] != seed or validation(proba) != best["interval_macro_f1"]:
            raise ValueError("Reloaded validation predictions differ from checkpoint selection")
        if (
            sha256(output / "reservation.json") != reservation_digest
            or check_receipt(output, previous) != receipt
        ):
            raise ValueError("The reservation changed during training")
        summary = {
            "reservation_sha256": reservation_digest,
            "previous_freeze_sha256": receipt["previous_freeze_sha256"],
            "artifacts": {
                n: sha256(folder / n) for n in ("weights.npz", "preprocessing.npz", "history.json")
            },
            "ordered_target_sha256": prior["refined_target_sha256"],
            "validation_references": prior["validation_references"],
            "control": prior["refined"],
            "candidate": {
                "per_seed": per_seed,
                "selected_seed": seed,
                "selected_update": best["update"],
                "selected_validation_interval_f1": best["interval_macro_f1"],
                "mean_validation_interval_f1": mean(s["interval_macro_f1"] for s in per_seed),
            },
            "parameters": {
                "control": sum(p.numel() for p in control.model.parameters()),
                "candidate": sum(p.numel() for p in model.parameters()),
            },
        }
        write(folder / "freeze.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--populations", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    args = parser.parse_args()
    fit(args.output, args.populations, args.previous)
