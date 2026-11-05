"""TCN fitting with explicit successful-update budgets and checkpoint selection."""

from collections.abc import Callable

import numpy as np
import torch
from pydantic import JsonValue
from torch import Tensor, nn

from regimex.features import IntArray
from regimex.model_registry import FixedUpdatePolicy, TCNParameters, TrainingRuntime
from regimex.prediction_inputs import PreparedPredictionInputs, require_prepared_inputs
from regimex.temporal import (
    SELECTION_BATCH_SIZE,
    Scorer,
    TCNClassifier,
    _boundary_weights,
    _interval_weights,
    _supervised_batch,
    _supervised_loss,
    predict_tcn,
)


def fit_tcn_updates(
    training: PreparedPredictionInputs,
    targets: IntArray,
    validation: PreparedPredictionInputs,
    *,
    parameters: TCNParameters,
    policy: FixedUpdatePolicy,
    runtime: TrainingRuntime,
    scorer: Scorer,
    check_budget: Callable[[], None] | None = None,
) -> tuple[TCNClassifier, list[dict[str, JsonValue]], int]:
    """Fit each seed for the full budget; retain the first best scored checkpoint."""
    require_prepared_inputs(training)
    require_prepared_inputs(validation)
    FixedUpdatePolicy.model_validate(policy.model_dump())
    if (
        targets.dtype != np.int64
        or targets.shape != (len(training.values),)
        or not np.isin(targets, [-1, 0, 1, 2]).all()
        or training.values.shape[1] != validation.values.shape[1]
    ):
        raise ValueError("TCN requires aligned direction targets and prepared features")
    torch.set_num_threads(runtime.threads)
    torch.use_deterministic_algorithms(True)
    labels = targets[targets >= 0]
    counts = np.bincount(labels, minlength=3)
    if (counts == 0).any():
        raise ValueError("TCN requires all three training directions")
    class_weights = torch.tensor(len(labels) / (3 * counts), dtype=torch.float32)
    weights = _interval_weights(targets, training.offsets, parameters.interval_weight_power)
    weights *= _boundary_weights(targets, training.offsets)
    best_state: dict[str, Tensor] | None = None
    best_score, best_seed = -1.0, policy.seeds[0]
    history: list[dict[str, JsonValue]] = []
    for seed in policy.seeds:
        if check_budget is not None:
            check_budget()
        torch.manual_seed(seed)
        rng = np.random.Generator(np.random.PCG64(seed))
        model = TCNClassifier(
            training.values.shape[1], parameters.channels, parameters.dropout, parameters.dilations
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=policy.learning_rate, weight_decay=policy.weight_decay
        )
        update, pass_index = 0, 0
        while update < policy.updates:
            pass_index += 1
            before_pass = update
            ordering = rng.permutation(len(training.info)).astype(np.int64)
            for batch_index, first in enumerate(range(0, len(ordering), policy.batch_size), 1):
                if check_budget is not None:
                    check_budget()
                selected: IntArray = ordering[first : first + policy.batch_size]
                inputs, batch_targets, lengths = _supervised_batch(training, targets, selected)
                eligible = int((batch_targets >= 0).sum())
                if not eligible:
                    continue
                sample_weights = torch.zeros_like(batch_targets, dtype=torch.float32)
                for row, index in enumerate(selected):
                    start, stop = training.offsets[index : index + 2]
                    sample_weights[row, : stop - start] = torch.from_numpy(weights[start:stop])
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = _supervised_loss(
                    model(inputs, lengths),
                    batch_targets,
                    class_weights,
                    sample_weights,
                    parameters.smoothing,
                )
                if not bool(torch.isfinite(loss)):
                    raise ValueError("nonfinite TCN training loss")
                torch.autograd.backward(loss)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                update += 1
                real_rows = int(lengths.sum())
                entry: dict[str, JsonValue] = {
                    "seed": seed,
                    "update": update,
                    "pass_index": pass_index,
                    "batch_in_pass": batch_index,
                    "training_loss": float(loss.detach()),
                    "trained_sequences": len(selected),
                    "real_rows": real_rows,
                    "padding_rows": int(inputs.shape[0] * inputs.shape[1]) - real_rows,
                    "eligible_rows": eligible,
                }
                if update in policy.validation_updates:
                    if check_budget is not None:
                        check_budget()
                    score = scorer(predict_tcn(model, validation, batch_size=SELECTION_BATCH_SIZE))
                    if check_budget is not None:
                        check_budget()
                    if not np.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError("nonfinite or invalid TCN validation score")
                    entry["validation_segment_macro_f1"] = score
                    if score > best_score:
                        best_score, best_seed = score, seed
                        best_state = {
                            name: tensor.detach().clone()
                            for name, tensor in model.state_dict().items()
                        }
                history.append(entry)
                if update == policy.updates:
                    break
            if update == before_pass:
                raise ValueError("TCN completed a pass without any supervised optimizer update")
    if check_budget is not None:
        check_budget()
    assert best_state is not None
    best = TCNClassifier(
        training.values.shape[1], parameters.channels, parameters.dropout, parameters.dilations
    )
    best.load_state_dict(best_state, strict=True)
    return best, history, best_seed
