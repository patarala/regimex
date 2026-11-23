"""Build a run's metrics from its verified training evidence and history."""

import json
import math
from collections.abc import Callable

from pydantic import JsonValue

from regimex.admin_contracts import RepeatOutcome, TrainingMetrics, TrainingSpec
from regimex.contracts import MAX_INPUT_BYTES
from regimex.infrastructure.object_store import ObjectIntegrityError
from regimex.model_bundles import BundleMember, ModelBundle
from regimex.model_registry import FixedUpdatePolicy, SingleFitPolicy


def build_training_metrics(
    spec: TrainingSpec, manifest: ModelBundle, *, read_history: Callable[[BundleMember], bytes]
) -> TrainingMetrics:
    member = next(item for item in manifest.members if item.role == "history")
    history: JsonValue = None
    note = None
    if member.path.endswith(".json") and member.size_bytes <= MAX_INPUT_BYTES:
        data = read_history(member)
        try:
            history = json.loads(data, parse_float=_history_number, parse_constant=_history_number)
        except (ValueError, RecursionError) as exc:
            raise ObjectIntegrityError("model training history is invalid JSON") from exc
    else:
        note = "The model's history format or size is outside the browser preview limits."
    return TrainingMetrics(
        model_version_id=manifest.model_version_id,
        selected_seed=manifest.training.selected_seed,
        validation_segment_macro_f1=manifest.training.validation_segment_macro_f1,
        history=history,
        history_note=note,
        repeat_outcomes=_repeat_outcomes(spec, history),
    )


def _history_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite metric")
    return number


def _repeat_outcomes(spec: TrainingSpec, history: JsonValue) -> tuple[RepeatOutcome, ...]:
    if spec.architecture_id != "tcn" or not isinstance(history, dict):
        return ()
    policy = spec.policy
    if isinstance(policy, SingleFitPolicy):
        return ()
    unit = "update" if isinstance(policy, FixedUpdatePolicy) else "epoch"
    rows = history.get(unit + "s")
    if not isinstance(rows, list):
        return ()
    # The native codec verifies every seed's complete history before publication.
    # An epoch repeat that finishes below its budget can only have exhausted patience.
    last_step: dict[int, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            return ()
        seed, step = row.get("seed"), row.get(unit)
        if type(seed) is not int or type(step) is not int:
            return ()
        last_step[seed] = step
    if tuple(last_step) != policy.seeds:
        return ()
    if isinstance(policy, FixedUpdatePolicy):
        return tuple(
            RepeatOutcome(seed=seed, step=step, reason="update_limit")
            for seed, step in last_step.items()
            if step == policy.updates
        )
    return tuple(
        RepeatOutcome(
            seed=seed,
            step=step,
            reason="epoch_limit" if step == policy.epochs else "early_stopping",
        )
        for seed, step in last_step.items()
    )
