"""Build a run's metrics from its verified training evidence and history."""

import json
import math
from collections.abc import Callable

from pydantic import JsonValue

from regimex.admin_contracts import (
    ImportedTrainingSpec,
    TrainingMetrics,
    TrainingSpec,
    legacy_repeat_outcomes,
)
from regimex.contracts import MAX_INPUT_BYTES
from regimex.infrastructure.object_store import ObjectIntegrityError
from regimex.model_bundles import BundleMember, ModelBundle


def build_training_metrics(
    spec: TrainingSpec | ImportedTrainingSpec,
    manifest: ModelBundle,
    *,
    read_history: Callable[[BundleMember], bytes],
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
        repeat_outcomes=manifest.training.repeat_outcomes or legacy_repeat_outcomes(spec, history),
    )


def _history_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite metric")
    return number
