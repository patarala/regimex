"""Restore a model in the same isolated computation environment used for analysis."""

from pathlib import Path

from pydantic import JsonValue

from regimex.application.analysis_computation import restore_predictor
from regimex.model_registry import ArchitectureRegistry
from regimex.service_contracts import ModelRecord
from regimex.storage import json_bytes


def compute(payload: dict[str, JsonValue], root: Path) -> dict[str, JsonValue]:
    model = ModelRecord.model_validate_json(json_bytes(payload["model"]))
    predictor, manifest = restore_predictor(
        root / "model", model, ArchitectureRegistry.discover(), worker_id="doctor"
    )
    return {
        "model_version_id": manifest.model_version_id,
        "bundle_sha256": model.bundle_sha256,
        "inference_policy_sha256": model.inference_policy_sha256,
        "architecture_id": predictor.architecture_id,
    }
