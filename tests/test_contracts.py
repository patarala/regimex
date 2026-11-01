"""Contract boundaries, missing-value semantics, and hostile inputs."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from regimex.contracts import Observation
from regimex.storage import json_bytes, output_directory


def observation(step: int, **changes: Any) -> Observation:
    return Observation(
        instrument_id="A/Q",
        step=step,
        last_price=changes.pop("last_price", 100.0),
        trade_count=0,
        trade_volume=0.0,
        buy_order_count=0,
        buy_order_volume=0.0,
        sell_order_count=0,
        sell_order_volume=0.0,
        best_bid=None,
        best_ask=None,
        **changes,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("last_price", -1.0),
        ("last_price", float("nan")),
        ("trade_count", -1),
        ("trade_count", True),
        ("trade_volume", 1.0),
        ("best_bid", 101.0),
    ],
)
def test_invalid_observations(field: str, value: object) -> None:
    data = observation(0).model_dump(mode="json")
    data[field] = value
    if field == "best_bid":
        data["best_ask"] = 100.0
    with pytest.raises(ValidationError):
        Observation.model_validate_json(json.dumps(data))


def test_atomic_output_preserves_existing_directory_and_cleans_failed_output(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    original = existing / "original.json"
    original.write_bytes(json_bytes({"original": True}))
    with pytest.raises(ValueError, match="already exists"), output_directory(existing):
        pytest.fail("Existing output must not be replaced")
    assert json.loads(original.read_bytes()) == {"original": True}
    folder = tmp_path / "run"
    with pytest.raises(RuntimeError), output_directory(folder) as pending:
        (pending / "partial").write_text("unfinished")
        raise RuntimeError("failed")
    assert not folder.exists()
    assert not list(tmp_path.glob(".run-*"))


def test_canonical_json_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="JSON"):
        json_bytes({"value": float("nan")})
