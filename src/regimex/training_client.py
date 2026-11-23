"""Submit a training request to the same queue used by the web workspace."""

import json
from collections.abc import Callable
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel

from regimex.contracts import MAX_INPUT_BYTES
from regimex.service_client import connect, read_file, request


class _AcceptedRun(BaseModel):
    run_id: UUID


async def submit_training(
    input_path: Path,
    *,
    origin: str,
    token_file: Path,
    idempotency_key: UUID | None,
    progress: Callable[[str], None],
) -> None:
    payload = read_file(input_path, MAX_INPUT_BYTES)
    if not isinstance(json.loads(payload), dict):
        raise ValueError("The training request must be a JSON object.")
    async with connect(origin, token_file, timeout=30) as client:
        key = idempotency_key or uuid4()
        progress(f"Idempotency key: {key}")
        response = await request(
            client,
            "training-runs",
            content=payload,
            key=key,
            command="submit-training",
            expected_status=202,
            limit=MAX_INPUT_BYTES + 64 * 1024,
        )
        run = _AcceptedRun.model_validate_json(response)
        progress(f"Open in browser: {client.base_url.join(f'/training/{run.run_id}')}")
