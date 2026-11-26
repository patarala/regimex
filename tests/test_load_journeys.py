"""Keep completed analysis IDs when the k6 process times out."""

import json
import runpy
import subprocess
from pathlib import Path
from uuid import UUID

import pytest

from regimex.exports import ExportV2


@pytest.mark.parametrize("broken_summary", [False, True])
def test_timeout_preserves_verified_jobs_and_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broken_summary: bool
) -> None:
    pytest.importorskip("numpy")
    from regimex import market_export

    root = Path(__file__).resolve().parents[1]
    journey = runpy.run_path(str(root / "scripts/load_journeys.py"))
    snapshot = ExportV2.model_validate_json(
        (root / "tests/fixtures/experiment-v2.json").read_bytes()
    )
    monkeypatch.setattr(market_export, "sample_snapshot", lambda **_: snapshot)
    verified = {"phase": "browse-first-anchor", "analysis_id": str(UUID(int=1))}

    def timeout(command: list[str], **_: object) -> None:
        (tmp_path / "console.log").write_text(
            "REGIMEX_VERIFIED " + json.dumps(verified) + "\n" + 'REGIMEX_VERIFIED {"phase":"'
        )
        if broken_summary:
            (tmp_path / "summary.json").write_text('{"metrics":')
        raise subprocess.TimeoutExpired(command, 1000)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired) as failure:
        journey["run_journey"]("browse", tmp_path, {}, "k6")
    assert json.loads((tmp_path / "verified.jsonl").read_bytes()) == verified
    if broken_summary:
        assert "collecting k6 evidence" in failure.value.__notes__[0]
    else:
        assert json.loads((tmp_path / "journey.json").read_bytes())["completed"] is None
