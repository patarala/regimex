import hashlib
import json
from pathlib import Path

import pytest

from regimex.external_latency import read_latency_evidence


def test_forest_uses_selected_candidate_and_wall_clock(tmp_path: Path) -> None:
    manifest = "a" * 64
    selection = tmp_path / "freeze.json"
    selection.write_text(json.dumps({"bundle_sha256": manifest, "challenger": "forest"}))
    panel = {
        "median": {
            "forest": {"predict_wall_ms": 2, "predict_decode_wall_ms": 6, "predict_cpu_ms": 12},
            "control": {"predict_wall_ms": 1, "predict_decode_wall_ms": 5},
        }
    }
    report = tmp_path / "timings.json"
    report.write_text(
        json.dumps(
            {
                "selection_manifest": manifest,
                "threads": 1,
                "results": {"1024": panel, "256": panel},
            }
        )
    )
    measured = read_latency_evidence(report, manifest_sha256=manifest, selection_file=selection)
    assert measured.report_sha256 == hashlib.sha256(report.read_bytes()).hexdigest()
    assert [item.steps for item in measured.timings] == [256, 1024]
    assert measured.timings[0].predict_ms == 2 and measured.threads == 1
    with pytest.raises(ValueError, match="matching bundle"):
        read_latency_evidence(report, manifest_sha256=manifest)
    selection.write_text(json.dumps({"bundle_sha256": "b" * 64, "challenger": "forest"}))
    with pytest.raises(ValueError, match="another bundle"):
        read_latency_evidence(report, manifest_sha256=manifest, selection_file=selection)


def test_blend_exposes_only_its_recorded_times(tmp_path: Path) -> None:
    manifest = "a" * 64
    report = tmp_path / "latency.json"
    report.write_text(
        json.dumps(
            {
                "manifest_sha256": manifest,
                "median_predict_decode_ms": {"256": {"blend": 17, "release": 9}},
            }
        )
    )
    measured = read_latency_evidence(report, manifest_sha256=manifest)
    assert measured.timings[0].predict_decode_ms == 17
    assert measured.timings[0].predict_ms is None and measured.threads is None
    with pytest.raises(ValueError, match="another bundle"):
        read_latency_evidence(report, manifest_sha256="b" * 64)
