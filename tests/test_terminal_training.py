"""Keep offline training imports opt-in and tie external scores to the frozen model."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from regimex.external_evaluations import read_external_evaluations
from regimex.storage import digest, write_json
from regimex.terminal_training import register_completed_bundle, register_frozen_selection


def test_offline_registration_uses_only_the_committed_bundle_and_token_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("REGIMEX_TRAINING_TOKEN_FILE", raising=False)
    run = Mock()
    monkeypatch.setattr("regimex.terminal_training.subprocess.run", run)
    register_completed_bundle(tmp_path / "bundle", "0" * 64)
    register_frozen_selection(tmp_path)
    run.assert_not_called()
    token = tmp_path / "admin.token"
    token.write_text("private example")
    monkeypatch.setenv("REGIMEX_TRAINING_TOKEN_FILE", str(token))
    selection = tmp_path / "selection"
    selection.mkdir()
    frozen = selection / "freeze.json"
    write_json(frozen, {"challenger": None})
    register_frozen_selection(tmp_path)
    run.assert_not_called()
    manifest = "1" * 64
    write_json(frozen, {"challenger": "blend", "bundle_sha256": manifest})
    final = tmp_path / "native.json"
    register_frozen_selection(tmp_path, native_final=final)
    run.assert_called_once_with(
        [
            sys.executable,
            "-m",
            "regimex",
            "service",
            "register-model",
            "--bundle",
            str(selection / "bundle"),
            "--manifest-sha256",
            manifest,
            "--token-file",
            str(token),
            "--selection-file",
            str(frozen),
            "--native-final",
            str(final),
        ],
        check=True,
    )


def test_unavailable_service_keeps_offline_results_and_prints_a_retry_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    token = tmp_path / "admin token"
    token.write_text("private example")
    monkeypatch.setenv("REGIMEX_TRAINING_TOKEN_FILE", str(token))
    run = Mock(side_effect=subprocess.CalledProcessError(1, ["registration"]))
    monkeypatch.setattr("regimex.terminal_training.subprocess.run", run)
    bundle = tmp_path / "local bundle"
    bundle.mkdir()
    quality = tmp_path / "direction scores.json"
    register_completed_bundle(bundle, "1" * 64, direction_quality=quality)
    error = capsys.readouterr().err
    assert f"--direction-quality '{quality}'" in error
    assert bundle.is_dir() and str(bundle) in error
    assert "Retry" in error and f"--token-file '{token}'" in error
    assert "private example" not in error


@pytest.mark.parametrize("nested", [False, True])
def test_controlled_reports_preserve_the_failed_gate_and_original_bytes(
    tmp_path: Path, nested: bool
) -> None:
    manifest, protocol = "1" * 64, "3" * 64
    freeze = tmp_path / "freeze.json"
    report = tmp_path / "transfer.json"
    write_json(
        freeze,
        {"challenger": "forest", "bundle_sha256": manifest, "controlled_plan_sha256": protocol},
    )
    gates = {
        "pooled": False,
        "stability": True,
        "per_class": {
            "sideways": False,
            "growth": True,
            "decline": False,
        },
    }
    data = {
        "challenger": "forest",
        "freeze_sha256": digest(freeze),
        "plan_sha256": protocol,
        "paired_paths": 384,
        "checks": {"forest": gates} if nested else gates,
        "results": {"forest": {"interval": {"pooled": {"macro_f1": 0.85}}}},
    }
    write_json(report, data)
    options = dict(selection_file=freeze, native_final=None, controlled_final=report)
    (evidence,) = read_external_evaluations(manifest, **options)
    assert evidence.kind == "controlled_transfer" and not evidence.passed
    assert evidence.bundle_sha256 == manifest and evidence.protocol_sha256 == protocol
    assert evidence.freeze_sha256 == digest(freeze) and evidence.report_sha256 == digest(report)
    assert evidence.interval_macro_f1 == 0.85 and evidence.point_macro_f1 is None
    with pytest.raises(ValueError, match="another bundle"):
        read_external_evaluations("0" * 64, **options)
    with pytest.raises(ValueError, match="require --selection-file"):
        read_external_evaluations(
            manifest, selection_file=None, native_final=None, controlled_final=report
        )
    write_json(report, {**data, "plan_sha256": "4" * 64})
    with pytest.raises(ValueError, match="Invalid controlled_transfer"):
        read_external_evaluations(manifest, **options)
