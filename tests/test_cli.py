"""The CLI exposes only current contracts and persistent service operations."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from support.report_fixtures import prediction_for
from typer.testing import CliRunner

from regimex.cli import app
from regimex.exports import ExportV2

runner = CliRunner()


@pytest.mark.parametrize("args", [[], ["--help"]])
def test_usage(args: list[str]) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output
    assert "service" in result.output and "report" in result.output


def test_version_and_module_entry_point() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and result.output.strip() == "regimex 0.1.0"
    completed = subprocess.run(
        [sys.executable, "-m", "regimex", "--version"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "NO_COLOR": "1"},
    )
    assert completed.stdout.strip() == "regimex 0.1.0"


@pytest.mark.parametrize(
    "command",
    [
        "generate",
        "prepare",
        "train",
        "predict",
        "evaluate",
        "evaluate-external",
        "reference",
        "review",
        "segment",
    ],
)
def test_removed_workflow_commands_are_rejected(command: str) -> None:
    result = runner.invoke(app, [command])
    assert result.exit_code == 2
    assert "No such command" in result.output


def test_cli_does_not_import_scientific_dependencies() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import regimex.cli; "
            "assert not {'numpy', 'sklearn', 'torch', 'matplotlib', 'plotly'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_current_export_validation_and_checksum_failure(tmp_path: Path) -> None:
    source = Path(__file__).parent / "fixtures" / "experiment-v2.json"
    result = runner.invoke(app, ["validate", "--input", str(source)])
    assert result.exit_code == 0 and "Valid snapshot:" in result.output
    data = json.loads(source.read_text())
    data["source"]["run_id"] = "changed-run"
    tampered = tmp_path / "changed.json"
    tampered.write_text(json.dumps(data))
    result = runner.invoke(app, ["validate", "--input", str(tampered)])
    assert result.exit_code == 1
    assert "SHA-256 mismatch" in result.output


def test_previous_format_is_not_implicitly_imported(tmp_path: Path) -> None:
    source = tmp_path / "previous.json"
    source.write_text(json.dumps({"schema_version": "1.0", "experiment_id": "retired", "seed": 1}))
    result = runner.invoke(app, ["validate", "--input", str(source)])
    assert result.exit_code == 1


def test_schema_export_stays_lightweight_and_excludes_previous_format(tmp_path: Path) -> None:
    output = tmp_path / "schemas"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from typer.testing import CliRunner; from regimex.cli import app; "
            "result = CliRunner().invoke(app, ['schema', '--output', sys.argv[1]]); "
            "assert result.exit_code == 0, result.output; "
            "assert not {'numpy', 'sklearn', 'torch', 'matplotlib', 'plotly'} & sys.modules.keys()",
            str(output),
        ],
        check=True,
    )
    assert {path.name for path in output.iterdir()} == {
        "annotations.schema.json",
        "experiment-v2.schema.json",
        "architecture.schema.json",
        "model-bundle.schema.json",
        "operation.schema.json",
        "raw-snapshot.schema.json",
        "raw-dataset.schema.json",
        "reference-set.schema.json",
    }
    observation = json.loads((output / "experiment-v2.schema.json").read_text())
    assert observation["properties"]["schema_version"]["const"] == "2.1"
    assert "v1_import" not in observation["properties"]


def test_report_renders_current_exports_and_blind_review(tmp_path: Path) -> None:
    pytest.importorskip("plotly")
    source = Path(__file__).parent / "fixtures" / "experiment-v2.json"
    saved = tmp_path / "result.json"
    saved.write_text(
        prediction_for(ExportV2.model_validate_json(source.read_bytes())).model_dump_json()
    )
    for blind in (False, True):
        output = tmp_path / str(blind)
        arguments = ["report", "--input", str(source), "--output", str(output)]
        if blind:
            arguments.append("--blind")
        else:
            arguments.extend(("--result", str(saved)))
        result = runner.invoke(app, arguments)
        assert result.exit_code == 0, result.output
        assert (output / "index.html").is_file()
        assert (output / "report-manifest.json").is_file()


def test_report_requires_saved_predictions_and_blind_review_rejects_them(tmp_path: Path) -> None:
    source = Path(__file__).parent / "fixtures" / "experiment-v2.json"
    output = tmp_path / "report"
    arguments = ["report", "--input", str(source), "--output", str(output)]
    missing = runner.invoke(app, arguments)
    assert missing.exit_code == 1 and "require --result" in missing.output
    assert not output.exists()
    saved = tmp_path / "result.json"
    saved.write_text(
        prediction_for(ExportV2.model_validate_json(source.read_bytes())).model_dump_json()
    )
    rejected = runner.invoke(app, [*arguments, "--blind", "--result", str(saved)])
    assert rejected.exit_code == 1 and "does not accept model results" in rejected.output
    assert not output.exists()


def test_rendering_saved_results_does_not_import_training_or_image_libraries(
    tmp_path: Path,
) -> None:
    pytest.importorskip("plotly")
    source = Path(__file__).parent / "fixtures" / "experiment-v2.json"
    saved = tmp_path / "result.json"
    saved.write_text(
        prediction_for(ExportV2.model_validate_json(source.read_bytes())).model_dump_json()
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from typer.testing import CliRunner; from regimex.cli import app; "
            "result = CliRunner().invoke(app, ['report', '--input', sys.argv[1], "
            "'--result', sys.argv[2], '--output', sys.argv[3]]); "
            "assert result.exit_code == 0, result.output; "
            "assert not {'torch', 'sklearn', 'matplotlib', 'regimex.segmentation'} "
            "& sys.modules.keys()",
            str(source),
            str(saved),
            str(tmp_path / "report"),
        ],
        check=True,
    )
