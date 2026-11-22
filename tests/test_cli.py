"""The CLI exposes only current contracts and persistent service operations."""

import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest
from support.report_fixtures import prediction_for
from typer.testing import CliRunner

from regimex.cli import app
from regimex.exports import ExportPayloadV2, ExportV2, seal_export
from regimex.market_identity import imported_seed_group, market_seed_group

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


@pytest.mark.parametrize("instruments", [1, 2])
def test_generate_writes_a_reproducible_complete_export(tmp_path: Path, instruments: int) -> None:
    pytest.importorskip("numpy")
    args = ["generate", "--seed", "42", "--steps", "32", "--instruments", str(instruments)]
    first = tmp_path / "exports" / "first.json"
    second = tmp_path / "second.json"
    for path in (first, second):
        result = runner.invoke(app, [*args, "--output", str(path)])
        assert result.exit_code == 0, result.output
        assert runner.invoke(app, ["validate", "--input", str(path)]).exit_code == 0
    assert first.read_bytes() == second.read_bytes()
    export = ExportV2.model_validate_json(first.read_bytes())
    assert len(export.instruments) == instruments
    assert export.time_grid.step_count == 32
    assert len(export.observations) == instruments * 32
    assert export.content_sha256 in result.output


def test_generated_run_id_tracks_all_simulation_inputs(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    ids = set()
    for seed, steps, instruments in ((0, 1, 1), (1, 1, 1), (0, 2, 1), (0, 1, 2)):
        path = tmp_path / f"{seed}-{steps}-{instruments}.json"
        args = ["--seed", str(seed), "--steps", str(steps), "--instruments", str(instruments)]
        result = runner.invoke(app, ["generate", *args, "--output", str(path)])
        assert result.exit_code == 0, result.output
        ids.add(ExportV2.model_validate_json(path.read_bytes()).source.run_id)
    assert len(ids) == 4


def test_native_snapshot_with_an_opaque_run_id_needs_its_matching_seed() -> None:
    pytest.importorskip("numpy")
    from regimex.market_export import sample_snapshot

    snapshot = sample_snapshot(seed=17, steps=64, instruments=1)
    value = snapshot.model_dump(mode="json", exclude={"content_sha256"})
    old_id = str(uuid5(NAMESPACE_URL, f"regimex:single-market:{snapshot.source.configuration_id}"))
    value["source"].update(run_id=old_id, snapshot_id=old_id)
    old = seal_export(ExportPayloadV2.model_validate_json(json.dumps(value)))
    group = market_seed_group(17)
    assert old.source.run_id != snapshot.source.run_id
    assert imported_seed_group(snapshot, None) == imported_seed_group(old, group) == group
    with pytest.raises(ValueError, match="requires its original seed"):
        imported_seed_group(old, None)
    with pytest.raises(ValueError, match="sampled market configuration"):
        imported_seed_group(old, market_seed_group(18))
    with pytest.raises(ValueError, match="declared split group"):
        imported_seed_group(snapshot, market_seed_group(18))


def test_generate_leaves_existing_file_in_place(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    path.write_bytes(b"already here")
    result = runner.invoke(app, ["generate", "--seed", "42", "--output", str(path)])
    assert result.exit_code == 1 and "already exists" in result.output
    assert path.read_bytes() == b"already here"


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
    published = Path(__file__).resolve().parents[1] / "schemas"
    exported_names = {path.name for path in output.iterdir()}
    assert exported_names == {path.name for path in published.iterdir()}
    assert exported_names == {
        "annotations.schema.json",
        "experiment-v2.schema.json",
        "architecture.schema.json",
        "model-bundle.schema.json",
        "operation.schema.json",
        "raw-snapshot.schema.json",
        "raw-dataset.schema.json",
        "reference-set.schema.json",
    }
    for path in output.iterdir():
        expected = json.loads((published / path.name).read_bytes())
        assert json.loads(path.read_bytes()) == expected, path.name
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
