"""Current observation tools and persistent service operations."""

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from regimex import __version__
from regimex.contracts import StrictModel
from regimex.service_cli import app as service_app
from regimex.storage import output_directory

app = typer.Typer(
    help="Inspect current market snapshots and administer the analysis service.",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
app.add_typer(service_app, name="service")


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"regimex {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_show_version, is_eager=True),
    ] = False,
) -> None:
    """Show help when no command is given."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


OutputPath = Annotated[Path, typer.Option("--output")]
InputPath = Annotated[Path, typer.Option("--input", exists=True, dir_okay=False, readable=True)]


def _execute(action: Callable[[], None]) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        action()
    except ModuleNotFoundError as exc:
        if exc.name != "plotly":
            raise
        typer.echo("Install report dependencies with uv sync --locked --extra viz", err=True)
        raise typer.Exit(1) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command("schema")
def schema_command(output: OutputPath) -> None:
    """Write the supported observation and service JSON Schemas."""

    def action() -> None:
        from regimex.data_catalog import DatasetManifest, ReferenceSet, SourceSnapshot
        from regimex.directions import DirectionAnnotations
        from regimex.exports import ExportV2
        from regimex.model_bundles import ModelBundle
        from regimex.model_registry import ArchitectureInfo
        from regimex.operations import Operation

        models: tuple[tuple[str, type[StrictModel]], ...] = (
            ("annotations", DirectionAnnotations),
            ("experiment-v2", ExportV2),
            ("architecture", ArchitectureInfo),
            ("model-bundle", ModelBundle),
            ("operation", Operation),
            ("raw-snapshot", SourceSnapshot),
            ("raw-dataset", DatasetManifest),
            ("reference-set", ReferenceSet),
        )
        with output_directory(output) as root:
            for name, model in models:
                (root / f"{name}.schema.json").write_text(
                    json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

    _execute(action)


@app.command("validate")
def validate_command(input_path: InputPath) -> None:
    """Validate a current observation export, including its content digest."""

    def action() -> None:
        from regimex.exports import load_export_v2

        export = load_export_v2(input_path)
        typer.echo(f"Valid snapshot: {export.source.snapshot_id}")

    _execute(action)


@app.command("report")
def report_command(
    inputs: Annotated[
        list[Path], typer.Option("--input", exists=True, dir_okay=False, readable=True)
    ],
    output: OutputPath,
    result_paths: Annotated[
        list[Path] | None,
        typer.Option(
            "--result",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Saved model result JSON; provide one per input snapshot.",
        ),
    ] = None,
    samples: Annotated[int, typer.Option(min=1, max=100)] = 12,
    blind: Annotated[
        bool, typer.Option(help="Show only observations for independent human review.")
    ] = False,
) -> None:
    """Render current snapshots with saved model predictions, or blind observations.

    Repeat --input and --result for multiple experiments; results match by input hash.
    Blind review accepts one experiment and no results.
    Generation, fitting, evaluation and publication use the administrative API.
    """

    def action() -> None:
        if blind and result_paths:
            raise ValueError("Blind review does not accept model results")
        if not blind and not result_paths:
            raise ValueError("Prediction reports require --result for every input snapshot")
        from regimex.application.service import AnalysisResult
        from regimex.contracts import MAX_INPUT_BYTES
        from regimex.data_catalog import SourceSnapshot, SplitGroup
        from regimex.exports import ExportV2
        from regimex.visualization import render_snapshot_report, review_reference

        snapshots = []
        for path in inputs:
            with path.open("rb") as stream:
                data = stream.read(MAX_INPUT_BYTES + 1)
            if len(data) > MAX_INPUT_BYTES:
                raise ValueError(f"{path.name} exceeds its input size limit")
            payload = json.loads(data)
            if isinstance(payload, dict) and payload.get("version") == "raw-snapshot/1.0":
                snapshot = SourceSnapshot.model_validate_json(data)
            else:
                export = ExportV2.model_validate_json(data)
                snapshot = SourceSnapshot(
                    source_namespace=export.source.source_id,
                    split_group=SplitGroup(
                        namespace=export.source.source_id, group_id=export.source.run_id
                    ),
                    export=export,
                )
            snapshots.append(snapshot)
        if blind:
            if len(snapshots) != 1:
                raise ValueError("Blind review requires exactly one input snapshot")
            review_reference(snapshots[0].export, output)
        else:
            results = []
            for path in result_paths or ():
                with path.open("rb") as stream:
                    data = stream.read(MAX_INPUT_BYTES + 1)
                if len(data) > MAX_INPUT_BYTES:
                    raise ValueError(f"{path.name} exceeds its result size limit")
                results.append(AnalysisResult.model_validate_json(data))
            render_snapshot_report(
                tuple(snapshots), output, results=tuple(results), sample_count=samples
            )
        typer.echo(f"Report: {output / 'index.html'}")

    _execute(action)
