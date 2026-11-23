"""Terminal entry points for the optional activity journal."""

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    help="Show selected local commands in the web activity journal.", add_completion=False
)
Origin = Annotated[str, typer.Option(envvar="REGIMEX_CLI_ORIGIN", help="Regimex service origin.")]
TokenFile = Annotated[
    Path, typer.Option(envvar="REGIMEX_CLI_TOKEN_FILE", help="User or administrator token file.")
]
DEFAULT_ORIGIN = "http://127.0.0.1:8000"
DEFAULT_TOKEN = Path("var/credentials/user.token")


@app.command("run")
def run(
    command: Annotated[list[str], typer.Argument(help="Local executable and arguments after --.")],
    title: Annotated[str, typer.Option(help="Title to show in Activity.")],
    note: Annotated[
        str | None, typer.Option(help="Optional description for the activity page.")
    ] = None,
    result_file: Annotated[
        Path | None, typer.Option(help="Attach this UTF-8 text file after the command exits.")
    ] = None,
    origin: Origin = DEFAULT_ORIGIN,
    token_file: TokenFile = DEFAULT_TOKEN,
    receipt_dir: Annotated[
        Path, typer.Option(help="Private directory for replay receipts.")
    ] = Path("var/terminal-activity"),
) -> None:
    """Run a local command and record its outcome under your account."""
    from regimex.activity_client import run_activity

    try:
        code = run_activity(
            command,
            title=title,
            note=note,
            result_file=result_file,
            origin=origin,
            token_file=token_file,
            receipt_dir=receipt_dir,
            progress=typer.echo,
            warn=lambda message: typer.echo(message, err=True),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    raise typer.Exit(code)


@app.command("sync")
def sync(
    receipt: Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)],
    origin: Origin = DEFAULT_ORIGIN,
    token_file: TokenFile = DEFAULT_TOKEN,
    interrupted: Annotated[
        bool, typer.Option(help="Close an unfinished receipt after a local process interruption.")
    ] = False,
) -> None:
    """Replay a saved receipt without executing the command."""
    from regimex.activity_client import sync_activity

    try:
        url = sync_activity(receipt, origin=origin, token_file=token_file, interrupted=interrupted)
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"Open in browser: {url}")
