"""Optional Training history for completed bundles produced by offline scripts."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def register_completed_bundle(
    bundle: Path,
    manifest_sha256: str,
    *,
    selection_file: Path | None = None,
    native_final: Path | None = None,
    controlled_final: Path | None = None,
) -> None:
    token_file = os.environ.get("REGIMEX_TRAINING_TOKEN_FILE")
    if not token_file:
        return
    command = [
        sys.executable,
        "-m",
        "regimex",
        "service",
        "register-model",
        "--bundle",
        str(bundle),
        "--manifest-sha256",
        manifest_sha256,
        "--token-file",
        token_file,
    ]
    for option, path in (
        ("--selection-file", selection_file),
        ("--native-final", native_final),
        ("--controlled-final", controlled_final),
    ):
        if path is not None:
            command.extend((option, str(path)))
    try:
        subprocess.run(command, check=True)
    except OSError, subprocess.CalledProcessError:
        print(
            f"Training history registration failed. The local bundle is saved at {bundle}.\n"
            f"Retry from {Path.cwd()} with:\n{shlex.join(command)}",
            file=sys.stderr,
        )


def register_frozen_selection(
    comparison: Path, *, native_final: Path | None = None, controlled_final: Path | None = None
) -> None:
    if not os.environ.get("REGIMEX_TRAINING_TOKEN_FILE"):
        return
    selection = comparison / "selection"
    freeze = selection / "freeze.json"
    # Use the digest committed at selection, even if the bundle on disk later changes.
    record = json.loads(freeze.read_bytes())
    if record["challenger"] is not None:
        register_completed_bundle(
            selection / "bundle",
            record["bundle_sha256"],
            selection_file=freeze if native_final or controlled_final else None,
            native_final=native_final,
            controlled_final=controlled_final,
        )
