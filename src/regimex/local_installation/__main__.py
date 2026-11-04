"""Command-line entry point for a trusted local release installation."""

import argparse
import sys
from pathlib import Path
from zipfile import BadZipFile

from .install import Installer, Options


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install or update a local Linux/systemd-user Regimex service."
    )
    parser.add_argument(
        "--archive", type=Path, required=True, help="Trusted release ZIP, including saved models."
    )
    parser.add_argument(
        "--sha256", required=True, help="Archive checksum obtained from a trusted source."
    )
    parser.add_argument(
        "--root", type=Path, default=Path("var"), help="Private persistent installation directory."
    )
    parser.add_argument(
        "--postgres-bin", type=Path, help="Installed PostgreSQL 16 binary directory."
    )
    parser.add_argument("--name", default="regimex", help="Stable systemd user-unit prefix.")
    parser.add_argument("--port", type=int, default=8000, help="Loopback HTTP port.")
    parser.add_argument(
        "--postgres-port", type=int, default=55439, help="Private Unix-socket PostgreSQL port."
    )
    parser.add_argument(
        "--adopt",
        action="store_true",
        help="Adopt an existing documented native var/ installation.",
    )
    args = parser.parse_args()
    try:
        Installer(Options(**vars(args))).install()
    except (OSError, ValueError, RuntimeError, BadZipFile) as error:
        print(f"Installation stopped: {error}", file=sys.stderr)
        return 1
    print(f"Regimex is ready at http://127.0.0.1:{args.port}/")
    print(f"Sign in with the token in {args.root.resolve() / 'credentials/user.token'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
