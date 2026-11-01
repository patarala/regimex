"""Run the deliberate fixture with operator-provided private configuration files."""

import argparse
import os
import stat
import sys
from pathlib import Path

from pydantic import SecretBytes, SecretStr, ValidationError

from regimex.contracts import MAX_INPUT_BYTES
from regimex.exports import parse_export_v2

from .app import FixtureCatalog, FixtureRun, create_gateway
from .client import GatewaySettings
from .jwt import SimulatorJWT


def private_bytes(path: Path, *, maximum: int = 4096) -> bytes:
    """Do not follow links or read shared/operator-owned credential files."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.getuid()
            or details.st_nlink != 1
            or stat.S_IMODE(details.st_mode) & 0o077
            or details.st_size > maximum
        ):
            raise ValueError("Fixture credentials must be private regular files.")
        data = os.read(descriptor, maximum + 1)
        if len(data) > maximum:
            raise ValueError("Fixture credential exceeds its size limit.")
        return data
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jwt-key-file", required=True, type=Path)
    parser.add_argument("--issuer", default="regimex-contract-simulator")
    parser.add_argument("--audience", default="regimex-contract-gateway")
    commands = parser.add_subparsers(dest="command", required=True)
    token = commands.add_parser("token", help="Issue a deliberate fixture JWT to stdout.")
    token.add_argument("subject")
    serve = commands.add_parser("serve", help="Serve one pre-existing completed fixture run.")
    serve.add_argument("--integration-token-file", required=True, type=Path)
    serve.add_argument("--regimex-origin", required=True)
    serve.add_argument("--allow-loopback-http", action="store_true")
    serve.add_argument("--export", required=True, type=Path)
    serve.add_argument("--owner", required=True)
    serve.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    try:
        jwt = SimulatorJWT(
            args.issuer, args.audience, SecretBytes(private_bytes(args.jwt_key_file))
        )
        if args.command == "token":
            print(jwt.issue(args.subject))
            return 0
        with args.export.open("rb") as stream:
            export = parse_export_v2(stream.read(MAX_INPUT_BYTES + 1))
        settings = GatewaySettings(
            source_namespace=export.source.source_id,
            regimex_origin=args.regimex_origin,
            integration_token=SecretStr(
                private_bytes(args.integration_token_file).decode().strip()
            ),
            allow_loopback_http=args.allow_loopback_http,
        )
        if not 1 <= args.port <= 65535:
            raise ValueError
        catalog = FixtureCatalog(settings.source_namespace)
        catalog.register(
            FixtureRun.completed(
                export.source.configuration_id,
                args.owner,
                export,
                results={
                    "fixture": True,
                    "run_id": export.source.run_id,
                    "message": "These deliberate simulator results remain available independently.",
                },
            )
        )
        import uvicorn

        uvicorn.run(
            create_gateway(settings, jwt, catalog),
            host="127.0.0.1",
            port=args.port,
            access_log=False,
            log_level="warning",
        )
        return 0
    except OSError, ValueError, ValidationError:
        # Invalid URLs and private file errors can contain credentials/paths.
        print(
            "Cannot start the contract fixture: check its private configuration.", file=sys.stderr
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
