"""Exercise the actual locked Compose service with artificial saved-model fixtures.

Run only against a disposable Docker engine/project. The helper creates private
credentials, a local test CA, its own volumes and PostgreSQL database, and always
tears down that project. It neither publishes images nor reads reserved data.
Docker and host permission to chown secret files to container users are required.
"""

import argparse
import hashlib
import json
import os
import secrets
import shutil
import socket
import ssl
import subprocess
import tempfile
import time
from pathlib import Path
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

type JSONValue = str | int | float | bool | list[JSONValue] | dict[str, JSONValue] | None


def text(value: dict[str, JSONValue], key: str) -> str:
    item = value[key]
    if not isinstance(item, str):
        raise AssertionError("a service identity field is not a string")
    return item


def command(
    arguments: list[str], environment: dict[str, str], *, required: bool = True
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        arguments, env=environment, stdin=subprocess.DEVNULL, capture_output=True, timeout=1800
    )
    if required and result.returncode:
        categories = {"build", "pull", "up", "exec", "run", "restart", "config", "down"}
        category = next((value for value in arguments if value in categories), "other")
        names = {
            "postgres",
            "migrate",
            "api",
            "analysis-worker",
            "admin-worker",
            "maintenance-worker",
            "ops",
        }
        services = sorted({value for value in arguments if value in names})
        # Never include argv, environment or arbitrary stdout: the private token
        # reader and provisioning paths deliberately stay out of CI diagnostics.
        raise RuntimeError(
            f"Compose command category={category} service={','.join(services) or 'project'} "
            f"exit_code={result.returncode}"
        )
    return result


def owner(path: Path, uid: int) -> None:
    if os.geteuid() == 0:
        os.chown(path, uid, uid)
    else:
        subprocess.run(
            ["sudo", "-n", "chown", f"{uid}:{uid}", str(path)],
            check=True,
            stdout=subprocess.DEVNULL,
        )


def start_services(compose: list[str], environment: dict[str, str], *, build: bool) -> None:
    """Prepare both application images and the pinned runtime database image."""
    command([*compose, "config", "--quiet"], environment)
    if build:
        command([*compose, "build", "api", "analysis-worker", "ops"], environment)
    # BuildKit's FROM postgres layer cache is not the daemon's runtime image store.
    # A clean engine still needs this image before the no-pull service startup.
    command([*compose, "pull", "postgres"], environment)
    command(
        [
            *compose,
            "up",
            "--detach",
            "--pull",
            "never",
            "postgres",
            "migrate",
            "api",
            "analysis-worker",
            "admin-worker",
            "maintenance-worker",
        ],
        environment,
    )


class API:
    def __init__(self, origin: str, ca: Path, tokens: dict[str, str]) -> None:
        self.origin = origin
        self.context = ssl.create_default_context(cafile=str(ca))
        self.tokens = tokens

    def raw(
        self,
        path: str,
        *,
        owner: str | None = "alice",
        body: dict[str, JSONValue] | None = None,
        key: str | None = None,
        expected: int = 200,
    ) -> bytes:
        headers = {}
        if owner is not None:
            headers["Authorization"] = "Bearer " + self.tokens[owner]
        if body is not None:
            headers["Content-Type"] = "application/json"
        if key is not None:
            headers["Idempotency-Key"] = key
        request = Request(
            self.origin + path,
            data=None if body is None else json.dumps(body).encode(),
            headers=headers,
        )
        try:
            with urlopen(request, context=self.context, timeout=15) as response:
                status = response.status
                payload = response.read(16 * 1024**2 + 1)
        except HTTPError as error:
            try:
                status = error.code
                payload = error.read(65537)
            finally:
                error.close()
        if not isinstance(payload, bytes):
            raise AssertionError("The deployment HTTP body is not bytes.")
        if status != expected or len(payload) > 16 * 1024**2:
            raise AssertionError("The deployment HTTP response has an unexpected status or size.")
        return payload

    def json(
        self,
        path: str,
        *,
        owner: str | None = "alice",
        body: dict[str, JSONValue] | None = None,
        key: str | None = None,
        expected: int = 200,
    ) -> dict[str, JSONValue]:
        value = json.loads(self.raw(path, owner=owner, body=body, key=key, expected=expected))
        if not isinstance(value, dict):
            raise AssertionError("The deployment HTTP response is not an object.")
        return cast(dict[str, JSONValue], value)

    def completed(self, run: dict[str, JSONValue]) -> dict[str, JSONValue]:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            status = self.json(text(run, "polling_url"))
            job = status["job"]
            if not isinstance(job, dict) or job.get("state") in {"failed", "cancelled"}:
                raise AssertionError("The real saved-model analysis did not succeed.")
            if job.get("state") == "succeeded":
                if status.get("report_available") is not True:
                    raise AssertionError("The successful analysis has no committed offline report.")
                return status
            time.sleep(0.2)
        raise AssertionError("The actual analysis worker timed out.")


def run(
    files: list[Path], *, build: bool, s3_credentials: Path | None = None, s3_ca: Path | None = None
) -> None:
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError(
            "Docker is unavailable; no container runtime verification was performed."
        )
    workspace = Path(__file__).resolve().parents[2]
    project = "regimex-smoke-" + uuid4().hex[:12]
    with tempfile.TemporaryDirectory(prefix="regimex-compose-smoke-") as temporary:
        root = Path(temporary)
        environment = dict(os.environ)
        if (s3_credentials is None) != (s3_ca is None):
            raise ValueError("S3 smoke requires both a credentials file and trusted CA")
        secret_root = root / "secrets"
        secret_root.mkdir(mode=0o700)
        if s3_credentials is not None and s3_ca is not None:
            shutil.copyfile(s3_credentials, secret_root / "s3-credentials")
            (secret_root / "s3-credentials").chmod(0o600)
            owner(secret_root / "s3-credentials", 10001)
            shutil.copyfile(s3_ca, secret_root / "s3-ca.crt")
        password, application_password = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        for name, value, uid in (
            ("database-password", password, 999),
            ("application-password", application_password, 999),
            (
                "database-dsn",
                f"postgresql://regimex:{application_password}@postgres:5432/regimex",
                10001,
            ),
        ):
            path = secret_root / name
            path.write_text(value + "\n")
            path.chmod(0o600)
            owner(path, uid)
        cert, tls_key = secret_root / "api.crt", secret_root / "api.key"
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-keyout",
                str(tls_key),
                "-out",
                str(cert),
                "-days",
                "1",
                "-subj",
                "/CN=localhost",
                "-addext",
                "subjectAltName=DNS:localhost",
                "-addext",
                "basicConstraints=critical,CA:TRUE",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        tls_key.chmod(0o600)
        owner(tls_key, 10001)
        ca = root / "client-ca.crt"
        shutil.copyfile(cert, ca)
        shutil.copyfile(cert, secret_root / "api-ca.crt")
        with socket.socket() as selected:
            selected.bind(("127.0.0.1", 0))
            port = selected.getsockname()[1]
        origin = f"https://localhost:{port}"
        environment.update(
            REGIMEX_PUBLIC_ORIGIN=origin,
            REGIMEX_SECRETS_DIR=str(secret_root),
            REGIMEX_HTTPS_PORT=str(port),
            REGIMEX_BIND_ADDRESS="127.0.0.1",
        )
        compose = [docker, "compose", "--project-name", project]
        for file in files:
            compose.extend(["--file", str(file.resolve())])
        compose.extend(["--profile", "service", "--profile", "ops"])
        failed = False
        success: dict[str, object] | None = None
        try:
            start_services(compose, environment, build=build)

            def ops(
                *args: str, entrypoint: str | None = None, volumes: tuple[str, ...] = ()
            ) -> bytes:
                arguments = [*compose, "run", "--rm", "--no-deps", "--pull", "never"]
                if entrypoint is not None:
                    arguments.extend(["--entrypoint", entrypoint])
                for volume in volumes:
                    arguments.extend(["--volume", volume])
                return command([*arguments, "ops", *args], environment).stdout

            deadline = time.monotonic() + 180
            for service in ("api", "analysis-worker", "admin-worker", "maintenance-worker"):
                component = (
                    ["--component", "ready", "--ca-file", "/run/secrets/api_ca"]
                    if service == "api"
                    else ["--component", "worker", "--queue", service.removesuffix("-worker")]
                )
                while time.monotonic() < deadline:
                    result = command(
                        [
                            *compose,
                            "exec",
                            "-T",
                            service,
                            "regimex",
                            "service",
                            "health",
                            *component,
                        ],
                        environment,
                        required=False,
                    )
                    if result.returncode == 0:
                        break
                    time.sleep(0.5)
                else:
                    raise AssertionError("A Compose service did not become healthy.")
            for name, role in (("admin", "admin"), ("alice", "user"), ("bob", "user")):
                ops(
                    "service",
                    "token-create",
                    "--owner",
                    "smoke-" + name,
                    "--role",
                    role,
                    "--output",
                    f"/operator-output/{name}.token",
                )
            fixture = json.loads(
                ops(
                    "/smoke/support/container_fixture.py",
                    "--template",
                    "/smoke/fixtures/experiment-v2.json",
                    "--output",
                    "/operator-output/fixture",
                    entrypoint="python",
                    volumes=(f"{workspace / 'tests'}:/smoke:ro",),
                )
            )
            payload = json.loads(
                ops(
                    "-c",
                    (
                        "import json;from pathlib import Path;p=Path('/operator-output');"
                        "print(json.dumps({'tokens':{n:(p/(n+'.token')).read_text().strip() "
                        "for n in ['admin','alice','bob']},"
                        "'export':json.loads((p/'fixture/export.json').read_text())}))"
                    ),
                    entrypoint="python",
                )
            )
            tokens = payload["tokens"]
            api = API(origin, ca, tokens)
            ops(
                "service",
                "register-model",
                "--bundle",
                "/operator-output/fixture/bundle",
                "--manifest-sha256",
                fixture["manifest_sha256"],
                "--token-file",
                "/operator-output/admin.token",
                "--activate",
                "--threads",
                "1",
            )
            # Both image flavors contain the same wheel and assets. Only the CPU
            # flavor has scientific packages; ordinary API startup stays light.
            command(
                [
                    *compose,
                    "exec",
                    "-T",
                    "api",
                    "python",
                    "-c",
                    (
                        "import importlib.util;assert all(importlib.util.find_spec(n) is None "
                        "for n in ['numpy','sklearn','torch','plotly'])"
                    ),
                ],
                environment,
            )
            for client in ("pg_dump", "pg_restore"):
                version = ops("--version", entrypoint="/opt/pg16/" + client).decode()
                if "16.15" not in version:
                    raise AssertionError(
                        "The operator image has an unexpected native client version."
                    )
            snapshot = api.json("/api/v1/snapshots", body=payload["export"], expected=201)
            key = "persistent-container-analysis"
            body: dict[str, JSONValue] = {
                "snapshot_id": snapshot["snapshot_id"],
                "model_version_id": "active",
            }
            analysis = api.json("/api/v1/analyses", body=body, key=key, expected=202)
            completed = api.completed(analysis)
            saved = api.json(text(completed, "result_url"))
            report = api.raw(text(completed, "report_url"))
            if (
                not report.startswith(b"<!doctype html>")
                or b"whole requested completed snapshot" not in report
            ):
                raise AssertionError("The actual worker did not produce a complete offline report.")
            for field in ("result_url", "report_url", "polling_url"):
                api.raw(text(completed, field), owner="bob", expected=404)
            command([*compose, "restart", "api", "analysis-worker"], environment)
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                try:
                    if api.json("/health/ready", owner=None).get("database") is True:
                        break
                except OSError, URLError:
                    pass
                time.sleep(0.25)
            else:
                raise AssertionError("The restarted API did not become ready.")
            if (
                api.json(text(completed, "result_url")) != saved
                or api.raw(text(completed, "report_url")) != report
            ):
                raise AssertionError(
                    "Committed observations/results/report changed across container restart."
                )
            replay = api.json("/api/v1/analyses", body=body, key=key, expected=202)
            if replay["analysis_id"] != analysis["analysis_id"]:
                raise AssertionError("Container restart did not preserve idempotency.")
            second = api.completed(
                api.json("/api/v1/analyses", body=body, key="cold-model-reload", expected=202)
            )
            restored = api.json(text(second, "result_url"))
            if restored["sequences"] != saved["sequences"]:
                raise AssertionError(
                    "The restarted actual worker did not reload the saved model exactly."
                )
            success = {
                "status": "passed",
                "project": project,
                "storage": "configured-compose-backend",
                "fixture": "artificial-automatic",
                "checks": [
                    "migration-before-services",
                    "direct-TLS-CA-SNI",
                    "exact-worker-heartbeats",
                    "lightweight-API-image",
                    "native-PG16-clients",
                    "actual-saved-baseline-analysis",
                    "offline-report",
                    "other-owner-denied",
                    "persistent-restart-idempotency",
                    "fresh-worker-saved-model-reload",
                    "project-teardown",
                ],
                "report_sha256": hashlib.sha256(report).hexdigest(),
            }
        except BaseException as error:
            failed = True
            diagnostic: dict[str, object] = {
                "status": "failed",
                "project": project,
                "error_type": type(error).__name__,
            }
            if isinstance(error, RuntimeError) and str(error).startswith(
                "Compose command category="
            ):
                diagnostic["command"] = str(error)
            # Filter Docker's ps output to known service names and bounded public
            # state fields. Do not inspect environments, mounts, commands or logs.
            try:
                status = command(
                    [*compose, "ps", "--all", "--format", "json"], environment, required=False
                )
                if status.returncode == 0 and len(status.stdout) <= 65536:
                    raw = status.stdout.decode()
                    records = (
                        json.loads(raw)
                        if raw.lstrip().startswith("[")
                        else [json.loads(line) for line in raw.splitlines() if line.strip()]
                    )
                    allowed = {
                        "postgres",
                        "migrate",
                        "api",
                        "analysis-worker",
                        "admin-worker",
                        "maintenance-worker",
                        "ops",
                    }
                    states = []
                    for item in records:
                        if not isinstance(item, dict) or item.get("Service") not in allowed:
                            continue
                        states.append(
                            {
                                name: str(item.get(name, ""))[:64]
                                for name in ("Service", "State", "Health", "ExitCode")
                            }
                        )
                    diagnostic["services"] = states
            except OSError, ValueError, subprocess.TimeoutExpired:
                diagnostic["services"] = "unavailable"
            print(json.dumps(diagnostic), flush=True)
            raise
        finally:
            cleanup_failure: dict[str, object] | None = None
            try:
                cleanup = command(
                    [*compose, "down", "--volumes", "--remove-orphans"], environment, required=False
                )
                if cleanup.returncode:
                    cleanup_failure = {
                        "status": "cleanup_failed",
                        "project": project,
                        "exit_code": cleanup.returncode,
                    }
            except (OSError, subprocess.SubprocessError) as cleanup_error:
                cleanup_failure = {
                    "status": "cleanup_failed",
                    "project": project,
                    "error_type": type(cleanup_error).__name__,
                }
            if cleanup_failure is not None:
                print(json.dumps(cleanup_failure), flush=True)
                if not failed:
                    raise RuntimeError("The disposable Compose project could not be fully removed")
        assert success is not None
        print(json.dumps(success), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose-file", type=Path, action="append", dest="files")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--s3-credentials-file", type=Path)
    parser.add_argument("--s3-ca-file", type=Path)
    args = parser.parse_args()
    run(
        args.files or [Path(__file__).resolve().parents[2] / "compose.yaml"],
        build=not args.no_build,
        s3_credentials=args.s3_credentials_file,
        s3_ca=args.s3_ca_file,
    )
