"""Run an analysis, back it up and restore it into a fresh Compose project.

Requires Docker and permission to assign secret files to the container users.
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
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

type JSONValue = str | int | float | bool | list[JSONValue] | dict[str, JSONValue] | None


def text(value: dict[str, JSONValue], key: str) -> str:
    item = value[key]
    if not isinstance(item, str):
        raise AssertionError("a service identity field is not a string")
    return item


def release_tcn_model(assets: Path) -> dict[str, str]:
    name = "models/tcn/manifest.json"
    data = (assets / name).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    entries = [
        line
        for line in (assets / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        if line.partition("  ")[2] == name
    ]
    if entries != [f"{digest}  {name}"]:
        raise ValueError("The release SHA256SUMS must pin the TCN manifest exactly once.")
    manifest = json.loads(data)
    if manifest["architecture_id"] != "tcn":
        raise ValueError("The supplied release bundle must contain a TCN.")
    return {
        "model_version_id": text(manifest, "model_version_id"),
        "bundle_sha256": digest,
    }


def command(
    arguments: list[str], environment: dict[str, str], *, required: bool = True
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            arguments, env=environment, stdin=subprocess.DEVNULL, capture_output=True, timeout=1800
        )
    except subprocess.TimeoutExpired:
        # TimeoutExpired includes the complete command in its message.
        result = subprocess.CompletedProcess[bytes](arguments, 124, b"", b"")
        failure = "timed_out"
    else:
        failure = f"exit_code={result.returncode}"
    if required and result.returncode:
        categories = {
            "build",
            "pull",
            "up",
            "exec",
            "run",
            "stop",
            "config",
            "down",
            "image",
            "volume",
        }
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
            f"{failure}"
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


def smoke_environment(project: str, *, build: bool, s3: bool) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("REGIMEX_", "COMPOSE_"))
    }
    environment["REGIMEX_IMAGE_TAG"] = (
        project if build else os.environ.get("REGIMEX_IMAGE_TAG", "local")
    )
    if s3:
        for key in ("REGIMEX_S3_ENDPOINT_URL", "REGIMEX_S3_REGION", "REGIMEX_S3_BUCKET"):
            if key in os.environ:
                environment[key] = os.environ[key]
        prefix = os.environ.get("REGIMEX_S3_PREFIX", "").rstrip("/")
        if not prefix:
            raise ValueError("Set REGIMEX_S3_PREFIX to a disposable smoke namespace.")
        environment["REGIMEX_S3_PREFIX"] = f"{prefix}/{project}"
    return environment


def remove_smoke_images(docker: str, project: str, environment: dict[str, str]) -> None:
    listing = command(
        [
            docker,
            "image",
            "ls",
            "--format",
            "{{.Repository}}:{{.Tag}}",
            "--filter",
            f"reference=regimex-*:{project}",
        ],
        environment,
    )
    owned = {f"regimex-{name}:{project}" for name in ("api", "cpu", "ops")}
    tags = sorted(owned.intersection(listing.stdout.decode().splitlines()))
    if tags:
        command([docker, "image", "rm", *tags], environment)


def start_services(compose: list[str], environment: dict[str, str], *, build: bool) -> None:
    """Prepare the application images and the pinned runtime database image."""
    command([*compose, "config", "--quiet"], environment)
    if build:
        command([*compose, "build", "api", "analysis-worker", "ops"], environment)
    # BuildKit's FROM postgres layer cache is not the daemon's runtime image store.
    # A clean engine still needs this image before the no-pull service startup.
    command([*compose, "pull", "postgres"], environment)
    start_empty_database(compose, environment)
    start_application(compose, environment)


def compose_up(compose: list[str], environment: dict[str, str], *services: str) -> None:
    command(
        [
            *compose,
            "up",
            "--detach",
            "--no-build",
            "--pull",
            "never",
            "--wait",
            "--wait-timeout",
            "180",
            *services,
        ],
        environment,
    )


def start_empty_database(compose: list[str], environment: dict[str, str]) -> None:
    compose_up(compose, environment, "postgres")
    # Offline restore inspects this inventory before writing to a fresh deployment.
    inventory = command(
        [
            *compose,
            "exec",
            "-T",
            "postgres",
            "bash",
            "-o",
            "pipefail",
            "-c",
            "pg_dump -U postgres -d regimex --schema-only --format=custom | pg_restore --list",
        ],
        environment,
    )
    if any(line.strip() and not line.startswith(b";") for line in inventory.stdout.splitlines()):
        raise RuntimeError("The newly provisioned database contains objects before migration.")


def start_application(compose: list[str], environment: dict[str, str]) -> None:
    compose_up(
        compose,
        environment,
        "migrate",
        "api",
        "analysis-worker",
        "admin-worker",
        "maintenance-worker",
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


def prepare_secrets(secret_root: Path, *, s3_credentials: Path | None, s3_ca: Path | None) -> Path:
    """Prepare bind-mounted secrets for both container UIDs and return the host's test CA."""
    secret_root.mkdir(mode=0o700)
    if s3_credentials is not None and s3_ca is not None:
        shutil.copyfile(s3_credentials, secret_root / "s3-credentials")
        (secret_root / "s3-credentials").chmod(0o600)
        owner(secret_root / "s3-credentials", 10001)
        provider_ca = secret_root / "s3-ca.crt"
        shutil.copyfile(s3_ca, provider_ca)
        provider_ca.chmod(0o444)
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
    cert.chmod(0o444)
    ca = secret_root / "api-ca.crt"
    shutil.copyfile(cert, ca)
    ca.chmod(0o444)
    return ca


def compose_command(docker: str, files: list[Path], project: str) -> list[str]:
    compose = [docker, "compose", "--env-file", os.devnull, "--project-name", project]
    for file in files:
        compose.extend(["--file", str(file.resolve())])
    return [*compose, "--profile", "service", "--profile", "ops"]


def run_operator(
    compose: list[str],
    environment: dict[str, str],
    *args: str,
    entrypoint: str | None = None,
    volumes: tuple[str, ...] = (),
) -> bytes:
    arguments = [*compose, "run", "--rm", "--no-deps", "-T"]
    if entrypoint is not None:
        arguments.extend(["--entrypoint", entrypoint])
    for volume in volumes:
        arguments.extend(["--volume", volume])
    return command([*arguments, "ops", *args], environment).stdout


def attest_no_writers(path: Path, project: str) -> None:
    path.write_text(
        json.dumps(
            {
                "stopped_at": datetime.now(UTC).isoformat(),
                "statement": f"The disposable project {project} has no running writers.",
                "api_stopped": True,
                "analysis_workers_stopped": True,
                "administrative_workers_stopped": True,
                "maintenance_stopped": True,
                "garbage_collection_stopped": True,
                "exclusive_object_access": True,
            }
        )
    )
    path.chmod(0o444)


def worker_probe_history(
    docker: str, container_id: str, environment: dict[str, str]
) -> list[dict[str, str | int]]:
    if not 12 <= len(container_id) <= 64 or any(c not in "0123456789abcdef" for c in container_id):
        return []
    inspected = command(
        [
            docker,
            "inspect",
            "--type",
            "container",
            "--format",
            "{{json .State.Health.Log}}",
            container_id,
        ],
        environment,
        required=False,
    )
    if inspected.returncode or len(inspected.stdout) > 65536:
        return []
    try:
        checks = json.loads(inspected.stdout)
    except ValueError:
        return []
    if not isinstance(checks, list):
        return []
    history: list[dict[str, str | int]] = []
    for check in checks[-5:]:
        if not isinstance(check, dict) or type(code := check.get("ExitCode")) is not int:
            continue
        output = check.get("Output")
        reason = "probe_error"
        if code == 0:
            reason = "healthy"
        elif isinstance(output, str):
            if code == -1 and output.startswith(
                (
                    "Health check exceeded timeout (",
                    "timed out starting health check for container ",
                )
            ):
                reason = "timed_out"
            elif code == 1:
                try:
                    # Docker combines stderr messages and the probe's final JSON response.
                    probe = json.loads(output.rstrip().rsplit("\n", 1)[-1])
                except ValueError:
                    probe = None
                if (
                    isinstance(probe, dict)
                    and probe.get("component") == "worker"
                    and probe.get("status") == "unhealthy"
                ):
                    reason = "worker_unhealthy"
        history.append({"exit_code": code, "reason": reason})
    return history


@contextmanager
def disposable_project(
    project: str, compose: list[str], environment: dict[str, str]
) -> Iterator[None]:
    failed = False
    try:
        yield
    except BaseException as error:
        failed = True
        diagnostic: dict[str, object] = {
            "status": "failed",
            "project": project,
            "error_type": type(error).__name__,
        }
        if isinstance(error, RuntimeError) and str(error).startswith("Compose command category="):
            diagnostic["command"] = str(error)
        # Only allowlisted probe outcomes may enter diagnostics. A failed
        # probe's raw output can contain credentials or connection details.
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
                states: list[dict[str, object]] = []
                for item in records:
                    if not isinstance(item, dict) or item.get("Service") not in allowed:
                        continue
                    state: dict[str, object] = {
                        name: str(item.get(name, ""))[:64]
                        for name in ("Service", "State", "Health", "ExitCode")
                    }
                    if state["Health"] == "unhealthy" and item["Service"].endswith("-worker"):
                        state["probes"] = worker_probe_history(
                            compose[0], str(item.get("ID", "")), environment
                        )
                    states.append(state)
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
        except (OSError, subprocess.SubprocessError, RuntimeError) as cleanup_error:
            cleanup_failure = {
                "status": "cleanup_failed",
                "project": project,
                "error_type": type(cleanup_error).__name__,
            }
        if cleanup_failure is not None:
            print(json.dumps(cleanup_failure), flush=True)
            if not failed:
                raise RuntimeError("The disposable Compose project could not be fully removed")


def run(
    files: list[Path] | None,
    *,
    build: bool,
    s3_credentials: Path | None = None,
    s3_ca: Path | None = None,
    release_tcn: Path | None = None,
) -> None:
    expected_model = release_tcn_model(release_tcn) if release_tcn else None
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError(
            "Docker is unavailable; no container runtime verification was performed."
        )
    workspace = Path(__file__).resolve().parents[2]
    project = "regimex-smoke-" + uuid4().hex[:12]
    with (
        tempfile.TemporaryDirectory(prefix="regimex-compose-smoke-") as temporary,
        ExitStack() as resources,
    ):
        root = Path(temporary)
        if (s3_credentials is None) != (s3_ca is None):
            raise ValueError("S3 smoke requires both a credentials file and trusted CA")
        environment = smoke_environment(project, build=build, s3=s3_credentials is not None)
        if files is None:
            files = [workspace / "compose.yaml"]
            if s3_credentials is not None:
                files.append(workspace / "compose.s3.yaml")
        secret_root = root / "secrets"
        ca = prepare_secrets(secret_root, s3_credentials=s3_credentials, s3_ca=s3_ca)
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
        compose = compose_command(docker, files, project)
        if build:
            resources.callback(remove_smoke_images, docker, project, environment)
        with disposable_project(project, compose, environment):
            start_services(compose, environment, build=build)
            ops = partial(run_operator, compose, environment)
            backend = "s3" if s3_credentials is not None else "local"
            storage = json.loads(ops("service", "storage-check"))
            if storage["backend"] != backend or storage["retained_versions"] != 0:
                raise AssertionError(f"The smoke deployment requires an empty {backend} store.")
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
            bundle = "/operator-output/baseline" if release_tcn is None else "/release-tcn"
            model_volumes = (
                (f"{release_tcn.resolve() / 'models/tcn'}:{bundle}:ro",) if release_tcn else ()
            )
            fixture = json.loads(
                ops(
                    "/smoke/support/container_fixture.py",
                    "--template",
                    "/smoke/fixtures/experiment-v2.json",
                    *(("--baseline-bundle", bundle) if release_tcn is None else ()),
                    entrypoint="python",
                    volumes=(f"{workspace / 'tests'}:/smoke:ro",),
                )
            )
            if expected_model is None:
                expected_model = cast(dict[str, str], fixture["model"])
            architecture = "tcn" if release_tcn else "baseline"
            tokens = {
                name: ops(f"/operator-output/{name}.token", entrypoint="cat").decode().strip()
                for name in ("alice", "bob")
            }
            api = API(origin, ca, tokens)
            ops(
                "service",
                "register-model",
                "--bundle",
                bundle,
                "--manifest-sha256",
                expected_model["bundle_sha256"],
                "--token-file",
                "/operator-output/admin.token",
                "--activate",
                "--threads",
                "1",
                volumes=model_volumes,
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
            snapshot = api.json("/api/v1/snapshots", body=fixture["export"], expected=201)
            key = "persistent-container-analysis"
            body: dict[str, JSONValue] = {
                "snapshot_id": snapshot["snapshot_id"],
                "model_version_id": "active",
            }
            analysis = api.json("/api/v1/analyses", body=body, key=key, expected=202)
            completed = api.completed(analysis)
            saved_result = api.raw(text(completed, "result_url"))
            saved = json.loads(saved_result)
            if not isinstance(saved, dict):
                raise AssertionError("The saved analysis result is not an object.")
            model = saved["model"]
            if (
                saved["architecture_id"] != architecture
                or not isinstance(model, dict)
                or any(model.get(k) != v for k, v in expected_model.items())
            ):
                raise AssertionError(
                    "The worker used a different model from the registered bundle."
                )
            report = api.raw(text(completed, "report_url"))
            if (
                not report.startswith(b"<!doctype html>")
                or b"whole requested completed snapshot" not in report
            ):
                raise AssertionError("The actual worker did not produce a complete offline report.")
            for field in ("result_url", "report_url", "polling_url"):
                api.raw(text(completed, field), owner="bob", expected=404)
            command(
                [
                    *compose,
                    "up",
                    "--detach",
                    "--no-deps",
                    "--no-build",
                    "--pull",
                    "never",
                    "--force-recreate",
                    "--wait",
                    "--wait-timeout",
                    "90",
                    "api",
                    "analysis-worker",
                ],
                environment,
            )
            if (
                api.json(text(completed, "result_url")) != saved
                or api.raw(text(completed, "report_url")) != report
            ):
                raise AssertionError(
                    "Committed observations/results/report changed across container recreation."
                )
            replay = api.json("/api/v1/analyses", body=body, key=key, expected=202)
            if replay["analysis_id"] != analysis["analysis_id"]:
                raise AssertionError("Container recreation did not preserve idempotency.")
            second = api.completed(
                api.json("/api/v1/analyses", body=body, key="cold-model-reload", expected=202)
            )
            restored = api.json(text(second, "result_url"))
            if restored["sequences"] != saved["sequences"]:
                raise AssertionError("The recreated worker did not reload the saved model exactly.")
            command(
                [*compose, "stop", "api", "analysis-worker", "admin-worker", "maintenance-worker"],
                environment,
            )
            attestation = root / "stopped.json"
            attest_no_writers(attestation, project)
            # The transfer volume outlives the source. Its only contents are the backup.
            transfer = f"{project}-backup"
            command([docker, "volume", "create", transfer], environment)
            resources.callback(command, [docker, "volume", "rm", transfer], environment)
            backup = "/operator-output/backup"
            confirmation = ops(
                "service",
                "backup",
                "--output",
                backup,
                "--attestation-file",
                "/stopped.json",
                volumes=(f"{transfer}:/operator-output", f"{attestation}:/stopped.json:ro"),
            )
            archive = (f"{transfer}:/recovery:ro",)
            manifest = ops("/recovery/backup/manifest.json", entrypoint="cat", volumes=archive)
            manifest_sha256 = hashlib.sha256(manifest).hexdigest()
            if not confirmation.strip().endswith(manifest_sha256.encode()):
                raise AssertionError(
                    "Backup manifest differs from the digest reported by the operator CLI."
                )
            report_digest = hashlib.sha256(report).hexdigest()
            if report_digest not in {obj["sha256"] for obj in json.loads(manifest)["objects"]}:
                raise AssertionError("The backup does not contain the saved report.")
            contents = ops(
                "--list",
                "/recovery/backup/database.dump",
                entrypoint="/opt/pg16/pg_restore",
                volumes=archive,
            )
            if b" TABLE DATA public regimex_operations " not in contents:
                raise AssertionError(
                    "The operator image could not read the archived operations table."
                )

        # The source network, database and local objects have already been removed.
        target = f"{project}-restored"
        target_env = environment.copy()
        target_secrets = root / "target-secrets"
        target_ca = prepare_secrets(target_secrets, s3_credentials=s3_credentials, s3_ca=s3_ca)
        target_env["REGIMEX_SECRETS_DIR"] = str(target_secrets)
        if s3_credentials is not None:
            target_env["REGIMEX_S3_PREFIX"] = f"{environment['REGIMEX_S3_PREFIX']}-restored"
        target_compose = compose_command(docker, files, target)
        with disposable_project(target, target_compose, target_env):
            start_empty_database(target_compose, target_env)
            target_attestation = root / "target-stopped.json"
            attest_no_writers(target_attestation, target)
            run_operator(
                target_compose,
                target_env,
                "service",
                "restore",
                "--input",
                "/recovery/backup",
                "--manifest-sha256",
                manifest_sha256,
                "--attestation-file",
                "/stopped.json",
                volumes=(*archive, f"{target_attestation}:/stopped.json:ro"),
            )
            start_application(target_compose, target_env)
            destination = API(origin, target_ca, tokens)
            if (
                destination.raw(text(completed, "result_url")) != saved_result
                or destination.raw(text(completed, "report_url")) != report
            ):
                raise AssertionError(
                    "The restored deployment returned different saved results or report bytes."
                )
            for field in ("result_url", "report_url", "polling_url"):
                destination.raw(text(completed, field), owner="bob", expected=404)
            new_run = destination.json(
                "/api/v1/analyses", body=body, key="after-restore", expected=202
            )
            if new_run["analysis_id"] in (analysis["analysis_id"], second["analysis_id"]):
                raise AssertionError("The restored deployment replayed an earlier analysis.")
            finished = destination.completed(new_run)
            prediction = destination.json(text(finished, "result_url"))
            for field in ("architecture_id", "model", "snapshot_sha256", "sequences"):
                if prediction[field] != saved[field]:
                    raise AssertionError(f"Restored-model analysis changed {field}.")

        success = {
            "status": "passed",
            "project": project,
            "restore_project": target,
            "storage": backend,
            "architecture": architecture,
            "model": expected_model,
            "checks": [
                "migration-before-services",
                "direct-TLS-CA-SNI",
                "exact-worker-heartbeats",
                "lightweight-API-image",
                "coordinated-postgres-and-object-backup",
                "saved-model-analysis",
                "offline-report",
                "other-owner-denied",
                "persistent-recreation-idempotency",
                "fresh-worker-saved-model-reload",
                "restore-into-empty-deployment",
                "restored-token-report-and-model",
                "project-teardown",
            ],
            "report_sha256": report_digest,
        }
    print(json.dumps(success), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose-file", type=Path, action="append", dest="files")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--s3-credentials-file", type=Path)
    parser.add_argument("--s3-ca-file", type=Path)
    parser.add_argument(
        "--release-tcn",
        type=Path,
        nargs="?",
        const=Path("release"),
        metavar="ASSET_DIR",
        help="Use the TCN pinned by the asset pack's SHA256SUMS (default: release/).",
    )
    args = parser.parse_args()
    run(
        args.files,
        build=not args.no_build,
        s3_credentials=args.s3_credentials_file,
        s3_ca=args.s3_ca_file,
        release_tcn=args.release_tcn,
    )
