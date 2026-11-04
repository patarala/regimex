"""Bounded installation operations executed by the staged service interpreter."""

import argparse
import json
import os
import secrets
import sys
import tempfile
from pathlib import Path


def private(path: Path) -> str:
    info = path.lstat()
    if (
        not path.is_file()
        or path.is_symlink()
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise ValueError("Installation credential must be a private owned regular file.")
    return path.read_text(encoding="utf-8").strip()


def create_private(path: Path, content: str) -> None:
    """Publish only complete new credentials; interruption cannot leave an empty final file."""
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix="." + path.name, delete=False
    ) as output:
        temporary = Path(output.name)
        try:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
            os.link(temporary, path)
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)


def provision(root: Path, port: int) -> None:
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    password = private(root / "credentials/postgres-bootstrap.password")
    with psycopg.connect(
        host=str(root / "postgres/socket"),
        port=port,
        user="regimex_bootstrap",
        password=password,
        dbname="postgres",
        autocommit=True,
        connect_timeout=5,
    ) as connection:
        credential = root / "credentials/database.dsn"
        role_exists = (
            connection.execute("SELECT 1 FROM pg_roles WHERE rolname='regimex'").fetchone()
            is not None
        )
        if not credential.exists():
            if role_exists:
                raise ValueError("An unrecorded application role requires manual recovery.")
            app_password = secrets.token_urlsafe(40)
            create_private(
                credential,
                make_conninfo(
                    host=str(root / "postgres/socket"),
                    port=port,
                    dbname="regimex",
                    user="regimex",
                    password=app_password,
                )
                + "\n",
            )
        from psycopg.conninfo import conninfo_to_dict

        parameters = conninfo_to_dict(private(credential))
        if any(
            parameters.get(key) != expected
            for key, expected in {
                "host": str(root / "postgres/socket"),
                "port": str(port),
                "dbname": "regimex",
                "user": "regimex",
            }.items()
        ) or not parameters.get("password"):
            raise ValueError("Bootstrap application credentials belong to another installation.")
        if not role_exists:
            connection.execute(
                sql.SQL(
                    "CREATE ROLE regimex LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD {}"
                ).format(sql.Literal(parameters["password"]))
            )
        if not connection.execute("SELECT 1 FROM pg_database WHERE datname='regimex'").fetchone():
            connection.execute("CREATE DATABASE regimex OWNER regimex")
        # A retry authenticates the original credentials; it never resets passwords.
        with psycopg.connect(private(credential), connect_timeout=5) as existing:
            existing.execute("SELECT 1")


def inspect(root: Path) -> dict[str, object]:
    import psycopg
    from psycopg.conninfo import conninfo_to_dict

    dsn = private(root / "credentials/database.dsn")
    parameters = conninfo_to_dict(dsn)
    if parameters.get("host") != str(root / "postgres/socket"):
        raise ValueError("Application connection does not use this installation's private socket.")
    with psycopg.connect(dsn, connect_timeout=5) as connection:
        row = connection.execute(
            "SELECT rolsuper,rolcreatedb,rolcreaterole FROM pg_roles WHERE rolname=current_user"
        ).fetchone()
        if row != (False, False, False):
            raise ValueError("Application database role must have no administrative privileges.")
        pending = connection.execute(
            "SELECT count(*) FROM regimex_operations "
            "WHERE state IN ('queued','running','cancel_requested')"
        ).fetchone()
        doing = connection.execute(
            "SELECT count(*) FROM procrastinate_jobs WHERE status='doing'"
        ).fetchone()
        assert pending is not None and doing is not None
        return {
            "pending": pending[0],
            "doing": doing[0],
            "postgres_port": int(parameters.get("port") or "5432"),
        }


def models(root: Path, source: Path) -> None:
    import hashlib

    from regimex.model_registry import InferenceRuntime
    from regimex.service_bootstrap import create_service
    from regimex.service_contracts import NotFound
    from regimex.service_settings import ServiceSettings

    service = create_service(ServiceSettings.from_env())
    principal = service.authenticate(private(root / "credentials/admin.token"))
    for architecture in ("baseline", "tcn"):
        bundle = source / "release/models" / architecture
        content = (bundle / "manifest.json").read_bytes()
        identity = json.loads(content)["model_version_id"]
        checksum = hashlib.sha256(content).hexdigest()
        try:
            registered = service.repository.get_model(identity)
        except NotFound:
            active = service.repository.get_active_model()
            service.register_bundle(
                principal,
                bundle,
                manifest_sha256=checksum,
                inference_runtime=InferenceRuntime(threads=2),
                activate=True,
                expected_previous=None if active is None else active.model_version_id,
            )
        else:
            if registered.bundle_sha256 != checksum:
                raise ValueError("An existing model version has different immutable content.")
        with service.repository.connection() as connection:
            published = (
                connection.execute(
                    "SELECT 1 FROM regimex_publication_audit WHERE model_version_id=%s LIMIT 1",
                    (identity,),
                ).fetchone()
                is not None
            )
        active = service.repository.get_active_model()
        if not published or (
            architecture == "tcn" and (active is None or active.model_version_id != identity)
        ):
            service.activate_model(
                principal,
                identity,
                expected_previous=None if active is None else active.model_version_id,
            )


def tokens(root: Path) -> None:
    from regimex.infrastructure.database import DatabaseRepository
    from regimex.service_contracts import Role
    from regimex.service_settings import ServiceSettings

    settings = ServiceSettings.from_env()
    repository = DatabaseRepository(
        settings.database_dsn.get_secret_value(), schema=settings.database_schema
    )
    credentials: tuple[tuple[Role, str, str], ...] = (
        ("admin", "admin", "admin.token"),
        ("user", "researcher", "user.token"),
    )
    for role, owner, name in credentials:
        path = root / "credentials" / name
        if path.exists():
            if repository.authenticate(private(path)).role != role:
                raise ValueError("Existing access credential has a different role.")
        else:
            token = repository.create_token(owner, role)
            create_private(path, token + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("provision", "inspect", "tokens", "models"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    arguments = parser.parse_args()
    try:
        result: object = None
        if arguments.action == "provision":
            provision(arguments.root, arguments.port)
        elif arguments.action == "inspect":
            result = inspect(arguments.root)
        elif arguments.action == "tokens":
            tokens(arguments.root)
        else:
            models(arguments.root, arguments.source)
        print(json.dumps(result))
        return 0
    except Exception as error:
        # Driver failures can contain complete DSNs; expose the class only.
        print(
            f"Installation database/model check failed ({type(error).__name__}).", file=sys.stderr
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
