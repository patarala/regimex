"""Database authority for browser sessions and asserted integration subjects."""

import hashlib
import re
import secrets
from datetime import datetime, timedelta
from typing import cast
from uuid import UUID, uuid4

import psycopg
from pydantic import SecretStr, TypeAdapter

from regimex.auth_contracts import ExternalIdentifier, IssuedSession, SessionRecord
from regimex.data_catalog import content_hash
from regimex.infrastructure.database import DatabaseRepository
from regimex.service_contracts import CapacityExceeded, Forbidden, Principal, Unauthorized

Connection = psycopg.Connection[tuple[object, ...]]
_TOKEN = re.compile(r"rgx_[A-Za-z0-9_-]{43}\Z")
SESSION = re.compile(r"rgxs_[A-Za-z0-9_-]{43}\Z")
_EXTERNAL = TypeAdapter(ExternalIdentifier)


def session_hash(raw_session: str) -> str:
    if SESSION.fullmatch(raw_session) is None:
        raise Unauthorized()
    return hashlib.sha256(raw_session.encode("ascii")).hexdigest()


class AuthRepository:
    def __init__(self, database: DatabaseRepository) -> None:
        self.database = database

    @staticmethod
    def _principal(row: tuple[object, ...]) -> Principal:
        return Principal.model_validate(
            dict(zip(("token_id", "owner_id", "role", "source_namespace"), row, strict=True))
        )

    @staticmethod
    def _parent(connection: Connection, raw_token: str, *, lock: bool) -> Principal:
        if _TOKEN.fullmatch(raw_token) is None:
            raise Unauthorized()
        query = (
            "SELECT token_id,owner_id,role,source_namespace FROM regimex_tokens "
            "WHERE token_sha256=%s AND revoked_at IS NULL"
        )
        row = connection.execute(
            query + (" FOR UPDATE" if lock else ""),
            (hashlib.sha256(raw_token.encode("ascii")).hexdigest(),),
        ).fetchone()
        if row is None:
            raise Unauthorized()
        return AuthRepository._principal(row)

    def issue_session(
        self, raw_token: str, *, ttl_seconds: int = 1800, maximum_sessions: int = 8
    ) -> IssuedSession:
        if type(ttl_seconds) is not int or not 60 <= ttl_seconds <= 8 * 3600:
            raise ValueError("Session lifetime must be between one minute and eight hours.")
        if type(maximum_sessions) is not int or not 1 <= maximum_sessions <= 32:
            raise ValueError("The active session limit must be between one and 32.")
        raw_session = "rgxs_" + secrets.token_urlsafe(32)
        with self.database.connection() as connection:
            principal = self._parent(connection, raw_token, lock=True)
            if principal.role == "integration":
                raise Forbidden("Integration credentials cannot create browser sessions.")
            now_row = connection.execute("SELECT clock_timestamp()").fetchone()
            assert now_row is not None
            now = cast(datetime, now_row[0])
            connection.execute(
                "DELETE FROM regimex_sessions WHERE token_id=%s AND expires_at<=%s",
                (principal.token_id, now),
            )
            count = connection.execute(
                "SELECT count(*) FROM regimex_sessions WHERE token_id=%s", (principal.token_id,)
            ).fetchone()
            assert count is not None
            if cast(int, count[0]) >= maximum_sessions:
                raise CapacityExceeded(
                    "This credential has reached its active browser-session limit."
                )
            record = SessionRecord(
                session_id=uuid4(),
                principal=principal,
                created_at=now,
                expires_at=now + timedelta(seconds=ttl_seconds),
            )
            connection.execute(
                "INSERT INTO regimex_sessions "
                "(session_id,token_id,session_sha256,created_at,expires_at) "
                "VALUES (%s,%s,%s,%s,%s)",
                (
                    record.session_id,
                    principal.token_id,
                    session_hash(raw_session),
                    now,
                    record.expires_at,
                ),
            )
        return IssuedSession(SecretStr(raw_session), record)

    def authenticate_session(self, raw_session: str) -> SessionRecord:
        checksum = session_hash(raw_session)
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT s.session_id,s.created_at,s.expires_at,"
                "t.token_id,t.owner_id,t.role,t.source_namespace "
                "FROM regimex_sessions s JOIN regimex_tokens t ON s.token_id=t.token_id "
                "WHERE s.session_sha256=%s AND s.expires_at>clock_timestamp() "
                "AND t.revoked_at IS NULL",
                (checksum,),
            ).fetchone()
        if row is None:
            raise Unauthorized()
        principal = self._principal(row[3:])
        if principal.role == "integration":
            raise Unauthorized()
        return SessionRecord(
            session_id=cast(UUID, row[0]),
            principal=principal,
            created_at=cast(datetime, row[1]),
            expires_at=cast(datetime, row[2]),
        )

    def revoke_session(self, raw_session: str) -> None:
        with self.database.connection() as connection:
            connection.execute(
                "DELETE FROM regimex_sessions WHERE session_sha256=%s", (session_hash(raw_session),)
            )

    def bind_subject(
        self, raw_token: str, subject: str, run_id: str, *, maximum_subjects: int = 100_000
    ) -> Principal:
        subject = _EXTERNAL.validate_python(subject, strict=True)
        run_id = _EXTERNAL.validate_python(run_id, strict=True)
        if type(maximum_subjects) is not int or not 1 <= maximum_subjects <= 1_000_000:
            raise ValueError("The integration subject bound must be between one and one million.")
        subject_sha256 = content_hash({"version": "external-subject/1.0", "subject": subject})
        with self.database.connection() as connection:
            principal = self._parent(connection, raw_token, lock=False)
            if principal.role != "integration" or principal.source_namespace is None:
                raise Forbidden(
                    "Only a registered integration backend can assert an external subject."
                )
            self.database._lock(
                connection,
                "integration:" + content_hash([principal.owner_id, principal.source_namespace]),
            )
            row = connection.execute(
                "SELECT effective_owner_id FROM regimex_integration_subjects "
                "WHERE integration_owner_id=%s AND source_namespace=%s AND subject_sha256=%s",
                (principal.owner_id, principal.source_namespace, subject_sha256),
            ).fetchone()
            if row is None:
                count = connection.execute(
                    "SELECT count(*) FROM regimex_integration_subjects "
                    "WHERE integration_owner_id=%s AND source_namespace=%s",
                    (principal.owner_id, principal.source_namespace),
                ).fetchone()
                assert count is not None
                if cast(int, count[0]) >= maximum_subjects:
                    raise CapacityExceeded(
                        "The integration has reached its registered subject limit."
                    )
                owner_id = "external_" + uuid4().hex
                connection.execute(
                    "INSERT INTO regimex_integration_subjects "
                    "(integration_owner_id,source_namespace,subject_sha256,effective_owner_id) "
                    "VALUES (%s,%s,%s,%s)",
                    (principal.owner_id, principal.source_namespace, subject_sha256, owner_id),
                )
            else:
                owner_id = cast(str, row[0])
            return Principal(
                owner_id=owner_id,
                role="integration",
                token_id=principal.token_id,
                source_namespace=principal.source_namespace,
                run_scope=run_id,
            )
