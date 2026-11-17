"""Immutable receipts for synchronous administrative resource creation."""

from typing import Literal, cast

import psycopg
from pydantic import ValidationError

from regimex.admin_contracts import RequestDigest, RequestFingerprint
from regimex.contracts import Identifier, StrictModel
from regimex.service_contracts import Conflict, ServiceError
from regimex.storage import json_bytes

IntakeKind = Literal["dataset_import", "reference_import", "final_reference_import"]
Connection = psycopg.Connection[tuple[object, ...]]


class IntakeRequest(StrictModel):
    idempotency_key: Identifier
    request_sha256: RequestDigest

    @property
    def digest(self) -> str:
        if isinstance(self.request_sha256, RequestFingerprint):
            return self.request_sha256.sha256
        return self.request_sha256

    def require_match(self, stored: object) -> None:
        if stored == self.digest:
            return
        if (
            isinstance(self.request_sha256, RequestFingerprint)
            and stored in self.request_sha256.previous_hashes
        ):
            return
        raise Conflict("The idempotency key was already used for a different request.")

    @classmethod
    def optional(
        cls, idempotency_key: str | None, request_sha256: RequestDigest | None
    ) -> IntakeRequest | None:
        if idempotency_key is None and request_sha256 is None:
            return None
        try:
            return cls.model_validate(
                {"idempotency_key": idempotency_key, "request_sha256": request_sha256},
                strict=True,
            )
        except ValidationError as exc:
            raise ServiceError(
                "invalid_data",
                "Idempotency requires a bounded key and canonical request SHA-256.",
                422,
            ) from exc


class IntakeReceipts:
    @staticmethod
    def find(
        connection: Connection,
        owner_id: str,
        kind: IntakeKind,
        request: IntakeRequest,
        *,
        lock: bool = False,
    ) -> str | None:
        if lock:
            # Serialize a key even before its first row exists. The lock and
            # receipt share the resource transaction, so rollback admits a retry.
            scope = json_bytes([owner_id, kind, request.idempotency_key]).decode()
            connection.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended("
                "current_schema() || ':regimex-intake:' || %s, 0))",
                (scope,),
            )
        row = connection.execute(
            "SELECT request_sha256,COALESCE(dataset_id,reference_set_id) "
            "FROM regimex_intake_keys WHERE owner_id=%s AND kind=%s AND idempotency_key=%s",
            (owner_id, kind, request.idempotency_key),
        ).fetchone()
        if row is None:
            return None
        request.require_match(row[0])
        return cast(str, row[1])

    @staticmethod
    def add(
        connection: Connection,
        owner_id: str,
        kind: IntakeKind,
        request: IntakeRequest,
        resource_id: str,
    ) -> None:
        connection.execute(
            "INSERT INTO regimex_intake_keys "
            "(owner_id,kind,idempotency_key,request_sha256,dataset_id,reference_set_id) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            (
                owner_id,
                kind,
                request.idempotency_key,
                request.digest,
                resource_id if kind == "dataset_import" else None,
                None if kind == "dataset_import" else resource_id,
            ),
        )
