"""Narrow HS256 profile for the deliberate simulator gateway fixture.

This is not a replacement for the unavailable simulator's authentication layer.
The real adapter must use that layer's verified principal, not decode untrusted
claims and pass them on as an owner.
"""

import base64
import hashlib
import hmac
import json
import re
import time
from dataclasses import dataclass, field
from typing import cast

from pydantic import JsonValue, SecretBytes

IDENTITY_PATTERN = r"[A-Za-z0-9._:@-]{1,128}"
_IDENTITY = re.compile(IDENTITY_PATTERN)
_BASE64URL = re.compile(r"[A-Za-z0-9_-]+")


class InvalidToken(ValueError):
    def __init__(self) -> None:
        super().__init__("Invalid simulator authentication.")


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON member.")
        result[key] = value
    return result


def strict_json(data: bytes) -> JsonValue:
    def reject_constant(value: str) -> None:
        raise ValueError("Nonfinite JSON value.")

    return cast(
        JsonValue,
        json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=reject_constant,
        ),
    )


def json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    if not _BASE64URL.fullmatch(value):
        raise InvalidToken
    try:
        result = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except ValueError:
        raise InvalidToken from None
    if _encode(result) != value:
        raise InvalidToken
    return result


def _verified_claims(token: str, key: SecretBytes) -> dict[str, JsonValue]:
    """Cryptographic codec only; SimulatorJWT.verify also checks identity/time claims."""
    if len(token) > 16 * 1024 or token.count(".") != 2:
        raise InvalidToken
    encoded_header, encoded_claims, signature = token.split(".")
    header = strict_json(_decode(encoded_header))
    # Reject key-selection/URL/critical extensions as well as other algorithms.
    if header != {"alg": "HS256", "typ": "JWT"}:
        raise InvalidToken
    expected = hmac.digest(
        key.get_secret_value(),
        f"{encoded_header}.{encoded_claims}".encode("ascii"),
        "sha256",
    )
    if not hmac.compare_digest(expected, _decode(signature)):
        raise InvalidToken
    claims = strict_json(_decode(encoded_claims))
    if not isinstance(claims, dict):
        raise InvalidToken
    return claims


@dataclass(frozen=True)
class SimulatorJWT:
    issuer: str
    audience: str
    key: SecretBytes = field(repr=False)

    def __post_init__(self) -> None:
        if not self.issuer or not self.audience or len(self.key.get_secret_value()) < 32:
            raise ValueError("The fixture requires issuer, audience and a 256-bit HS256 key.")

    def verify(self, token: str, *, now: int | None = None) -> str:
        try:
            return self._verify(token, int(time.time()) if now is None else now)
        except ValueError, UnicodeError, TypeError, OverflowError, RecursionError:
            raise InvalidToken from None

    def _verify(self, token: str, now: int) -> str:
        claims = _verified_claims(token, self.key)
        subject = claims.get("sub")
        expiration, issued_at = claims.get("exp"), claims.get("iat")
        not_before = claims.get("nbf", issued_at)
        if (
            not isinstance(subject, str)
            or not _IDENTITY.fullmatch(subject)
            or claims.get("iss") != self.issuer
            or claims.get("aud") != self.audience
            or type(expiration) is not int
            or type(issued_at) is not int
            or type(not_before) is not int
        ):
            raise InvalidToken
        assert isinstance(expiration, int) and isinstance(issued_at, int)
        assert isinstance(not_before, int)
        if not (issued_at <= now < expiration and not_before <= now):
            raise InvalidToken
        if not 0 < expiration - issued_at <= 3600 or not_before >= expiration:
            raise InvalidToken
        return subject

    def issue(self, subject: str, *, now: int | None = None, lifetime: int = 900) -> str:
        """Only the deliberate fixture CLI/tests issue tokens with their own key."""
        if not _IDENTITY.fullmatch(subject) or not 1 <= lifetime <= 3600:
            raise ValueError("Invalid fixture subject or token lifetime.")
        issued = int(time.time()) if now is None else now
        header = _encode(json_bytes({"alg": "HS256", "typ": "JWT"}))
        payload = _encode(
            json_bytes(
                {
                    "sub": subject,
                    "iss": self.issuer,
                    "aud": self.audience,
                    "iat": issued,
                    "exp": issued + lifetime,
                }
            )
        )
        signature = hmac.digest(
            self.key.get_secret_value(), f"{header}.{payload}".encode("ascii"), hashlib.sha256
        )
        return f"{header}.{payload}.{_encode(signature)}"
