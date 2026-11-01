"""Session policy and trusted, run-scoped integration identities."""

import ipaddress
from dataclasses import dataclass
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import AwareDatetime, ConfigDict, Field, SecretStr, model_validator

from regimex.contracts import Identifier, StrictModel
from regimex.service_contracts import Principal

ExternalIdentifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9._:@-]{1,128}$")]


def canonical_origin(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ValueError("The public origin must be a valid HTTP(S) origin.") from None
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or any(ord(char) <= 32 or ord(char) == 127 for char in value)
        or port == 0
    ):
        raise ValueError("The public origin must not contain credentials, paths or controls.")
    host = parsed.hostname.lower()
    if ":" in host:
        host = "[" + host + "]"
    default = 443 if parsed.scheme == "https" else 80
    suffix = f":{port}" if port is not None and port != default else ""
    return parsed.scheme + "://" + host + suffix


class AuthPolicy(StrictModel):
    model_config = ConfigDict(hide_input_in_errors=True)
    public_origin: Annotated[str, Field(min_length=1, max_length=2048)]
    allow_loopback_http: bool = False
    session_ttl_seconds: Annotated[int, Field(ge=60, le=8 * 3600)] = 1800
    max_sessions_per_token: Annotated[int, Field(ge=1, le=32)] = 8
    max_subjects_per_integration: Annotated[int, Field(ge=1, le=1_000_000)] = 100_000

    @model_validator(mode="after")
    def transport(self) -> Self:
        origin = canonical_origin(self.public_origin)
        parsed = urlsplit(origin)
        if parsed.scheme == "http":
            try:
                loopback = ipaddress.ip_address(parsed.hostname or "").is_loopback
            except ValueError:
                loopback = False
            if not self.allow_loopback_http or not loopback:
                raise ValueError("HTTP sessions require explicitly enabled literal loopback.")
        elif self.allow_loopback_http:
            raise ValueError("The loopback HTTP allowance must affect the configured origin.")
        return self

    @property
    def origin(self) -> str:
        return canonical_origin(self.public_origin)

    @property
    def secure(self) -> bool:
        return self.origin.startswith("https://")

    @property
    def cookie_name(self) -> str:
        return "__Host-regimex_session" if self.secure else "regimex_session"


class BrowserPrincipal(StrictModel):
    owner_id: Identifier
    role: Literal["user", "admin"]


class SessionRecord(StrictModel):
    session_id: UUID
    principal: Principal
    created_at: AwareDatetime
    expires_at: AwareDatetime


class SessionView(StrictModel):
    principal: BrowserPrincipal
    csrf_token: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$", repr=False)]
    expires_at: AwareDatetime


@dataclass(frozen=True, slots=True, repr=False)
class IssuedSession:
    cookie_value: SecretStr
    record: SessionRecord


@dataclass(frozen=True, slots=True, repr=False)
class BrowserSession:
    cookie_value: SecretStr
    view: SessionView


@dataclass(frozen=True, slots=True, repr=False)
class RequestAuth:
    method: str
    scheme: str
    authority: str
    headers: tuple[tuple[bytes, bytes], ...] = ()

    def values(self, name: str) -> tuple[str, ...]:
        key = name.lower().encode("ascii")
        return tuple(
            value.decode("latin-1") for header, value in self.headers if header.lower() == key
        )

    def one(self, name: str) -> str | None:
        from regimex.service_contracts import ServiceError

        values = self.values(name)
        if len(values) > 1:
            raise ServiceError(
                "invalid_request", "Duplicate authentication headers are unsupported.", 400
            )
        return values[0] if values else None
