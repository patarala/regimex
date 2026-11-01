"""Lightweight settings and maintenance metadata for the tested S3 deployment."""

import ipaddress
from pathlib import Path
from typing import Annotated, Self
from urllib.parse import urlsplit

from pydantic import ConfigDict, Field, model_validator

from regimex.contracts import StrictModel
from regimex.infrastructure.object_inventory import IncompleteUpload, StoredVersion
from regimex.infrastructure.object_store import _absolute_parts, _key_parts


class S3Config(StrictModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    endpoint_url: Annotated[str, Field(min_length=1, max_length=2048)]
    region: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")]
    bucket: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")]
    prefix: Annotated[str, Field(min_length=1, max_length=512)]
    lock_root: Path
    allow_loopback_http: bool = False
    ca_bundle: Path | None = None
    credential_profile: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    connect_timeout_seconds: Annotated[int, Field(ge=1, le=30)] = 5
    read_timeout_seconds: Annotated[int, Field(ge=1, le=120)] = 10
    max_attempts: Annotated[int, Field(ge=1, le=5)] = 3
    max_inventory_entries: Annotated[int, Field(ge=1, le=1000000)] = 100000

    @property
    def endpoint_origin(self) -> str:
        url = urlsplit(self.endpoint_url)
        host = url.hostname or ""
        if ":" in host:
            host = "[" + host + "]"
        port = url.port
        default_port = 443 if url.scheme == "https" else 80
        suffix = f":{port}" if port is not None and port != default_port else ""
        return url.scheme + "://" + host.lower() + suffix

    @model_validator(mode="after")
    def checked_deployment(self) -> Self:
        try:
            url = urlsplit(self.endpoint_url)
            port = url.port
        except ValueError:
            raise ValueError("S3 endpoint is not a valid origin") from None
        if (
            not url.hostname
            or url.username is not None
            or url.password is not None
            or url.path not in ("", "/")
            or url.query
            or url.fragment
            or any(character.isspace() for character in self.endpoint_url)
            or port == 0
        ):
            raise ValueError("S3 endpoint must be a plain origin without credentials")
        if url.scheme == "http":
            try:
                loopback = ipaddress.ip_address(url.hostname).is_loopback
            except ValueError:
                loopback = False
            if not self.allow_loopback_http or not loopback:
                raise ValueError("HTTP S3 requires explicitly enabled literal loopback")
        elif url.scheme != "https":
            raise ValueError("remote S3 endpoints require HTTPS")
        _key_parts(self.prefix)
        if len(self.prefix.encode("utf-8")) > 1022:
            raise ValueError("S3 prefix must leave room for a relative object key")
        _absolute_parts(self.lock_root)
        if self.lock_root.absolute() == Path("/"):
            raise ValueError("S3 coordination requires a dedicated local directory")
        if self.ca_bundle is not None:
            _absolute_parts(self.ca_bundle)
            if url.scheme != "https":
                raise ValueError("an S3 CA bundle requires HTTPS")
        if self.credential_profile is not None and not self.credential_profile.strip():
            raise ValueError("S3 credential profile must not be blank")
        return self


S3ObjectVersion = StoredVersion
S3MultipartUpload = IncompleteUpload
