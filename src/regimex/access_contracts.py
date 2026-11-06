"""Public records for server integrations and their revocable credentials."""

import unicodedata
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, SecretStr, field_validator, model_validator

from regimex.contracts import Identifier, StrictModel

IntegrationScope = Literal["read", "upload", "analyze"]
INTEGRATION_SCOPES: tuple[IntegrationScope, ...] = ("read", "upload", "analyze")
AccessAction = Literal["integration.created", "integration.updated", "key.created", "key.revoked"]
AccessName = Annotated[str, Field(min_length=1, max_length=120)]


def _name(value: str) -> str:
    result = unicodedata.normalize("NFC", value).strip()
    if not 1 <= len(result) <= 120 or any(not character.isprintable() for character in result):
        raise ValueError("Names must contain visible characters without control characters.")
    return result


class IntegrationCreate(StrictModel):
    owner_id: Identifier
    source_namespace: Identifier
    display_name: AccessName

    _display_name = field_validator("display_name")(_name)

    @field_validator("owner_id", "source_namespace")
    @classmethod
    def visible_identity(cls, value: str) -> str:
        if value != value.strip() or any(not character.isprintable() for character in value):
            raise ValueError("Integration identities cannot contain controls or outer whitespace.")
        return value


class IntegrationUpdate(StrictModel):
    expected_revision: Annotated[int, Field(ge=0)]
    display_name: AccessName | None = None
    enabled: bool | None = None

    @field_validator("display_name")
    @classmethod
    def visible_name(cls, value: str | None) -> str | None:
        return None if value is None else _name(value)

    @model_validator(mode="after")
    def has_change(self) -> Self:
        if self.display_name is None and self.enabled is None:
            raise ValueError("Supply a name or enabled state to update an integration.")
        return self


class IntegrationView(StrictModel):
    integration_id: UUID
    owner_id: Identifier
    source_namespace: Identifier
    display_name: AccessName
    enabled: bool
    revision: Annotated[int, Field(ge=0)]
    created_at: AwareDatetime
    updated_at: AwareDatetime


class IntegrationKeyCreate(StrictModel):
    name: AccessName
    lifetime_days: Annotated[int, Field(ge=1, le=365)] = 30
    scopes: Annotated[tuple[IntegrationScope, ...], Field(min_length=1, max_length=3)] = (
        INTEGRATION_SCOPES
    )

    _name = field_validator("name")(_name)

    @field_validator("scopes")
    @classmethod
    def distinct_scopes(cls, value: tuple[IntegrationScope, ...]) -> tuple[IntegrationScope, ...]:
        if len(set(value)) != len(value):
            raise ValueError("Integration scopes must not be repeated.")
        return tuple(scope for scope in INTEGRATION_SCOPES if scope in value)


class IntegrationKeyView(StrictModel):
    token_id: UUID
    integration_id: UUID
    name: AccessName
    scopes: tuple[IntegrationScope, ...]
    created_at: AwareDatetime
    expires_at: AwareDatetime | None
    last_used_at: AwareDatetime | None
    revoked_at: AwareDatetime | None


class IssuedIntegrationKey(StrictModel):
    key: IntegrationKeyView
    token: SecretStr = Field(repr=False)


class AccessEventView(StrictModel):
    event_id: UUID
    integration_id: UUID
    created_at: AwareDatetime
    actor_owner_id: Identifier | None
    actor_token_id: UUID | None
    action: AccessAction
    target_token_id: UUID | None
    enabled: bool
    display_name: AccessName
