"""Object-payload usage shared by the service and its browser interface."""

from typing import Annotated, Self

from pydantic import Field, model_validator

from regimex.contracts import StrictModel
from regimex.service_contracts import CapacityExceeded


class StorageUsage(StrictModel):
    used_bytes: Annotated[int, Field(ge=0)]
    reserved_bytes: Annotated[int, Field(ge=0)]
    limit_bytes: Annotated[int, Field(ge=1)]
    available_bytes: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def remaining_capacity(self) -> Self:
        if self.available_bytes != max(0, self.limit_bytes - self.used_bytes - self.reserved_bytes):
            raise ValueError("Available storage must account for committed and reserved bytes.")
        return self


class StorageQuotaExceeded(CapacityExceeded):
    """A safe public explanation for an owner or connection storage refusal."""

    def __init__(self, *, integration: bool = False) -> None:
        subject = "connection" if integration else "account"
        super().__init__(
            f"This {subject} has reached its storage limit. "
            "Ask an administrator to review storage usage or increase the limit."
        )
