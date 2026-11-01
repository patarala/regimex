"""Storage inventory metadata without SDK or application-workflow imports."""

from typing import Annotated

from pydantic import AwareDatetime, Field

from regimex.contracts import StrictModel


class StoredVersion(StrictModel):
    """Inventory metadata cannot establish content SHA256 or authorize deletion."""

    key: str
    version_id: Annotated[str, Field(min_length=1, max_length=256)]
    size_bytes: Annotated[int, Field(ge=0)]
    last_modified: AwareDatetime
    is_delete_marker: bool = False


class IncompleteUpload(StrictModel):
    key: str
    upload_id: Annotated[str, Field(min_length=1, max_length=1024)]
    initiated_at: AwareDatetime
