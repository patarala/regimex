"""Editable display names and grouped experiment history."""

import unicodedata
from typing import Annotated
from uuid import UUID

from pydantic import AfterValidator, AwareDatetime, Field, TypeAdapter

from regimex.contracts import MAX_STEPS, Identifier, StrictModel
from regimex.service_contracts import AnalysisSummary


def _readable_name(value: str) -> str:
    if any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value):
        raise ValueError("Names cannot contain control characters.")
    if not (name := value.strip()):
        raise ValueError("Use null to restore the default name.")
    return unicodedata.normalize("NFC", name)


# NFC can expand combining characters, so check the length after normalization.
DisplayName = Annotated[str, AfterValidator(_readable_name), Field(min_length=1, max_length=120)]
_DISPLAY_NAME = TypeAdapter(DisplayName)


def validate_display_name(value: str | None) -> str | None:
    return None if value is None else _DISPLAY_NAME.validate_python(value, strict=True)


class DisplayNameUpdate(StrictModel):
    display_name: DisplayName | None
    expected_revision: Annotated[int, Field(ge=0, le=2**63 - 2)]


class ExperimentNameUpdate(DisplayNameUpdate):
    """An edit to a snapshot's name in experiment history."""


class ExperimentName(StrictModel):
    snapshot_id: UUID
    display_name: DisplayName | None
    revision: Annotated[int, Field(ge=0)]


class ExperimentMetadata(StrictModel):
    snapshot_id: UUID
    source_id: Identifier
    run_id: Identifier
    instrument_ids: tuple[Identifier, ...]
    step_count: Annotated[int, Field(ge=1, le=MAX_STEPS)] | None
    created_at: AwareDatetime
    display_name: DisplayName | None
    name_revision: Annotated[int, Field(ge=0)]
    analysis_count: Annotated[int, Field(ge=0)]
    successful_analysis_count: Annotated[int, Field(ge=0)]
    matching_analysis_count: Annotated[int, Field(ge=0)]


class ExperimentSummary(ExperimentMetadata):
    latest_analysis: AnalysisSummary | None
