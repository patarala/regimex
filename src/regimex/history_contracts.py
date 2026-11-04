"""Mutable experiment names and grouped history, separate from immutable exports."""

import unicodedata
from typing import Annotated
from uuid import UUID

from pydantic import AwareDatetime, Field, field_validator

from regimex.contracts import MAX_STEPS, Identifier, StrictModel
from regimex.service_contracts import AnalysisSummary

DisplayName = Annotated[str, Field(min_length=1, max_length=120)]


class ExperimentNameUpdate(StrictModel):
    display_name: DisplayName | None
    expected_revision: Annotated[int, Field(ge=0, le=2**63 - 2)]

    @field_validator("display_name", mode="before")
    @classmethod
    def readable_name(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        if any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value):
            raise ValueError("Experiment names cannot contain control characters.")
        if not (name := value.strip()):
            raise ValueError("Use null to restore the default experiment name.")
        # Normalize before Field's length check: NFC can expand some combining
        # characters, so validating the original length alone is insufficient.
        return unicodedata.normalize("NFC", name)


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
    analysis_count: Annotated[int, Field(ge=1)]
    successful_analysis_count: Annotated[int, Field(ge=0)]
    matching_analysis_count: Annotated[int, Field(ge=1)]


class ExperimentSummary(ExperimentMetadata):
    latest_analysis: AnalysisSummary
