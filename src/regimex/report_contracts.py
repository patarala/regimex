"""Immutable offline analysis report metadata, with no visualization imports."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, Field

from regimex.contracts import Identifier, StrictModel
from regimex.exports import SourceIdentity
from regimex.infrastructure.object_store import ObjectInfo
from regimex.operations import ResolvedModel, Sha256

MAX_REPORT_BYTES = 16 * 1024 * 1024
REPORT_RENDERER_VERSION: Literal["analysis-html/3.0"] = "analysis-html/3.0"


class AnalysisReportRecord(StrictModel):
    analysis_id: UUID
    owner_id: Identifier
    snapshot_id: UUID
    snapshot_sha256: Sha256
    source: SourceIdentity
    model: ResolvedModel
    result_sha256: Sha256
    renderer_version: Literal["analysis-html/3.0"] = REPORT_RENDERER_VERSION
    plotly_version: Annotated[str, Field(min_length=1, max_length=128)]
    object: ObjectInfo
    created_at: AwareDatetime
