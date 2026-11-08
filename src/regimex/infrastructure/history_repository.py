"""Scoped experiment history and optimistic naming without rewriting observations."""

import unicodedata
from typing import TYPE_CHECKING, cast
from uuid import UUID

from regimex.history_contracts import ExperimentName, ExperimentNameUpdate, ExperimentSummary
from regimex.service_contracts import (
    AnalysisHistoryFilter,
    Conflict,
    HistoryState,
    Principal,
    SnapshotRecord,
)
from regimex.storage import json_bytes

if TYPE_CHECKING:
    from regimex.infrastructure.database import DatabaseRepository


def analysis_scope(principal: Principal) -> tuple[str, list[object]]:
    query = (
        "FROM regimex_analyses a JOIN regimex_operations o "
        "ON a.operation_id=o.operation_id JOIN regimex_snapshots s "
        "ON a.snapshot_id=s.snapshot_id WHERE o.owner_id=%s"
    )
    parameters: list[object] = [principal.owner_id]
    if principal.role == "integration":
        query += " AND s.source_id=%s"
        parameters.append(principal.source_namespace)
        if principal.run_scope is not None:
            query += " AND s.run_id=%s"
            parameters.append(principal.run_scope)
    return query, parameters


def analysis_filters(filters: AnalysisHistoryFilter) -> tuple[str, list[object]]:
    clauses = ["true"]
    parameters: list[object] = []
    if filters.state == "active":
        clauses.append("o.state IN ('queued','running','cancel_requested')")
    elif filters.state is not None:
        clauses.append("o.state=%s")
        parameters.append(filters.state)
    if search := filters.q.strip():
        # Casefold names independently of the database locale, which may only
        # handle ASCII under C. Literal matching also keeps % and _ ordinary text.
        clauses.append(
            "strpos(concat_ws(' ',s.display_name_search,translate(concat_ws(' ',"
            "s.source_id,s.run_id,"
            "s.source_snapshot_id,s.document->>'instrument_ids',a.model_version_id,"
            "a.analysis_id::text),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')),"
            "%s) > 0"
        )
        parameters.append(unicodedata.normalize("NFC", search).casefold())
    return " AND ".join(clauses), parameters


class HistoryRepository:
    def __init__(self, database: DatabaseRepository) -> None:
        self.database = database

    def list_experiments(
        self,
        principal: Principal,
        *,
        limit: int = 100,
        offset: int = 0,
        q: str = "",
        state: HistoryState | None = None,
    ) -> tuple[ExperimentSummary, ...]:
        self.database._page(limit, offset)
        predicate, filter_parameters = analysis_filters(AnalysisHistoryFilter(q=q, state=state))
        scope = (
            "FROM regimex_snapshots s LEFT JOIN regimex_analyses a "
            "ON a.snapshot_id=s.snapshot_id LEFT JOIN regimex_operations o "
            "ON a.operation_id=o.operation_id WHERE s.owner_id=%s"
        )
        scope_parameters: list[object] = [principal.owner_id]
        if principal.role == "integration":
            scope += " AND s.source_id=%s"
            scope_parameters.append(principal.source_namespace)
            if principal.run_scope is not None:
                scope += " AND s.run_id=%s"
                scope_parameters.append(principal.run_scope)
        # Count all authorized runs before filtering; choose one matching run per
        # snapshot before pagination so a large experiment cannot split a page.
        query = (
            "WITH candidates AS (SELECT a.analysis_id,s.snapshot_id,o.document AS operation,"
            "a.result IS NOT NULL AS result_available,"
            "EXISTS(SELECT 1 FROM regimex_analysis_reports r WHERE r.analysis_id=a.analysis_id) "
            "AS report_available,s.document AS snapshot,s.display_name,s.name_revision,"
            "COALESCE(o.document->>'created_at',s.document->>'created_at')::timestamptz "
            "AS created_at,o.state='succeeded' "
            f"AS successful,({predicate}) AS matches {scope}), "
            "counted AS (SELECT *,count(analysis_id) OVER snapshot_group AS analysis_count,"
            "count(*) FILTER (WHERE successful) OVER snapshot_group AS successful_count,"
            "count(analysis_id) FILTER (WHERE matches) OVER snapshot_group AS matching_count "
            "FROM candidates WINDOW snapshot_group AS (PARTITION BY snapshot_id)), "
            "matching AS (SELECT *,row_number() OVER (PARTITION BY snapshot_id "
            "ORDER BY created_at DESC,analysis_id DESC) AS rank FROM counted WHERE matches) "
            "SELECT analysis_id,snapshot_id,operation,result_available,report_available,"
            "snapshot,display_name,name_revision,analysis_count,successful_count,matching_count "
            "FROM matching WHERE rank=1 "
            "ORDER BY created_at DESC,analysis_id DESC NULLS LAST,snapshot_id DESC "
            "LIMIT %s OFFSET %s"
        )
        with self.database.connection() as connection:
            rows = connection.execute(
                query, [*filter_parameters, *scope_parameters, limit, offset]
            ).fetchall()
            summaries = []
            for row in rows:
                snapshot = SnapshotRecord.model_validate_json(json_bytes(row[5]))
                summaries.append(
                    ExperimentSummary(
                        snapshot_id=snapshot.snapshot_id,
                        source_id=snapshot.source_id,
                        run_id=snapshot.run_id,
                        instrument_ids=snapshot.instrument_ids,
                        step_count=snapshot.step_count,
                        created_at=snapshot.created_at,
                        display_name=cast(str | None, row[6]),
                        name_revision=cast(int, row[7]),
                        analysis_count=cast(int, row[8]),
                        successful_analysis_count=cast(int, row[9]),
                        matching_analysis_count=cast(int, row[10]),
                        latest_analysis=(
                            None if row[0] is None else self.database._summary_row(row[:6])
                        ),
                    )
                )
            return tuple(summaries)

    def rename(
        self, principal: Principal, snapshot_id: UUID, update: ExperimentNameUpdate
    ) -> ExperimentName:
        update = ExperimentNameUpdate.model_validate_json(update.model_dump_json())
        with self.database.connection() as connection:
            snapshot = self.database._snapshot(connection, snapshot_id)
            self.database._authorize_snapshot(principal, snapshot)
            row = connection.execute(
                "UPDATE regimex_snapshots SET display_name=%s,display_name_search=%s,"
                "name_revision=name_revision+1 "
                "WHERE snapshot_id=%s AND name_revision=%s RETURNING display_name,name_revision",
                (
                    update.display_name,
                    None if update.display_name is None else update.display_name.casefold(),
                    snapshot_id,
                    update.expected_revision,
                ),
            ).fetchone()
            if row is None:
                raise Conflict("The experiment name changed. Refresh it before saving again.")
            return ExperimentName(
                snapshot_id=snapshot_id,
                display_name=cast(str | None, row[0]),
                revision=cast(int, row[1]),
            )
