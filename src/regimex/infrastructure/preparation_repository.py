"""Schedule the next stage in the same transaction that publishes the previous result."""

from typing import cast
from uuid import UUID, uuid4

from psycopg.types.json import Jsonb
from pydantic import TypeAdapter

from regimex.admin_contracts import (
    AdminRunRecord,
    GenerationSpec,
    ReferenceGenerationSpec,
    admin_request_hash,
)
from regimex.infrastructure.admin_repository import AdminRepository, Connection
from regimex.infrastructure.database import _read_operation
from regimex.infrastructure.intake_repository import IntakeRequest
from regimex.operations import LaunchSource, Operation
from regimex.preparation import (
    STAGES,
    PreparationPage,
    PreparationPlan,
    PreparationRequest,
    PreparationView,
    Stage,
    selection_id,
)
from regimex.service_contracts import CapacityExceeded, Conflict, NotFound, Principal
from regimex.storage import json_bytes

_OUTPUTS = TypeAdapter(dict[Stage, str])
_SELECT = """SELECT p.preparation_id,p.created_at,p.plan,p.stage,p.outputs,o.document,
    b.benchmark_id,c.benchmark_id IS NOT NULL
    FROM regimex_training_preparations p JOIN regimex_operations o USING (operation_id)
    LEFT JOIN regimex_benchmark_bindings b
      ON b.benchmark_id=p.selection_id AND b.owner_id=p.owner_id
    LEFT JOIN regimex_benchmark_choices c ON c.benchmark_id=b.benchmark_id
    WHERE p.owner_id=%s"""


class PreparationRepository:
    def __init__(self, admin: AdminRepository) -> None:
        self.admin = admin
        self.database = admin.database

    @staticmethod
    def _view(row: tuple[object, ...]) -> PreparationView:
        preparation_id, created, plan, stage, outputs, document, benchmark, frozen = row
        operation = _read_operation(document)
        saved = _OUTPUTS.validate_python(outputs)
        return PreparationView.model_validate(
            {
                "preparation_id": preparation_id,
                "created_at": created,
                "request": PreparationPlan.model_validate_json(json_bytes(plan)).request,
                "stage": stage,
                "operation_id": operation.operation_id,
                "state": operation.state,
                "failure": operation.failure.message if operation.failure else None,
                "outputs": saved,
                "benchmark_id": benchmark,
                "selection_closed": frozen,
                "workspace_url": f"/training/new?preparation={preparation_id}",
            }
        )

    def _read(self, connection: Connection, owner: str, preparation_id: UUID) -> PreparationView:
        row = connection.execute(
            _SELECT + " AND p.preparation_id=%s", (owner, preparation_id)
        ).fetchone()
        if row is None:
            raise NotFound()
        return self._view(row)

    def get(self, principal: Principal, preparation_id: UUID) -> PreparationView:
        self.database._admin(principal)
        with self.database.connection() as connection:
            return self._read(connection, principal.owner_id, preparation_id)

    def list(self, principal: Principal, limit: int, offset: int) -> PreparationPage:
        self.database._admin(principal)
        self.database._page(limit, offset)
        with self.database.connection() as connection:
            rows = connection.execute(
                _SELECT + " ORDER BY p.created_at DESC,p.preparation_id DESC LIMIT %s OFFSET %s",
                (principal.owner_id, limit, offset),
            ).fetchall()
            return PreparationPage(
                items=tuple(self._view(row) for row in rows), limit=limit, offset=offset
            )

    def _enqueue(
        self,
        connection: Connection,
        owner: str,
        spec: GenerationSpec | ReferenceGenerationSpec,
        launch_source: LaunchSource | None,
        key: str | None = None,
    ) -> UUID:
        request = IntakeRequest(
            idempotency_key=key or str(uuid4()), request_sha256=admin_request_hash(spec, None)
        )
        self.admin._validate_spec(connection, owner, spec)
        return self.admin._insert_run(connection, owner, spec, request, launch_source).run_id

    def _browser_job(
        self,
        connection: Connection,
        owner: str,
        preparation_id: UUID,
        stage: Stage,
        expected: GenerationSpec | ReferenceGenerationSpec,
    ) -> AdminRunRecord | None:
        # Previous browser versions submitted stages independently with these receipt keys.
        row = connection.execute(
            "SELECT operation_id FROM regimex_operations WHERE owner_id=%s AND kind=%s "
            "AND idempotency_key ~ %s ORDER BY length(idempotency_key) DESC, "
            "idempotency_key DESC LIMIT 1 FOR UPDATE",
            (owner, expected.kind, f"^{preparation_id}:{stage}:[0-9]{{1,10}}$"),
        ).fetchone()
        if row is None:
            return None
        run = self.admin._run(connection, cast(UUID, row[0]))
        actual: GenerationSpec | ReferenceGenerationSpec
        if isinstance(run.spec, GenerationSpec):
            actual = run.spec.model_copy(update={"dataset_id": expected.dataset_id})
        elif isinstance(run.spec, ReferenceGenerationSpec) and isinstance(
            expected, ReferenceGenerationSpec
        ):
            actual = run.spec.model_copy(update={"reference_set_id": expected.reference_set_id})
        else:
            raise Conflict("The saved preparation refers to a different type of job.")
        if actual != expected:
            raise Conflict("The saved job settings differ from this preparation.")
        return run

    def _begin(
        self,
        connection: Connection,
        owner: str,
        preparation_id: UUID,
        plan: PreparationPlan,
        launch_source: LaunchSource | None,
        max_pending: int,
    ) -> None:
        outputs: dict[Stage, str] = {}
        for stage in STAGES:
            spec = plan.next_job(stage, outputs)
            previous = None
            key = None
            if plan.request.browser_seeds is not None:
                previous = self._browser_job(connection, owner, preparation_id, stage, spec)
                key = f"{preparation_id}:{stage}:0"
            if previous is None:
                self.admin._check_capacity(connection, owner, max_pending)
                run_id = self._enqueue(connection, owner, spec, launch_source, key)
                break
            run_id = previous.run_id
            if previous.operation.state != "succeeded":
                break
            committed = previous.spec
            assert isinstance(committed, GenerationSpec | ReferenceGenerationSpec)
            outputs[stage] = (
                committed.dataset_id
                if isinstance(committed, GenerationSpec)
                else committed.reference_set_id
            )
        selection = (
            selection_id(owner, outputs["development"], outputs["final"], outputs["test"])
            if "test" in outputs
            else None
        )
        connection.execute(
            "INSERT INTO regimex_training_preparations "
            "(owner_id,preparation_id,plan,stage,operation_id,outputs,selection_id) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (
                owner,
                preparation_id,
                Jsonb(plan.model_dump(mode="json")),
                stage,
                run_id,
                Jsonb(outputs),
                selection,
            ),
        )

    def start(
        self,
        principal: Principal,
        preparation_id: UUID,
        request: PreparationRequest,
        *,
        launch_source: LaunchSource | None,
        max_experiments: int,
        max_pending: int,
    ) -> PreparationView:
        self.database._admin(principal)
        owner = principal.owner_id
        with self.database.connection() as connection:
            self.database._lock(connection, f"owner:{owner}")
            previous = connection.execute(
                _SELECT + " AND p.preparation_id=%s", (owner, preparation_id)
            ).fetchone()
            if previous is not None:
                replay = self._view(previous)
                if replay.request != request:
                    raise Conflict("This preparation ID already belongs to a different request.")
                return replay
            if max(request.development, request.final) > max_experiments:
                raise CapacityExceeded(
                    f"A dataset can contain at most {max_experiments} generated experiments."
                )
            self._begin(
                connection,
                owner,
                preparation_id,
                PreparationPlan.create(request),
                launch_source,
                max_pending,
            )
            return self._read(connection, owner, preparation_id)

    def advance(
        self,
        connection: Connection,
        completed: Operation,
        spec: GenerationSpec | ReferenceGenerationSpec,
    ) -> None:
        row = connection.execute(
            "SELECT plan,stage,outputs,preparation_id FROM regimex_training_preparations "
            "WHERE operation_id=%s FOR UPDATE",
            (completed.operation_id,),
        ).fetchone()
        if row is None:
            return
        plan = PreparationPlan.model_validate_json(json_bytes(row[0]))
        stage = cast(Stage, row[1])
        outputs = _OUTPUTS.validate_python(row[2])
        outputs[stage] = (
            spec.dataset_id if isinstance(spec, GenerationSpec) else spec.reference_set_id
        )
        next_id = completed.operation_id
        selection = None
        if stage == "test":
            selection = selection_id(
                completed.owner_id, outputs["development"], outputs["final"], outputs["test"]
            )
        else:
            # The new job takes the completed job's quota slot in this transaction.
            stage = STAGES[STAGES.index(stage) + 1]
            next_id = self._enqueue(
                connection,
                completed.owner_id,
                plan.next_job(stage, outputs),
                completed.launch_source,
                f"{row[3]}:{stage}:0" if plan.request.browser_seeds else None,
            )
        connection.execute(
            "UPDATE regimex_training_preparations SET operation_id=%s,stage=%s,outputs=%s,"
            "selection_id=%s WHERE operation_id=%s",
            (next_id, stage, Jsonb(outputs), selection, completed.operation_id),
        )

    def recover(self, max_pending: int) -> int:
        # A worker from the previous release can finish a stage during a rolling update.
        with self.database.connection() as connection:
            missed = connection.execute(
                "SELECT p.owner_id,p.operation_id FROM regimex_training_preparations p "
                "JOIN regimex_operations o USING (operation_id) "
                "WHERE NOT (p.outputs ? p.stage) AND o.state='succeeded' "
                "AND (p.stage='test' OR (SELECT count(*) FROM regimex_operations busy "
                "WHERE busy.owner_id=p.owner_id AND busy.state IN "
                "('queued','running','cancel_requested')) < %s) "
                "ORDER BY p.created_at,p.preparation_id LIMIT 100",
                (max_pending,),
            ).fetchall()
        recovered = 0
        for owner, operation_id in missed:
            with self.database.connection() as connection:
                self.database._lock(connection, f"owner:{owner}")
                stage = connection.execute(
                    "SELECT stage FROM regimex_training_preparations "
                    "WHERE operation_id=%s AND NOT (outputs ? stage) FOR UPDATE",
                    (operation_id,),
                ).fetchone()
                if stage is None:
                    continue
                if stage[0] != "test":
                    # Between releases the completed job's quota slot may have been claimed.
                    try:
                        self.admin._check_capacity(connection, cast(str, owner), max_pending)
                    except CapacityExceeded:
                        continue
                run = self.admin._run(connection, cast(UUID, operation_id))
                if not isinstance(run.spec, GenerationSpec | ReferenceGenerationSpec):
                    raise Conflict("A preparation stage must generate data or references.")
                self.advance(connection, run.operation, run.spec)
                recovered += 1
        return recovered

    def retry(
        self, principal: Principal, preparation_id: UUID, failed_operation: UUID, max_pending: int
    ) -> PreparationView:
        self.database._admin(principal)
        owner = principal.owner_id
        with self.database.connection() as connection:
            self.database._lock(connection, f"owner:{owner}")
            row = connection.execute(
                _SELECT + " AND p.preparation_id=%s FOR UPDATE OF p", (owner, preparation_id)
            ).fetchone()
            if row is None:
                raise NotFound()
            current = self._view(row)
            if current.operation_id != failed_operation:
                return current
            if current.state not in {"failed", "cancelled"}:
                raise Conflict("Only a failed or cancelled preparation stage can be retried.")
            self.admin._check_capacity(connection, owner, max_pending)
            plan = PreparationPlan.model_validate_json(json_bytes(row[2]))
            previous = _read_operation(row[5])
            key = None
            if plan.request.browser_seeds:
                attempt = int(previous.idempotency_key.rsplit(":", 1)[1]) + 1
                key = f"{preparation_id}:{current.stage}:{attempt}"
            run_id = self._enqueue(
                connection,
                owner,
                plan.next_job(current.stage, current.outputs),
                previous.launch_source,
                key,
            )
            connection.execute(
                "UPDATE regimex_training_preparations SET operation_id=%s WHERE operation_id=%s",
                (run_id, failed_operation),
            )
            return self._read(connection, owner, preparation_id)
