"""Stage-0 compatibility proof on a disposable real PostgreSQL, never a mock."""

import asyncio
import os
from uuid import uuid4

import pytest


@pytest.mark.parametrize("commit", [True, False])
def test_operation_and_queue_submission_share_one_transaction(commit: bool) -> None:
    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    psycopg = pytest.importorskip("psycopg", reason="install the queue-test dependency group")
    procrastinate = pytest.importorskip(
        "procrastinate", reason="install the queue-test dependency group"
    )
    schema = "regimex_queue_test_" + uuid4().hex
    operation_id = str(uuid4())
    # Only our generated identifier enters options. Credentials never enter task arguments.
    options = f"-csearch_path={schema}"
    with psycopg.connect(dsn, autocommit=True) as administration:
        administration.execute(
            psycopg.sql.SQL("CREATE SCHEMA {}").format(psycopg.sql.Identifier(schema))
        )
        try:
            app = procrastinate.App(
                connector=procrastinate.SyncPsycopgConnector(
                    conninfo=dsn, kwargs={"options": options}
                )
            )
            with app.open():
                app.schema_manager.apply_schema()
                with psycopg.connect(dsn, options=options, autocommit=True) as observer:
                    observer.execute(
                        "CREATE TABLE operations (id uuid PRIMARY KEY, state text NOT NULL)"
                    )
                    with psycopg.connect(dsn, options=options) as transaction:
                        transaction.execute(
                            "INSERT INTO operations (id, state) VALUES (%s, %s)",
                            (operation_id, "queued"),
                        )
                        job_id = app.configure_task(
                            name="regimex.analysis", queue="analysis", connection=transaction
                        ).defer(operation_id=operation_id)
                        assert observer.execute("SELECT count(*) FROM operations").fetchone() == (
                            0,
                        )
                        assert observer.execute(
                            "SELECT count(*) FROM procrastinate_jobs"
                        ).fetchone() == (0,)
                        if commit:
                            transaction.commit()
                        else:
                            transaction.rollback()
                    assert observer.execute("SELECT count(*) FROM operations").fetchone() == (
                        int(commit),
                    )
                    assert observer.execute(
                        "SELECT count(*) FROM procrastinate_jobs"
                    ).fetchone() == (int(commit),)
                    if commit:
                        row = observer.execute(
                            "SELECT id, queue_name, args FROM procrastinate_jobs WHERE id = %s",
                            (job_id,),
                        ).fetchone()
                        assert row == (job_id, "analysis", {"operation_id": operation_id})
        finally:
            administration.execute(
                psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(psycopg.sql.Identifier(schema))
            )


async def _exercise_async_submission(*, commit: bool, run_worker: bool) -> None:
    dsn = os.environ.get("REGIMEX_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("set REGIMEX_TEST_DATABASE_DSN to a disposable PostgreSQL database")
    psycopg = pytest.importorskip("psycopg", reason="install the queue-test dependency group")
    procrastinate = pytest.importorskip(
        "procrastinate", reason="install the queue-test dependency group"
    )
    schema = "regimex_async_queue_test_" + uuid4().hex
    options = f"-csearch_path={schema}"
    operation_id = str(uuid4())
    executed_ids: list[str] = []
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as administration:
        await administration.execute(
            psycopg.sql.SQL("CREATE SCHEMA {}").format(psycopg.sql.Identifier(schema))
        )
        try:
            app = procrastinate.App(
                connector=procrastinate.PsycopgConnector(
                    conninfo=dsn, kwargs={"options": options}, min_size=1, max_size=4
                )
            )

            async def record_execution(operation_id: str) -> None:
                # This is a compatibility probe, not an implemented analysis task.
                async with await psycopg.AsyncConnection.connect(dsn, options=options) as result:
                    cursor = await result.execute(
                        "UPDATE operations SET state = 'executed', executions = executions + 1 "
                        "WHERE id = %s AND state = 'queued' RETURNING id",
                        (operation_id,),
                    )
                    row = await cursor.fetchone()
                    assert row is not None and str(row[0]) == operation_id
                executed_ids.append(operation_id)

            app.task(name="regimex.compatibility_probe", queue="analysis")(record_execution)

            async def drain_worker() -> None:
                await asyncio.wait_for(
                    app.run_worker_async(
                        queues=["analysis"],
                        concurrency=1,
                        wait=False,
                        listen_notify=False,
                        install_signal_handlers=False,
                        shutdown_graceful_timeout=1.0,
                    ),
                    timeout=15.0,
                )

            async with app.open_async():
                await app.schema_manager.apply_schema_async()
                async with await psycopg.AsyncConnection.connect(
                    dsn, options=options, autocommit=True
                ) as observer:
                    await observer.execute(
                        "CREATE TABLE operations (id uuid PRIMARY KEY, state text NOT NULL, "
                        "executions integer NOT NULL DEFAULT 0)"
                    )
                    async with await psycopg.AsyncConnection.connect(
                        dsn, options=options
                    ) as transaction:
                        await transaction.execute(
                            "INSERT INTO operations (id, state) VALUES (%s, %s)",
                            (operation_id, "queued"),
                        )
                        job_id = await app.configure_task(
                            name="regimex.compatibility_probe",
                            queue="analysis",
                            connection=transaction,
                        ).defer_async(operation_id=operation_id)
                        operations = await observer.execute("SELECT count(*) FROM operations")
                        jobs = await observer.execute("SELECT count(*) FROM procrastinate_jobs")
                        assert await operations.fetchone() == (0,)
                        assert await jobs.fetchone() == (0,)
                        if run_worker:
                            await drain_worker()
                            assert executed_ids == []
                        if commit:
                            await transaction.commit()
                        else:
                            await transaction.rollback()
                    operations = await observer.execute("SELECT count(*) FROM operations")
                    jobs = await observer.execute("SELECT count(*) FROM procrastinate_jobs")
                    assert await operations.fetchone() == (int(commit),)
                    assert await jobs.fetchone() == (int(commit),)
                    if commit:
                        job = await observer.execute(
                            "SELECT id, queue_name, args, status FROM procrastinate_jobs "
                            "WHERE id = %s",
                            (job_id,),
                        )
                        assert await job.fetchone() == (
                            job_id,
                            "analysis",
                            {"operation_id": operation_id},
                            "todo",
                        )
                    if run_worker:
                        # A second rolled-back submission must never reach the callback.
                        discarded_id = str(uuid4())
                        async with await psycopg.AsyncConnection.connect(
                            dsn, options=options
                        ) as discarded:
                            await discarded.execute(
                                "INSERT INTO operations (id, state) VALUES (%s, %s)",
                                (discarded_id, "queued"),
                            )
                            await app.configure_task(
                                name="regimex.compatibility_probe", connection=discarded
                            ).defer_async(operation_id=discarded_id)
                            await discarded.rollback()
                        await drain_worker()
                        assert executed_ids == [operation_id]
                        operation = await observer.execute(
                            "SELECT id::text, state, executions FROM operations"
                        )
                        assert await operation.fetchall() == [(operation_id, "executed", 1)]
                        job = await observer.execute(
                            "SELECT id, status FROM procrastinate_jobs ORDER BY id"
                        )
                        assert await job.fetchall() == [(job_id, "succeeded")]
        finally:
            await administration.execute(
                psycopg.sql.SQL("DROP SCHEMA {} CASCADE").format(psycopg.sql.Identifier(schema))
            )


@pytest.mark.parametrize("commit", [True, False])
def test_async_operation_and_queue_submission_share_one_transaction(commit: bool) -> None:
    asyncio.run(_exercise_async_submission(commit=commit, run_worker=False))


def test_async_worker_executes_only_committed_operation_id() -> None:
    asyncio.run(_exercise_async_submission(commit=True, run_worker=True))
