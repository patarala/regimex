"""Exercise request admission on one worker event loop, independently of transport keep-alive."""

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from regimex.api import AnalysisAPIService


def test_overload_precedes_body_reads_and_keeps_static_assets_available() -> None:
    httpx = pytest.importorskip("httpx2")
    pytest.importorskip("fastapi")
    pytest.importorskip("psycopg")
    from regimex.api import create_app

    class Readiness:
        def is_ready(self, *, require_model: bool = False) -> bool:
            return True

    async def exercise() -> None:
        app = create_app(cast("AnalysisAPIService", Readiness()), http_concurrency_limit=2)
        finish = asyncio.Event()

        async def upload(started: asyncio.Event) -> AsyncIterator[bytes]:
            started.set()
            yield b"{"
            await finish.wait()
            yield b"}"

        third_body_read = asyncio.Event()

        async def third_upload() -> AsyncIterator[bytes]:
            third_body_read.set()
            yield b"{}"

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            started = [asyncio.Event(), asyncio.Event()]
            uploads = [
                asyncio.create_task(client.post("/api/v1/snapshots", content=upload(event)))
                for event in started
            ]
            try:
                await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 2)
                refused = await client.post("/api/v1/snapshots", content=third_upload())
                assert refused.status_code == 503
                assert not third_body_read.is_set()
                assert refused.json()["error"]["code"] == "capacity_limit"
                assert refused.headers["retry-after"] == "1"
                assert refused.headers["connection"] == "close"
                assert refused.headers["cache-control"] == "no-store"
                for path in ("/", "/training", "/api/v1/jobs", "/health/models", "/health/ready"):
                    assert (await client.get(path)).status_code == 503
                for path in ("/health/live", "/assets/ui.js"):
                    assert (await client.get(path)).status_code == 200
                assert (await client.post("/assets/ui.js")).status_code == 503
            finally:
                finish.set()
                await asyncio.gather(*uploads)
            assert (await client.get("/health/ready")).status_code == 200

    asyncio.run(exercise())


def test_stream_holds_slot_until_completion_or_cancellation_and_errors_release_it() -> None:
    httpx = pytest.importorskip("httpx2")
    pytest.importorskip("fastapi")
    pytest.importorskip("psycopg")
    from starlette.types import Receive, Scope, Send

    from regimex.api import RequestAdmission

    async def exercise() -> None:
        streaming, finish = asyncio.Event(), asyncio.Event()

        async def application(scope: Scope, receive: Receive, send: Send) -> None:
            if scope["path"] == "/fail":
                raise RuntimeError("handler failed")
            await send({"type": "http.response.start", "status": 200, "headers": []})
            if scope["path"] == "/stream":
                await send({"type": "http.response.body", "body": b"part", "more_body": True})
                streaming.set()
                await finish.wait()
            await send({"type": "http.response.body", "body": b"end"})

        admitted = RequestAdmission(application, limit=1)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=admitted), base_url="http://test"
        ) as client:
            for cancelled in (False, True):
                streaming.clear()
                finish.clear()
                pending = asyncio.create_task(client.get("/stream"))
                try:
                    await asyncio.wait_for(streaming.wait(), 2)
                    assert (await client.get("/other")).status_code == 503
                    assert (await client.head("/assets/ui.js")).status_code == 200
                    if cancelled:
                        pending.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await pending
                    else:
                        finish.set()
                        assert (await pending).content == b"partend"
                finally:
                    finish.set()
                    await asyncio.gather(pending, return_exceptions=True)
                assert (await client.get("/other")).status_code == 200
            with pytest.raises(RuntimeError, match="handler failed"):
                await client.get("/fail")
            assert (await client.get("/other")).status_code == 200

    asyncio.run(exercise())
