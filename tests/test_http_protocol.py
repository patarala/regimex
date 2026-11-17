"""Real HTTP connections cannot extend intake deadlines by trickling bytes."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from starlette.types import Receive, Scope, Send


@asynccontextmanager
async def server(*, timeout: float = 0.2) -> AsyncIterator[int]:
    uvicorn = pytest.importorskip("uvicorn")
    from uvicorn.server import ServerState

    from regimex.http_protocol import bounded_http_protocol

    async def application(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["path"] in {"/early-response", "/stream-response"}:
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-length", b"2")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b"ok" if scope["path"] == "/early-response" else b"o",
                    "more_body": scope["path"] == "/stream-response",
                }
            )
            if scope["path"] == "/early-response":
                return
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if not message.get("more_body", False):
                break
        if scope["path"] == "/slow-response":
            await asyncio.sleep(timeout * 2)
        if scope["path"] == "/stream-response":
            await send({"type": "http.response.body", "body": b"k"})
            return
        await send(
            {"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"2")]}
        )
        await send({"type": "http.response.body", "body": b"ok"})

    config = uvicorn.Config(
        application,
        lifespan="off",
        ws="none",
        log_config=None,
        limit_concurrency=8,
        timeout_keep_alive=10,
    )
    config.load()
    state = ServerState()
    protocol = bounded_http_protocol(header_timeout_seconds=timeout, body_timeout_seconds=timeout)
    listener = await asyncio.get_running_loop().create_server(
        lambda: protocol(config, state, {}), "127.0.0.1", 0
    )
    try:
        yield listener.sockets[0].getsockname()[1]
    finally:
        listener.close()
        await listener.wait_closed()
        for connection in tuple(state.connections):
            connection.transport.close()
        if state.tasks:
            await asyncio.wait_for(asyncio.gather(*tuple(state.tasks)), timeout=2)


async def response(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    header = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=2)
    length = next(
        (
            int(line.split(b":", 1)[1])
            for line in header.split(b"\r\n")
            if line.lower().startswith(b"content-length:")
        ),
        0,
    )
    body = await asyncio.wait_for(reader.readexactly(length), timeout=2)
    return int(header.split()[1]), body


@pytest.mark.parametrize(
    "partial",
    [
        b"",
        b"GET / HTTP/1.1\r\nHost: local",
        b"POST / HTTP/1.1\r\nHost: local\r\nContent-Length: 10\r\n\r\nx",
        b"POST / HTTP/1.1\r\nHost: local\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nx",
    ],
)
def test_incomplete_requests_expire_and_release_all_connection_slots(partial: bytes) -> None:
    async def exercise() -> None:
        async with server() as port:
            connections = [await asyncio.open_connection("127.0.0.1", port) for _ in range(7)]
            try:
                for _, writer in connections:
                    writer.write(partial)
                    await writer.drain()
                replies = await asyncio.gather(*(response(reader) for reader, _ in connections))
                assert all(status == 408 for status, _ in replies)
                endings = await asyncio.gather(
                    *(asyncio.wait_for(reader.read(), timeout=2) for reader, _ in connections)
                )
                assert endings == [b""] * 7
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                try:
                    writer.write(b"GET / HTTP/1.1\r\nHost: local\r\n\r\n")
                    assert await response(reader) == (200, b"ok")
                finally:
                    writer.close()
                    await writer.wait_closed()
            finally:
                for _, writer in connections:
                    writer.close()
                    await writer.wait_closed()

    asyncio.run(exercise())


@pytest.mark.parametrize("body", [False, True])
def test_dripped_bytes_do_not_restart_deadline(body: bool) -> None:
    async def exercise() -> None:
        async with server(timeout=0.2) as port:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)

            async def drip() -> None:
                while True:
                    await asyncio.sleep(0.05)
                    writer.write(b"x")
                    await writer.drain()

            sender = asyncio.create_task(drip())
            try:
                writer.write(
                    b"POST / HTTP/1.1\r\nHost: local\r\nContent-Length: 10000\r\n\r\n"
                    if body
                    else b"GET / HTTP/1.1\r\nHost: "
                )
                assert (await response(reader))[0] == 408
            finally:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
                writer.close()
                await writer.wait_closed()

    asyncio.run(exercise())


def test_complete_requests_can_have_long_responses_and_pipelined_successors() -> None:
    async def exercise() -> None:
        async with server() as port:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            try:
                writer.write(
                    b"POST /slow-response HTTP/1.1\r\nHost: local\r\n"
                    b"Content-Length: 2\r\n\r\nok"
                    b"GET / HTTP/1.1\r\nHost: local\r\n\r\n"
                )
                assert await response(reader) == (200, b"ok")
                assert await response(reader) == (200, b"ok")
                writer.write(b"GET / HTTP/1.1\r\nHost: unfinished")
                assert (await response(reader))[0] == 408
            finally:
                writer.close()
                await writer.wait_closed()

    asyncio.run(exercise())


@pytest.mark.parametrize("path", ["/early-response", "/stream-response"])
def test_unread_body_still_expires_without_injecting_a_second_response(path: str) -> None:
    async def exercise() -> None:
        async with server() as port:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            try:
                writer.write(
                    f"POST {path} HTTP/1.1\r\nHost: local\r\nContent-Length: 10\r\n\r\nx".encode()
                )
                payload = await asyncio.wait_for(reader.read(), timeout=2)
                assert payload.startswith(b"HTTP/1.1 200")
                assert payload.count(b"HTTP/1.1") == 1
                assert b"408" not in payload
            finally:
                writer.close()
                await writer.wait_closed()

    asyncio.run(exercise())
