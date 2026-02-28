"""Browser interop tests for stream edge cases: FIN, reset, stop, read modes."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

import web_transport

if TYPE_CHECKING:
    from .conftest import RunJS, ServerFactory

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_browser_writer_close_signals_eof(
    start_server: ServerFactory, run_js: RunJS
) -> None:
    """Browser writer.close() causes server recv.read() to return data then EOF."""
    async with start_server() as (server, port, hash_b64):
        received: bytes = b""
        eof_bytes: bytes = b"sentinel"

        async def server_side() -> None:
            nonlocal received, eof_bytes
            request = await server.accept()
            assert request is not None
            session = await request.accept()
            async with session:
                send, recv = await session.accept_bi()
                async with send:
                    received = await recv.read()
                    eof_bytes = await recv.read()
                await session.wait_closed()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            await run_js(
                port,
                hash_b64,
                """
                const stream = await transport.createBidirectionalStream();
                await writeAllString(stream.writable, "data");
                return true;
            """,
            )

    assert received == b"data"
    assert eof_bytes == b""


async def test_server_finish_signals_eof_to_browser(
    start_server: ServerFactory, run_js: RunJS
) -> None:
    """Server send.finish() causes browser reader to return done: true."""
    async with start_server() as (server, port, hash_b64):

        async def server_side() -> None:
            request = await server.accept()
            assert request is not None
            session = await request.accept()
            async with session:
                send, recv = await session.accept_bi()
                await send.write(b"payload")
                await send.finish()
                await recv.read()  # wait for browser to close
                await session.wait_closed()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            result = await run_js(
                port,
                hash_b64,
                """
                const stream = await transport.createBidirectionalStream();
                const data = await readAllString(stream.readable);
                // Close writable to let server side complete
                const writer = stream.writable.getWriter();
                await writer.close();
                return data;
            """,
            )

    assert result == "payload"


async def test_browser_abort_with_code(
    start_server: ServerFactory, run_js: RunJS
) -> None:
    """Browser writer.abort(42) → server recv sees StreamClosedByPeer(kind='reset', code=42)."""
    async with start_server() as (server, port, hash_b64):
        error: BaseException | None = None

        async def server_side() -> None:
            nonlocal error
            request = await server.accept()
            assert request is not None
            session = await request.accept()
            async with session:
                send, recv = await session.accept_bi()
                async with send:
                    try:
                        await recv.read()
                    except web_transport.StreamClosedByPeer as e:
                        error = e

        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            await run_js(
                port,
                hash_b64,
                """
                const stream = await transport.createBidirectionalStream();
                const writer = stream.writable.getWriter();
                await new Promise(r => setTimeout(r, 200));
                let err = new WebTransportError({ message: "abort", streamErrorCode: 42 });
                await writer.abort(err);
                await transport.closed;
                return true;
            """,
            )

    assert isinstance(error, web_transport.StreamClosedByPeer)
    assert error.kind == "reset"
    assert error.code == 42


async def test_server_reset_causes_browser_read_error(
    start_server: ServerFactory, run_js: RunJS
) -> None:
    """Server send.reset(7) causes browser reader.read() to reject."""
    async with start_server() as (server, port, hash_b64):

        async def server_side() -> None:
            request = await server.accept()
            assert request is not None
            session = await request.accept()
            async with session:
                send, recv = await session.accept_bi()
                send.reset(7)
                await recv.read()  # wait for browser close
                await session.wait_closed()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            result: Any = await run_js(
                port,
                hash_b64,
                """
                const stream = await transport.createBidirectionalStream();
                const reader = stream.readable.getReader();
                try {
                    await reader.read();
                    return { errored: false };
                } catch (e) {
                    return { errored: true, message: e.toString() };
                } finally {
                    const writer = stream.writable.getWriter();
                    await writer.close();
                }
            """,
            )

    assert isinstance(result, dict)
    assert result["errored"] is True


async def test_browser_cancel_recv_with_code(
    start_server: ServerFactory, run_js: RunJS
) -> None:
    """Browser reader.cancel(42) → server send.write() raises StreamClosedByPeer(kind='stop', code=42)."""
    async with start_server() as (server, port, hash_b64):
        error: BaseException | None = None

        async def server_side() -> None:
            nonlocal error
            request = await server.accept()
            assert request is not None
            session = await request.accept()
            async with session:
                send, recv = await session.accept_bi()
                # Wait a moment for the browser to cancel
                try:
                    # Write enough data to trigger the stop
                    for _ in range(10):
                        await send.write(b"x" * 65536)
                except (
                    web_transport.StreamClosedByPeer,
                    web_transport.SessionClosedByPeer,
                ) as e:
                    error = e
                try:
                    await recv.read()
                except web_transport.SessionClosed:
                    pass
                await session.wait_closed()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            await run_js(
                port,
                hash_b64,
                """
                const stream = await transport.createBidirectionalStream();
                const reader = stream.readable.getReader();
                let err = new WebTransportError({ message: "cancel", streamErrorCode: 42 });
                await reader.cancel(err);
                // Close writable so server recv completes
                const writer = stream.writable.getWriter();
                await writer.close();
                return true;
            """,
            )

    assert error is not None
    if isinstance(error, web_transport.StreamClosedByPeer):
        assert error.kind == "stop"
        assert error.code == 42


async def test_server_stop_causes_browser_write_error(
    start_server: ServerFactory, run_js: RunJS
) -> None:
    """Server recv.stop(7) causes browser writer.write() to reject."""
    async with start_server() as (server, port, hash_b64):

        async def server_side() -> None:
            request = await server.accept()
            assert request is not None
            session = await request.accept()
            async with session:
                send, recv = await session.accept_bi()
                async with send:
                    recv.stop(7)
                await session.wait_closed()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            result: Any = await run_js(
                port,
                hash_b64,
                """
                const stream = await transport.createBidirectionalStream();
                const writer = stream.writable.getWriter();
                // Give server time to send STOP_SENDING
                await new Promise(r => setTimeout(r, 200));
                try {
                    // Write enough to trigger the error
                    await writer.write(new Uint8Array(65536));
                    await writer.write(new Uint8Array(65536));
                    return { errored: false };
                } catch (e) {
                    return { errored: true, message: e.toString() };
                }
            """,
            )

    assert isinstance(result, dict)
    assert result["errored"] is True


async def test_recv_read_partial(start_server: ServerFactory, run_js: RunJS) -> None:
    """Server recv.read(10) returns at most 10 bytes."""
    async with start_server() as (server, port, hash_b64):
        chunk: bytes = b""

        async def server_side() -> None:
            nonlocal chunk
            request = await server.accept()
            assert request is not None
            session = await request.accept()
            async with session:
                send, recv = await session.accept_bi()
                async with send:
                    chunk = await recv.read(10)
                await session.wait_closed()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            await run_js(
                port,
                hash_b64,
                """
                const stream = await transport.createBidirectionalStream();
                const payload = new Uint8Array(100);
                for (let i = 0; i < 100; i++) payload[i] = i;
                await writeAll(stream.writable, payload);
                return true;
            """,
            )

    assert 0 < len(chunk) <= 10


async def test_recv_readexactly_success(
    start_server: ServerFactory, run_js: RunJS
) -> None:
    """Browser sends exactly 10 bytes → server readexactly(10) succeeds."""
    async with start_server() as (server, port, hash_b64):
        data: bytes = b""

        async def server_side() -> None:
            nonlocal data
            request = await server.accept()
            assert request is not None
            session = await request.accept()
            async with session:
                send, recv = await session.accept_bi()
                async with send:
                    data = await recv.readexactly(10)
                await session.wait_closed()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            await run_js(
                port,
                hash_b64,
                """
                const stream = await transport.createBidirectionalStream();
                const payload = new Uint8Array([0,1,2,3,4,5,6,7,8,9]);
                await writeAll(stream.writable, payload);
                return true;
            """,
            )

    assert data == bytes(range(10))


async def test_recv_readexactly_incomplete(
    start_server: ServerFactory, run_js: RunJS
) -> None:
    """Browser sends 5 bytes + FIN → server readexactly(10) raises StreamIncompleteReadError."""
    async with start_server() as (server, port, hash_b64):
        error: web_transport.StreamIncompleteReadError | None = None

        async def server_side() -> None:
            nonlocal error
            request = await server.accept()
            assert request is not None
            session = await request.accept()
            async with session:
                send, recv = await session.accept_bi()
                async with send:
                    try:
                        await recv.readexactly(10)
                    except web_transport.StreamIncompleteReadError as e:
                        error = e
                await session.wait_closed()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            await run_js(
                port,
                hash_b64,
                """
                const stream = await transport.createBidirectionalStream();
                const payload = new Uint8Array([0,1,2,3,4]);
                await writeAll(stream.writable, payload);
                return true;
            """,
            )

    assert error is not None
    assert error.expected == 10
    assert len(error.partial) == 5


async def test_recv_read_with_limit_exceeded(
    start_server: ServerFactory, run_js: RunJS
) -> None:
    """Browser sends >100 bytes → server recv.read(limit=100) raises StreamTooLongError."""
    async with start_server() as (server, port, hash_b64):
        error: web_transport.StreamTooLongError | None = None

        async def server_side() -> None:
            nonlocal error
            request = await server.accept()
            assert request is not None
            session = await request.accept()
            async with session:
                send, recv = await session.accept_bi()
                async with send:
                    try:
                        await recv.read(limit=100)
                    except web_transport.StreamTooLongError as e:
                        error = e
                await session.wait_closed()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            await run_js(
                port,
                hash_b64,
                """
                const stream = await transport.createBidirectionalStream();
                const payload = new Uint8Array(200);
                await writeAll(stream.writable, payload);
                return true;
            """,
            )

    assert error is not None
    assert error.limit == 100


async def test_recv_read_with_limit_not_exceeded(
    start_server: ServerFactory, run_js: RunJS
) -> None:
    """Browser sends 50 bytes → server recv.read(limit=100) succeeds."""
    async with start_server() as (server, port, hash_b64):
        data: bytes = b""

        async def server_side() -> None:
            nonlocal data
            request = await server.accept()
            assert request is not None
            session = await request.accept()
            async with session:
                send, recv = await session.accept_bi()
                async with send:
                    data = await recv.read(limit=100)
                await session.wait_closed()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            await run_js(
                port,
                hash_b64,
                """
                const stream = await transport.createBidirectionalStream();
                const payload = new Uint8Array(50);
                await writeAll(stream.writable, payload);
                return true;
            """,
            )

    assert len(data) == 50


async def test_recv_async_iteration(start_server: ServerFactory, run_js: RunJS) -> None:
    """Server uses async for to collect chunks from browser."""
    async with start_server() as (server, port, hash_b64):
        chunks: list[bytes] = []

        async def server_side() -> None:
            request = await server.accept()
            assert request is not None
            session = await request.accept()
            async with session:
                send, recv = await session.accept_bi()
                async with send:
                    async for chunk in recv:
                        chunks.append(chunk)
                await session.wait_closed()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            await run_js(
                port,
                hash_b64,
                """
                const stream = await transport.createBidirectionalStream();
                const writer = stream.writable.getWriter();
                await writer.write(new TextEncoder().encode("a"));
                await writer.write(new TextEncoder().encode("b"));
                await writer.write(new TextEncoder().encode("c"));
                await writer.close();
                return true;
            """,
            )

    assert len(chunks) > 0
    assert b"".join(chunks) == b"abc"


async def test_send_context_manager_finishes_on_clean_exit(
    start_server: ServerFactory, run_js: RunJS
) -> None:
    """async with send: clean exit → browser reads EOF."""
    async with start_server() as (server, port, hash_b64):

        async def server_side() -> None:
            request = await server.accept()
            assert request is not None
            session = await request.accept()
            async with session:
                send, recv = await session.accept_bi()
                async with send:
                    await send.write(b"context-data")
                # send is finished after context manager exits
                await recv.read()  # wait for browser close
                await session.wait_closed()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            result = await run_js(
                port,
                hash_b64,
                """
                const stream = await transport.createBidirectionalStream();
                const data = await readAllString(stream.readable);
                const writer = stream.writable.getWriter();
                await writer.close();
                return data;
            """,
            )

    assert result == "context-data"


async def test_send_context_manager_resets_on_exception(
    start_server: ServerFactory, run_js: RunJS
) -> None:
    """async with send: + raise → browser read rejects."""
    async with start_server() as (server, port, hash_b64):

        async def server_side() -> None:
            request = await server.accept()
            assert request is not None
            session = await request.accept()
            async with session:
                send, recv = await session.accept_bi()
                try:
                    async with send:
                        await send.write(b"partial")
                        raise ValueError("intentional error")
                except ValueError:
                    pass
                await recv.read()  # wait for browser close
                await session.wait_closed()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            result: Any = await run_js(
                port,
                hash_b64,
                """
                const stream = await transport.createBidirectionalStream();
                const reader = stream.readable.getReader();
                try {
                    // Read until error or done
                    while (true) {
                        const { value, done } = await reader.read();
                        if (done) break;
                    }
                    return { errored: false };
                } catch (e) {
                    return { errored: true, message: e.toString() };
                } finally {
                    const writer = stream.writable.getWriter();
                    await writer.close();
                }
            """,
            )

    assert isinstance(result, dict)
    assert result["errored"] is True


async def test_recv_context_manager_stops_if_not_eof(
    start_server: ServerFactory, run_js: RunJS
) -> None:
    """async with recv: exit before EOF → browser's subsequent write rejects."""
    async with start_server() as (server, port, hash_b64):

        async def server_side() -> None:
            request = await server.accept()
            assert request is not None
            session = await request.accept()
            async with session:
                send, recv = await session.accept_bi()
                async with send:
                    async with recv:
                        # Read one chunk, then exit (stop without reading to EOF)
                        await recv.read(10)
                await session.wait_closed()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            result: Any = await run_js(
                port,
                hash_b64,
                """
                const stream = await transport.createBidirectionalStream();
                const writer = stream.writable.getWriter();
                // Write initial data
                await writer.write(new Uint8Array(20));
                // Give server time to stop
                await new Promise(r => setTimeout(r, 300));
                try {
                    // Try writing more — should eventually fail
                    for (let i = 0; i < 10; i++) {
                        await writer.write(new Uint8Array(65536));
                    }
                    return { errored: false };
                } catch (e) {
                    return { errored: true };
                }
            """,
            )

    assert isinstance(result, dict)
    assert result["errored"] is True


async def test_write_some_returns_count(
    start_server: ServerFactory, run_js: RunJS
) -> None:
    """Server send.write_some(data) returns int in range (0, len(data)]."""
    async with start_server() as (server, port, hash_b64):
        written: int = 0

        async def server_side() -> None:
            nonlocal written
            request = await server.accept()
            assert request is not None
            session = await request.accept()
            async with session:
                send, recv = await session.accept_bi()
                async with send:
                    written = await send.write_some(b"hello world")
                    await recv.read()  # wait for browser close
                await session.wait_closed()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            await run_js(
                port,
                hash_b64,
                """
                const stream = await transport.createBidirectionalStream();
                // Read whatever server sent
                const reader = stream.readable.getReader();
                await reader.read();
                reader.releaseLock();
                // Close writable
                const writer = stream.writable.getWriter();
                await writer.close();
                return true;
            """,
            )

    assert 0 < written <= len(b"hello world")


async def test_read_after_eof_returns_empty(
    start_server: ServerFactory, run_js: RunJS
) -> None:
    """After EOF, recv.read() returns b'' idempotently."""
    async with start_server() as (server, port, hash_b64):
        reads: list[bytes] = []

        async def server_side() -> None:
            request = await server.accept()
            assert request is not None
            session = await request.accept()
            async with session:
                send, recv = await session.accept_bi()
                async with send:
                    # Read until EOF
                    data = await recv.read()
                    reads.append(data)
                    # Read again — should be b""
                    data2 = await recv.read()
                    reads.append(data2)
                    # And again
                    data3 = await recv.read()
                    reads.append(data3)
                await session.wait_closed()

        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            await run_js(
                port,
                hash_b64,
                """
                const stream = await transport.createBidirectionalStream();
                await writeAllString(stream.writable, "x");
                return true;
            """,
            )

    assert reads[0] == b"x"
    assert reads[1] == b""
    assert reads[2] == b""
