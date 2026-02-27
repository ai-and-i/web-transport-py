"""Integration tests for stream reset, stop, cancellation, and half-closed states."""

import asyncio

import pytest

import web_transport


@pytest.mark.asyncio
async def test_send_reset_with_code(session_pair):
    """send.reset(42) -> peer read() raises StreamClosedByPeer(reset, 42)."""
    server_session, client_session = session_pair

    send, _recv = await client_session.open_bi()

    async def server_side():
        _send_s, recv_s = await server_session.accept_bi()
        with pytest.raises(web_transport.StreamClosedByPeer) as exc_info:
            await recv_s.read()
        assert exc_info.value.kind == "reset"
        assert exc_info.value.code == 42

    task = asyncio.create_task(server_side())
    await asyncio.sleep(0.05)
    send.reset(42)
    await asyncio.wait_for(task, timeout=5.0)


@pytest.mark.asyncio
async def test_send_reset_default_code(session_pair):
    """send.reset() -> code=0."""
    server_session, client_session = session_pair

    send, _recv = await client_session.open_bi()

    async def server_side():
        _send_s, recv_s = await server_session.accept_bi()
        with pytest.raises(web_transport.StreamClosedByPeer) as exc_info:
            await recv_s.read()
        assert exc_info.value.kind == "reset"
        assert exc_info.value.code == 0

    task = asyncio.create_task(server_side())
    await asyncio.sleep(0.05)
    send.reset()
    await asyncio.wait_for(task, timeout=5.0)


@pytest.mark.asyncio
async def test_double_reset_raises(session_pair):
    """reset() twice -> StreamClosedLocally."""
    _server_session, client_session = session_pair

    send, _recv = await client_session.open_bi()
    send.reset()
    with pytest.raises(web_transport.StreamClosedLocally):
        send.reset()


@pytest.mark.asyncio
async def test_write_after_reset_raises(session_pair):
    """reset() -> write() -> StreamClosedLocally."""
    _server_session, client_session = session_pair

    send, _recv = await client_session.open_bi()
    send.reset()
    with pytest.raises(web_transport.StreamClosedLocally):
        await send.write(b"x")


@pytest.mark.asyncio
async def test_write_after_finish_raises(session_pair):
    """finish() -> write() -> StreamClosedLocally."""
    _server_session, client_session = session_pair

    send, _recv = await client_session.open_bi()
    send.finish()
    with pytest.raises(web_transport.StreamClosedLocally):
        await send.write(b"x")


@pytest.mark.asyncio
async def test_recv_stop_with_code(session_pair):
    """recv.stop(42) -> peer write() raises StreamClosedByPeer(stop, 42)."""
    server_session, client_session = session_pair

    send, recv = await client_session.open_bi()

    async def server_side():
        send_s, _recv_s = await server_session.accept_bi()
        # Wait for STOP_SENDING to arrive
        await asyncio.sleep(0.1)
        with pytest.raises(web_transport.StreamClosedByPeer) as exc_info:
            for _ in range(100):
                await send_s.write(b"x" * 1024)
        assert exc_info.value.kind == "stop"
        assert exc_info.value.code == 42

    task = asyncio.create_task(server_side())
    recv.stop(42)
    await asyncio.wait_for(task, timeout=5.0)


@pytest.mark.asyncio
async def test_recv_stop_default_code(session_pair):
    """recv.stop() -> code=0."""
    server_session, client_session = session_pair

    send, recv = await client_session.open_bi()

    async def server_side():
        send_s, _recv_s = await server_session.accept_bi()
        await asyncio.sleep(0.1)
        with pytest.raises(web_transport.StreamClosedByPeer) as exc_info:
            for _ in range(100):
                await send_s.write(b"x" * 1024)
        assert exc_info.value.kind == "stop"
        assert exc_info.value.code == 0

    task = asyncio.create_task(server_side())
    recv.stop()
    await asyncio.wait_for(task, timeout=5.0)


@pytest.mark.asyncio
async def test_double_stop_raises(session_pair):
    """stop() twice -> StreamClosedLocally."""
    _server_session, client_session = session_pair

    _send, recv = await client_session.open_bi()
    recv.stop()
    with pytest.raises(web_transport.StreamClosedLocally):
        recv.stop()


@pytest.mark.asyncio
async def test_read_after_stop_raises(session_pair):
    """stop() -> read() -> StreamClosedLocally."""
    _server_session, client_session = session_pair

    _send, recv = await client_session.open_bi()
    recv.stop()
    with pytest.raises(web_transport.StreamClosedLocally):
        await recv.read()


@pytest.mark.asyncio
async def test_reset_cancels_pending_write(session_pair):
    """Start write() task, call reset() -> write raises StreamClosedLocally."""
    server_session, client_session = session_pair

    send, _recv = await client_session.open_bi()

    # Accept the stream on the server side but don't read (causes backpressure)
    async def server_side():
        _send_s, _recv_s = await server_session.accept_bi()
        await asyncio.sleep(2.0)

    server_task = asyncio.create_task(server_side())

    async def write_large():
        with pytest.raises(web_transport.StreamClosedLocally):
            while True:
                await send.write(b"x" * 65536)

    write_task = asyncio.create_task(write_large())
    await asyncio.sleep(0.1)
    send.reset()
    await asyncio.wait_for(write_task, timeout=5.0)
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_stop_cancels_pending_read(session_pair):
    """Start read() on idle stream, call stop() -> StreamClosedLocally."""
    server_session, client_session = session_pair

    _send, recv = await client_session.open_bi()

    # Accept the stream on the server side but don't write anything
    async def server_side():
        _send_s, _recv_s = await server_session.accept_bi()
        await asyncio.sleep(2.0)

    server_task = asyncio.create_task(server_side())

    async def read_pending():
        with pytest.raises(web_transport.StreamClosedLocally):
            await recv.read()

    read_task = asyncio.create_task(read_pending())
    await asyncio.sleep(0.1)
    recv.stop()
    await asyncio.wait_for(read_task, timeout=5.0)
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_send_context_manager_resets_on_exception(session_pair):
    """async with send: raise -> peer sees StreamClosedByPeer(reset)."""
    server_session, client_session = session_pair

    send, _recv = await client_session.open_bi()

    async def server_side():
        _send_s, recv_s = await server_session.accept_bi()
        with pytest.raises(web_transport.StreamClosedByPeer) as exc_info:
            await recv_s.read()
        assert exc_info.value.kind == "reset"

    task = asyncio.create_task(server_side())

    with pytest.raises(RuntimeError):
        async with send:
            raise RuntimeError("intentional")

    await asyncio.wait_for(task, timeout=5.0)


@pytest.mark.asyncio
async def test_recv_context_manager_stops_on_exception(session_pair):
    """async with recv: raise -> peer's write raises StreamClosedByPeer(stop)."""
    server_session, client_session = session_pair

    send, recv = await client_session.open_bi()

    async def server_side():
        send_s, _recv_s = await server_session.accept_bi()
        # Give time for STOP_SENDING to propagate
        await asyncio.sleep(0.2)
        with pytest.raises(web_transport.StreamClosedByPeer) as exc_info:
            for _ in range(100):
                await send_s.write(b"x" * 1024)
        assert exc_info.value.kind == "stop"

    task = asyncio.create_task(server_side())

    with pytest.raises(RuntimeError):
        async with recv:
            raise RuntimeError("intentional")

    await asyncio.wait_for(task, timeout=5.0)


@pytest.mark.asyncio
async def test_recv_context_manager_stops_if_not_at_eof(session_pair):
    """Clean exit before EOF -> peer's write raises StreamClosedByPeer(stop)."""
    server_session, client_session = session_pair

    send, recv = await client_session.open_bi()

    async def server_side():
        send_s, _recv_s = await server_session.accept_bi()
        # Write some data but don't finish
        await send_s.write(b"hello")
        # Give time for STOP_SENDING after context manager exit
        await asyncio.sleep(0.2)
        with pytest.raises(web_transport.StreamClosedByPeer) as exc_info:
            for _ in range(100):
                await send_s.write(b"x" * 1024)
        assert exc_info.value.kind == "stop"

    task = asyncio.create_task(server_side())
    await asyncio.sleep(0.05)

    async with recv:
        # Read some data but exit before EOF
        data = await recv.read(5)
        assert data == b"hello"
        # Exit without reaching EOF -> stop() should be sent

    await asyncio.wait_for(task, timeout=5.0)


@pytest.mark.asyncio
async def test_recv_context_manager_no_stop_at_eof(session_pair):
    """Read all data then clean exit -> no stop sent."""
    server_session, client_session = session_pair

    send, recv = await client_session.open_bi()

    async def server_side():
        send_s, _recv_s = await server_session.accept_bi()
        async with send_s:
            await send_s.write(b"hello")
        # finish() was called via context manager, so client will see EOF

    task = asyncio.create_task(server_side())

    async with recv:
        data = await recv.read()
        assert data == b"hello"

    await task  # Server task should complete without errors


@pytest.mark.asyncio
async def test_half_closed_write_after_recv_eof(session_pair):
    """Peer finishes send -> our send still works on bidi stream."""
    server_session, client_session = session_pair

    send, recv = await client_session.open_bi()

    async def server_side():
        send_s, recv_s = await server_session.accept_bi()
        # Finish our send side
        send_s.finish()
        # Read what the client sends after our EOF
        data = await recv_s.read()
        return data

    task = asyncio.create_task(server_side())

    # Wait for peer's FIN
    data = await recv.read()
    assert data == b""  # EOF

    # Our send side should still work
    async with send:
        await send.write(b"still works")

    result = await asyncio.wait_for(task, timeout=5.0)
    assert result == b"still works"


@pytest.mark.asyncio
async def test_half_closed_read_after_send_finish(session_pair):
    """We finish send -> our recv still works on bidi stream."""
    server_session, client_session = session_pair

    send, recv = await client_session.open_bi()

    async def server_side():
        send_s, recv_s = await server_session.accept_bi()
        # Wait for client's FIN
        data = await recv_s.read()
        assert data == b"hello"
        # Send our response
        async with send_s:
            await send_s.write(b"response")

    task = asyncio.create_task(server_side())

    # Finish our send side
    async with send:
        await send.write(b"hello")

    # Our recv side should still work
    data = await recv.read()
    assert data == b"response"

    await task


@pytest.mark.asyncio
async def test_peer_reset_during_read(session_pair):
    """Peer reset(7) while we await recv.read() -> StreamClosedByPeer."""
    server_session, client_session = session_pair

    _send, recv = await client_session.open_bi()

    async def server_side():
        send_s, _recv_s = await server_session.accept_bi()
        await asyncio.sleep(0.1)
        send_s.reset(7)

    task = asyncio.create_task(server_side())

    with pytest.raises(web_transport.StreamClosedByPeer) as exc_info:
        await recv.read()

    assert exc_info.value.kind == "reset"
    assert exc_info.value.code == 7
    await task


@pytest.mark.asyncio
async def test_uni_send_reset(session_pair):
    """Uni stream reset -> peer sees StreamClosedByPeer(reset)."""
    server_session, client_session = session_pair

    send = await client_session.open_uni()

    async def server_side():
        recv = await server_session.accept_uni()
        with pytest.raises(web_transport.StreamClosedByPeer) as exc_info:
            await recv.read()
        assert exc_info.value.kind == "reset"
        assert exc_info.value.code == 0

    task = asyncio.create_task(server_side())
    await asyncio.sleep(0.05)
    send.reset()
    await asyncio.wait_for(task, timeout=5.0)


@pytest.mark.asyncio
async def test_uni_recv_stop(session_pair):
    """Uni stream stop -> peer write raises StreamClosedByPeer(stop)."""
    server_session, client_session = session_pair

    send = await client_session.open_uni()

    async def server_side():
        recv = await server_session.accept_uni()
        recv.stop(42)

    task = asyncio.create_task(server_side())
    await asyncio.sleep(0.2)

    with pytest.raises(web_transport.StreamClosedByPeer) as exc_info:
        for _ in range(100):
            await send.write(b"x" * 1024)

    assert exc_info.value.kind == "stop"
    assert exc_info.value.code == 42
    await task


@pytest.mark.asyncio
async def test_readexactly_during_peer_reset(session_pair):
    """readexactly() interrupted by peer reset -> StreamClosedByPeer."""
    server_session, client_session = session_pair

    _send, recv = await client_session.open_bi()

    async def server_side():
        send_s, _recv_s = await server_session.accept_bi()
        await asyncio.sleep(0.1)
        send_s.reset(7)

    task = asyncio.create_task(server_side())

    with pytest.raises(web_transport.StreamClosedByPeer) as exc_info:
        await recv.readexactly(1000)

    assert exc_info.value.kind == "reset"
    assert exc_info.value.code == 7
    await task


@pytest.mark.asyncio
async def test_double_finish_raises(session_pair):
    """finish() after finish() -> StreamClosedLocally."""
    _server_session, client_session = session_pair

    send, _recv = await client_session.open_bi()
    await send.finish()
    with pytest.raises(web_transport.StreamClosedLocally):
        await send.finish()
