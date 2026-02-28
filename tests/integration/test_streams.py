"""Integration tests for stream read/write operations."""

import asyncio

import pytest

import web_transport


@pytest.mark.asyncio
async def test_write_some(session_pair):
    server_session, client_session = session_pair

    async def server_side():
        send, recv = await server_session.accept_bi()
        data = await recv.read()
        async with send:
            await send.write(data)

    server_task = asyncio.create_task(server_side())

    send, recv = await client_session.open_bi()
    n = await send.write_some(b"hello")
    assert isinstance(n, int)
    assert n > 0
    await send.finish()

    response = await recv.read()
    assert response == b"hello"
    await server_task


@pytest.mark.asyncio
async def test_send_stream_context_manager_finish(session_pair):
    server_session, client_session = session_pair

    send, _recv = await client_session.open_bi()
    async with send:
        await send.write(b"test data")
    # After clean exit, stream should be finished (no exception)


@pytest.mark.asyncio
async def test_recv_stream_async_iteration(session_pair):
    server_session, client_session = session_pair

    async def server_side():
        send, recv = await server_session.accept_bi()
        chunks = []
        async for chunk in recv:
            chunks.append(chunk)
        async with send:
            await send.write(b"".join(chunks))

    send, recv = await client_session.open_bi()

    server_task = asyncio.create_task(server_side())

    async with send:
        await send.write(b"chunk1")
        await send.write(b"chunk2")

    response = await recv.read()
    assert response == b"chunk1chunk2"
    await server_task


@pytest.mark.asyncio
async def test_readexactly(session_pair):
    server_session, client_session = session_pair

    async def server_side():
        send, _recv = await server_session.accept_bi()
        async with send:
            await send.write(b"exactly10!")

    _send, recv = await client_session.open_bi()
    server_task = asyncio.create_task(server_side())

    data = await recv.readexactly(10)
    assert data == b"exactly10!"
    await server_task


@pytest.mark.asyncio
async def test_readexactly_incomplete(session_pair):
    server_session, client_session = session_pair

    async def server_side():
        send, _recv = await server_session.accept_bi()
        async with send:
            await send.write(b"short")

    _send, recv = await client_session.open_bi()
    server_task = asyncio.create_task(server_side())

    with pytest.raises(web_transport.StreamIncompleteReadError) as exc_info:
        await recv.readexactly(100)

    assert exc_info.value.expected == 100
    assert exc_info.value.partial == b"short"
    await server_task


@pytest.mark.asyncio
async def test_priority(session_pair):
    _server_session, client_session = session_pair

    send, _recv = await client_session.open_bi()
    # Default priority
    p = send.priority
    assert isinstance(p, int)

    send.priority = 42
    assert send.priority == 42
    await send.finish()


@pytest.mark.asyncio
async def test_read_n_bytes(session_pair):
    """read(5) returns up to 5 bytes from longer stream."""
    server_session, client_session = session_pair

    async def server_side():
        send, _recv = await server_session.accept_bi()
        async with send:
            await send.write(b"hello world")

    _send, recv = await client_session.open_bi()
    task = asyncio.create_task(server_side())

    data = await recv.read(5)
    assert len(data) <= 5
    assert len(data) > 0
    await task


@pytest.mark.asyncio
async def test_read_n_returns_less_at_eof(session_pair):
    """read(100) on short stream returns available bytes."""
    server_session, client_session = session_pair

    async def server_side():
        send, _recv = await server_session.accept_bi()
        async with send:
            await send.write(b"hi")

    _send, recv = await client_session.open_bi()
    task = asyncio.create_task(server_side())

    # Read all chunks until empty
    chunks = []
    while True:
        data = await recv.read(100)
        if not data:
            break
        chunks.append(data)

    assert b"".join(chunks) == b"hi"
    await task


@pytest.mark.asyncio
async def test_read_zero_returns_empty(session_pair):
    """read(0) returns b"" immediately."""
    _server_session, client_session = session_pair

    send, recv = await client_session.open_bi()
    data = await recv.read(0)
    assert data == b""
    await send.finish()


@pytest.mark.asyncio
async def test_read_all_to_eof(session_pair):
    """read() (default n=-1) reads until EOF."""
    server_session, client_session = session_pair

    async def server_side():
        send, _recv = await server_session.accept_bi()
        async with send:
            await send.write(b"all the data")

    _send, recv = await client_session.open_bi()
    task = asyncio.create_task(server_side())

    data = await recv.read()
    assert data == b"all the data"
    await task


@pytest.mark.asyncio
async def test_read_with_limit(session_pair):
    """read(limit=10) on 20-byte stream -> StreamTooLongError."""
    server_session, client_session = session_pair

    async def server_side():
        send, _recv = await server_session.accept_bi()
        async with send:
            await send.write(b"x" * 20)

    _send, recv = await client_session.open_bi()
    task = asyncio.create_task(server_side())

    with pytest.raises(web_transport.StreamTooLongError) as exc_info:
        await recv.read(limit=10)

    assert exc_info.value.limit == 10
    await task


@pytest.mark.asyncio
async def test_read_after_eof_returns_empty(session_pair):
    """After EOF, read() returns b"" (idempotent)."""
    server_session, client_session = session_pair

    async def server_side():
        send, _recv = await server_session.accept_bi()
        async with send:
            await send.write(b"data")

    _send, recv = await client_session.open_bi()
    task = asyncio.create_task(server_side())

    # Read until EOF
    data = await recv.read()
    assert data == b"data"

    # Read again after EOF
    data2 = await recv.read()
    assert data2 == b""

    # And again
    data3 = await recv.read()
    assert data3 == b""

    await task


@pytest.mark.asyncio
async def test_read_n_after_eof_returns_empty(session_pair):
    """read(5) after EOF returns b"" (idempotent for n>0)."""
    server_session, client_session = session_pair

    async def server_side():
        send, _recv = await server_session.accept_bi()
        async with send:
            await send.write(b"data")

    _send, recv = await client_session.open_bi()
    task = asyncio.create_task(server_side())

    # Read until EOF
    data = await recv.read()
    assert data == b"data"

    # read(5) after EOF should return b""
    data2 = await recv.read(5)
    assert data2 == b""

    await task


@pytest.mark.asyncio
async def test_readexactly_zero(session_pair):
    """readexactly(0) returns b""."""
    _server_session, client_session = session_pair

    send, recv = await client_session.open_bi()
    data = await recv.readexactly(0)
    assert data == b""
    await send.finish()


@pytest.mark.asyncio
async def test_readexactly_after_eof(session_pair):
    """readexactly(5) after EOF -> StreamIncompleteReadError."""
    server_session, client_session = session_pair

    async def server_side():
        send, _recv = await server_session.accept_bi()
        await send.finish()  # Immediately EOF

    _send, recv = await client_session.open_bi()
    task = asyncio.create_task(server_side())

    # Wait for EOF
    data = await recv.read()
    assert data == b""

    with pytest.raises(web_transport.StreamIncompleteReadError) as exc_info:
        await recv.readexactly(5)

    assert exc_info.value.expected == 5
    assert exc_info.value.partial == b""
    await task


@pytest.mark.asyncio
async def test_write_explicit(session_pair):
    """write(data) writes all bytes (verify roundtrip)."""
    server_session, client_session = session_pair

    payload = b"explicit write test data"

    async def server_side():
        send, recv = await server_session.accept_bi()
        data = await recv.read()
        async with send:
            await send.write(data)

    task = asyncio.create_task(server_side())

    send, recv = await client_session.open_bi()
    await send.write(payload)
    await send.finish()

    response = await recv.read()
    assert response == payload
    await task


@pytest.mark.asyncio
async def test_finish_explicit(session_pair):
    """finish() signals EOF to peer."""
    server_session, client_session = session_pair

    async def server_side():
        _send, recv = await server_session.accept_bi()
        data = await recv.read()
        assert data == b""  # Only got EOF, no data

    send, _recv = await client_session.open_bi()
    task = asyncio.create_task(server_side())

    await send.finish()  # Send EOF immediately
    await asyncio.wait_for(task, timeout=5.0)


@pytest.mark.asyncio
async def test_large_transfer(session_pair):
    """Write 1 MB -> recv reads all 1 MB correctly."""
    server_session, client_session = session_pair

    size = 1024 * 1024  # 1 MB
    payload = bytes(range(256)) * (size // 256)

    async def server_side():
        send, recv = await server_session.accept_bi()
        data = await recv.read()
        async with send:
            await send.write(data)

    task = asyncio.create_task(server_side())

    send, recv = await client_session.open_bi()
    async with send:
        await send.write(payload)

    response = await recv.read()
    assert len(response) == len(payload)
    assert response == payload
    await task


@pytest.mark.asyncio
async def test_empty_write(session_pair):
    """write(b"") succeeds without error."""
    _server_session, client_session = session_pair

    send, _recv = await client_session.open_bi()
    await send.write(b"")
    await send.finish()
