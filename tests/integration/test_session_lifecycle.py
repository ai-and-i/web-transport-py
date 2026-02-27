"""Integration tests for session lifecycle management."""

import asyncio

import pytest

import web_transport


@pytest.mark.asyncio
async def test_session_close_with_code_and_reason(session_pair):
    """close(42, "goodbye") -> peer sees SessionClosedByPeer."""
    server_session, client_session = session_pair

    async def accept_side():
        with pytest.raises(web_transport.SessionClosedByPeer) as exc_info:
            await server_session.accept_bi()
        assert exc_info.value.source == "session"
        assert exc_info.value.code == 42
        assert exc_info.value.reason == "goodbye"

    # Start waiting before close so the notification is properly received
    task = asyncio.create_task(accept_side())
    await asyncio.sleep(0.05)
    client_session.close(42, "goodbye")
    await asyncio.wait_for(task, timeout=5.0)


@pytest.mark.asyncio
async def test_session_close_reason_before_close(session_pair):
    """close_reason returns None on an open session."""
    server_session, client_session = session_pair

    assert client_session.close_reason is None
    assert server_session.close_reason is None


@pytest.mark.asyncio
async def test_session_close_reason_after_close(session_pair):
    """After close() + wait_closed(), close_reason is set."""
    server_session, client_session = session_pair

    client_session.close(1, "test")
    await client_session.wait_closed()

    reason = client_session.close_reason
    assert reason is not None
    assert isinstance(reason, web_transport.SessionError)


@pytest.mark.asyncio
async def test_session_wait_closed(session_pair):
    """wait_closed() returns only after peer closes."""
    server_session, client_session = session_pair

    closed = False

    async def wait_for_close():
        nonlocal closed
        await server_session.wait_closed()
        closed = True

    task = asyncio.create_task(wait_for_close())
    await asyncio.sleep(0.1)
    assert not closed

    client_session.close()
    await asyncio.wait_for(task, timeout=5.0)
    assert closed


@pytest.mark.asyncio
async def test_session_remote_address(session_pair):
    """remote_address returns (str, int) with valid IP and port."""
    server_session, client_session = session_pair

    addr = client_session.remote_address
    assert isinstance(addr, tuple)
    assert len(addr) == 2
    host, port = addr
    assert isinstance(host, str)
    assert isinstance(port, int)
    assert port > 0

    addr_s = server_session.remote_address
    assert isinstance(addr_s, tuple)
    host_s, port_s = addr_s
    assert isinstance(host_s, str)
    assert isinstance(port_s, int)
    assert port_s > 0


@pytest.mark.asyncio
async def test_session_rtt(session_pair):
    """rtt returns a positive float."""
    _server_session, client_session = session_pair

    rtt = client_session.rtt
    assert isinstance(rtt, float)
    assert rtt > 0


@pytest.mark.asyncio
async def test_session_context_manager_closes(self_signed_cert, cert_hash):
    """async with session -> session closes on exit."""
    cert, key = self_signed_cert

    async with web_transport.Server(
        certificate_chain=[cert], private_key=key, bind="[::1]:0"
    ) as server:
        _, port = server.local_addr

        async def server_task():
            request = await server.accept()
            assert request is not None
            session = await request.accept()
            # Wait for the client to close
            await session.wait_closed()
            reason = session.close_reason
            assert reason is not None

        async def client_task():
            async with web_transport.Client(
                server_certificate_hashes=[cert_hash]
            ) as client:
                session = await client.connect(f"https://[::1]:{port}")
                async with session:
                    pass  # Clean exit -> session closes

        await asyncio.gather(
            asyncio.create_task(server_task()),
            asyncio.create_task(client_task()),
        )


@pytest.mark.asyncio
async def test_session_context_manager_closes_on_exception(self_signed_cert, cert_hash):
    """async with session: raise -> session closes, peer sees it."""
    cert, key = self_signed_cert

    async with web_transport.Server(
        certificate_chain=[cert], private_key=key, bind="[::1]:0"
    ) as server:
        _, port = server.local_addr

        async def server_task():
            request = await server.accept()
            assert request is not None
            session = await request.accept()
            await session.wait_closed()
            reason = session.close_reason
            assert reason is not None

        async def client_task():
            async with web_transport.Client(
                server_certificate_hashes=[cert_hash]
            ) as client:
                session = await client.connect(f"https://[::1]:{port}")
                with pytest.raises(RuntimeError):
                    async with session:
                        raise RuntimeError("intentional")

        await asyncio.gather(
            asyncio.create_task(server_task()),
            asyncio.create_task(client_task()),
        )


@pytest.mark.asyncio
async def test_open_stream_after_close(session_pair):
    """close() then open_bi() -> SessionClosedLocally."""
    _server_session, client_session = session_pair

    client_session.close()
    await client_session.wait_closed()

    with pytest.raises(web_transport.SessionClosedLocally):
        await client_session.open_bi()


@pytest.mark.asyncio
async def test_accept_stream_after_peer_close(session_pair):
    """Peer closes -> pending accept_bi() -> SessionClosedByPeer."""
    server_session, client_session = session_pair

    async def accept_side():
        with pytest.raises(web_transport.SessionClosedByPeer):
            await server_session.accept_bi()

    task = asyncio.create_task(accept_side())
    await asyncio.sleep(0.05)
    client_session.close()
    await asyncio.wait_for(task, timeout=5.0)


@pytest.mark.asyncio
async def test_send_datagram_after_close(session_pair):
    """close() then send_datagram() -> SessionClosed."""
    _server_session, client_session = session_pair

    client_session.close()
    await client_session.wait_closed()

    with pytest.raises(web_transport.SessionClosed):
        client_session.send_datagram(b"x")


@pytest.mark.asyncio
async def test_receive_datagram_after_peer_close(session_pair):
    """Peer closes -> pending receive_datagram() -> SessionClosedByPeer."""
    server_session, client_session = session_pair

    async def recv_side():
        with pytest.raises(web_transport.SessionClosedByPeer):
            await server_session.receive_datagram()

    task = asyncio.create_task(recv_side())
    await asyncio.sleep(0.05)
    client_session.close()
    await asyncio.wait_for(task, timeout=5.0)
