"""Browser interop tests for SessionRequest handling."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

import web_transport
from .conftest import _webtransport_connect_js

if TYPE_CHECKING:
    from .conftest import RunJS, RunJSRaw, ServerFactory

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_request_url_matches_target(
    start_server: ServerFactory, run_js: RunJS
) -> None:
    """request.url contains expected host and port."""
    async with start_server() as (server, port, hash_b64):
        request_url: str = ""

        async def server_side() -> None:
            nonlocal request_url
            request = await server.accept()
            assert request is not None
            request_url = request.url
            session = await request.accept()
            async with session:
                pass

        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            await run_js(port, hash_b64, "await transport.closed; return true;")

    assert "127.0.0.1" in request_url
    assert str(port) in request_url


async def test_request_url_with_path(
    start_server: ServerFactory, run_js_raw: RunJSRaw
) -> None:
    """Browser connects to https://host:port/custom/path → request.url includes path."""
    async with start_server() as (server, port, hash_b64):
        request_url: str = ""

        async def server_side() -> None:
            nonlocal request_url
            request = await server.accept()
            assert request is not None
            request_url = request.url
            session = await request.accept()
            async with session:
                await session.wait_closed()

        setup = _webtransport_connect_js(port, hash_b64)
        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            await run_js_raw(f"""
                {setup}
                const pathUrl = "https://127.0.0.1:{port}/custom/path";
                const transport = new WebTransport(pathUrl, transportOptions);
                await transport.ready;
                transport.close();
                return true;
            """)

    assert "/custom/path" in request_url


async def test_accept_then_reject_raises(
    start_server: ServerFactory, run_js_raw: RunJSRaw
) -> None:
    """request.accept() then request.reject() raises SessionError."""
    async with start_server() as (server, port, hash_b64):
        error: BaseException | None = None

        async def server_side() -> None:
            nonlocal error
            request = await server.accept()
            assert request is not None
            session = await request.accept()
            try:
                await request.reject(404)
            except web_transport.SessionError as e:
                error = e
            async with session:
                session.close()

        setup = _webtransport_connect_js(port, hash_b64)
        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            await run_js_raw(f"""
                {setup}
                const transport = new WebTransport(url, transportOptions);
                try {{
                    await transport.ready;
                }} catch (e) {{
                    // May fail if session was closed
                }}
                try {{ transport.close(); }} catch (e) {{}}
                return true;
            """)

    assert isinstance(error, web_transport.SessionError)


async def test_reject_then_accept_raises(
    start_server: ServerFactory, run_js_raw: RunJSRaw
) -> None:
    """request.reject() then request.accept() raises SessionError."""
    async with start_server() as (server, port, hash_b64):
        error: BaseException | None = None

        async def server_side() -> None:
            nonlocal error
            request = await server.accept()
            assert request is not None
            await request.reject(404)
            try:
                await request.accept()
            except web_transport.SessionError as e:
                error = e

        setup = _webtransport_connect_js(port, hash_b64)
        async with asyncio.TaskGroup() as tg:
            tg.create_task(server_side())
            await run_js_raw(f"""
                {setup}
                const transport = new WebTransport(url, transportOptions);
                try {{
                    await transport.ready;
                }} catch (e) {{
                    // Expected: session was rejected
                }}
                return true;
            """)

    assert isinstance(error, web_transport.SessionError)
