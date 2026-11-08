"""Unit tests for the framework-free API client, using `httpx.MockTransport` (no network)."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from ui.client import ApiError, CompassClient, StreamEvent


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> CompassClient:
    return CompassClient("http://testserver", transport=httpx.MockTransport(handler))


def auth_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/auth/login":
        return httpx.Response(
            200,
            json={
                "access_token": "jwt-token",
                "token_type": "bearer",
                "user": {
                    "user_id": "u1",
                    "name": "Vera",
                    "role": "viewer",
                    "department": "risk",
                },
            },
        )
    return httpx.Response(404, json={"error": {"code": "not_found", "message": "nope"}})


def test_login_success_stores_token_and_user() -> None:
    client = make_client(auth_handler)
    body = client.login("vera", "viewer-demo")

    assert body["access_token"] == "jwt-token"
    assert client.token == "jwt-token"
    assert client.user is not None
    assert client.user["role"] == "viewer"


def test_login_bad_credentials_raises_api_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401, json={"error": {"code": "invalid_credentials", "message": "Bad credentials"}}
        )

    client = make_client(handler)
    with pytest.raises(ApiError) as excinfo:
        client.login("vera", "wrong")
    assert excinfo.value.status == 401
    assert excinfo.value.code == "invalid_credentials"
    assert excinfo.value.message == "Bad credentials"


SSE_BODY = """event: node_start
data: {"type": "node_start", "node": "supervisor", "data": {}, "ts": 1.0}

event: token
data: {"type": "token", "node": "response", "data": {"text": "Hello"}, "ts": 2.0}

event: broken
data: {not-valid-json

event: done
data: {"type": "done", "node": null, "data": {"thread_id": "t-123"}, "ts": 3.0}

"""


def test_stream_chat_parses_sse_and_sends_auth_and_body() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/login":
            return auth_handler(request)
        captured.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=SSE_BODY.encode()
        )

    client = make_client(handler)
    client.login("vera", "viewer-demo")
    events = list(client.stream_chat("What changed?", "t-123"))

    assert [e.type for e in events] == ["node_start", "token", "done"]  # malformed line skipped
    assert events[0].node == "supervisor"
    assert events[1].data == {"text": "Hello"}
    assert events[2].data["thread_id"] == "t-123"
    assert events[2].ts == 3.0

    assert len(captured) == 1
    sent = captured[0]
    assert sent.headers["Authorization"] == "Bearer jwt-token"
    assert sent.headers["Accept"] == "text/event-stream"
    sent_body: dict[str, Any] = json.loads(captured[0].read())
    assert sent_body == {"message": "What changed?", "thread_id": "t-123"}


def test_stream_chat_429_sets_retry_after() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/login":
            return auth_handler(request)
        return httpx.Response(
            429,
            headers={"Retry-After": "7"},
            json={"error": {"code": "rate_limited", "message": "Too many requests"}},
        )

    client = make_client(handler)
    client.login("vera", "viewer-demo")
    with pytest.raises(ApiError) as excinfo:
        list(client.stream_chat("hi", None))
    assert excinfo.value.status == 429
    assert excinfo.value.retry_after == 7.0
    assert excinfo.value.code == "rate_limited"


def test_stream_chat_without_login() -> None:
    client = make_client(auth_handler)
    with pytest.raises(ApiError) as excinfo:
        list(client.stream_chat("hi", None))
    assert excinfo.value.code == "not_authenticated"
    assert excinfo.value.status == 0


def test_history_without_login() -> None:
    client = make_client(auth_handler)
    with pytest.raises(ApiError) as excinfo:
        client.history("t-1")
    assert excinfo.value.code == "not_authenticated"


def test_transport_error_raises_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = make_client(handler)
    with pytest.raises(ApiError) as excinfo:
        client.login("vera", "viewer-demo")
    assert excinfo.value.code == "unreachable"
    assert excinfo.value.status == 0


def test_transport_error_during_stream_raises_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/auth/login":
            return auth_handler(request)
        raise httpx.ConnectError("connection refused")

    client = make_client(handler)
    client.login("vera", "viewer-demo")
    with pytest.raises(ApiError) as excinfo:
        list(client.stream_chat("hi", None))
    assert excinfo.value.code == "unreachable"


def test_health_returns_false_on_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = make_client(handler)
    assert client.health() is False


def test_stream_event_is_frozen() -> None:
    event = StreamEvent(type="token", node="response", data={"text": "x"}, ts=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.type = "other"  # type: ignore[misc]
