"""Framework-free HTTP/SSE client for the Compass API.

Kept separate from Streamlit so the transport layer (auth, error mapping, SSE parsing)
stays unit-testable with `httpx.MockTransport` and reusable outside the UI.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx
from httpx_sse import connect_sse

DEMO_USERS: dict[str, tuple[str, str]] = {
    "vera": ("viewer-demo", "Vera — Viewer"),
    "anil": ("analyst-demo", "Anil — Analyst"),
    "ada": ("admin-demo", "Ada — Admin"),
}


class ApiError(Exception):
    """Normalised API failure raised instead of leaking httpx/streamlit internals upstream."""

    def __init__(
        self, status: int, code: str, message: str, retry_after: float | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after = retry_after

    @classmethod
    def from_response(cls, response: httpx.Response) -> ApiError:
        """Build an error from any non-2xx response, tolerating non-JSON or empty bodies."""
        response.read()
        try:
            body: dict[str, Any] = response.json()
        except ValueError:
            body = {}
        payload = body.get("error") if isinstance(body, dict) else None
        payload = payload if isinstance(payload, dict) else {}
        retry_header = response.headers.get("Retry-After")
        try:
            retry_after = float(retry_header) if retry_header is not None else None
        except ValueError:
            retry_after = None
        return cls(
            status=response.status_code,
            code=str(payload.get("code") or "http_error"),
            message=str(payload.get("message") or f"HTTP {response.status_code}"),
            retry_after=retry_after,
        )


@dataclass(frozen=True)
class StreamEvent:
    """One decoded SSE event from the chat stream."""

    type: str
    node: str | None
    data: dict[str, Any]
    ts: float | None


class CompassClient:
    """Synchronous client for the Compass FastAPI backend."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 180.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=5.0),
            transport=transport,
        )
        self._token: str | None = None
        self._user: dict[str, Any] | None = None

    @property
    def token(self) -> str | None:
        return self._token

    @property
    def user(self) -> dict[str, Any] | None:
        return self._user

    def login(self, username: str, password: str) -> dict[str, Any]:
        """Exchange credentials for a bearer token; non-200 raises `ApiError`."""
        response = self._request(
            "POST", "/auth/login", json={"username": username, "password": password}
        )
        body = response.json()
        self._token = str(body.get("access_token") or "")
        user = body.get("user")
        self._user = user if isinstance(user, dict) else None
        return body

    def logout(self) -> None:
        """Drop credentials locally; tokens are stateless server-side so no API call is needed."""
        self._token = None
        self._user = None

    def _sse_stream(self, method: str, url: str, payload: dict[str, Any]) -> Iterator[StreamEvent]:
        """Yield decoded SSE events for any streaming endpoint (chat and resume)."""
        token = self._require_token()
        try:
            with connect_sse(
                self._client,
                method,
                url,
                json=payload,
                headers={"Authorization": f"Bearer {token}"},
            ) as source:
                if source.response.status_code != 200:
                    raise ApiError.from_response(source.response)
                for sse in source.iter_sse():
                    event = _parse_event(sse)
                    if event is not None:
                        yield event
        except httpx.TransportError as exc:
            raise ApiError(0, "unreachable", f"Cannot reach the Compass API: {exc}") from exc

    def stream_chat(self, message: str, thread_id: str | None) -> Iterator[StreamEvent]:
        """Yield decoded SSE events for one chat turn.

        Non-200 responses raise `ApiError` (body is read first); individual events with
        malformed JSON are skipped rather than aborting the stream.
        """
        return self._sse_stream(
            "POST", "/chat/stream", {"message": message, "thread_id": thread_id}
        )

    def resume_chat(self, thread_id: str, approved: bool) -> Iterator[StreamEvent]:
        """Continue a thread paused on a human approval with the chosen decision."""
        return self._sse_stream(
            "POST", "/chat/resume", {"thread_id": thread_id, "approved": approved}
        )

    def history(self, thread_id: str) -> dict[str, Any]:
        """Fetch a stored conversation, so the UI can rehydrate a thread after reload."""
        token = self._require_token()
        response = self._request(
            "GET",
            f"/chat/history/{thread_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        body = response.json()
        return body if isinstance(body, dict) else {}

    def health(self) -> bool:
        """Cheap liveness probe for the sidebar indicator; never raises."""
        try:
            return self._request("GET", "/health").is_success
        except (httpx.HTTPError, ApiError):
            return False

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Send one request, mapping transport failures and non-2xx to `ApiError`."""
        try:
            response = self._client.request(method, url, **kwargs)
        except httpx.TransportError as exc:
            raise ApiError(0, "unreachable", f"Cannot reach the Compass API: {exc}") from exc
        if not response.is_success:
            raise ApiError.from_response(response)
        return response

    def _require_token(self) -> str:
        if not self._token:
            raise ApiError(0, "not_authenticated", "Sign in before chatting.")
        return self._token


def _parse_event(sse: Any) -> StreamEvent | None:
    """Decode one SSE payload; returns None for events whose JSON is unusable."""
    try:
        body = json.loads(sse.data)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(body, dict):
        return None
    data = body.get("data")
    ts = body.get("ts")
    return StreamEvent(
        type=str(body.get("type") or sse.event or "unknown"),
        node=body.get("node") if isinstance(body.get("node"), str) else None,
        data=data if isinstance(data, dict) else {},
        ts=ts if isinstance(ts, float | int) else None,
    )
