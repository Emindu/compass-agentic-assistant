"""POST /chat/resume over the FastAPI app, network-free.

A minimal app (chat + auth routers, LLM-stubbed graph on an in-memory checkpointer)
exercises the approval round-trip over real SSE: who may resume, when resume is refused,
and that the stream surfaces the pending approval before `done`.
"""

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import ASGITransport
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver

from app.agents import graph as graph_mod
from app.agents.nodes import response, supervisor
from app.agents.nodes import tools as tools_node_mod
from app.agents.nodes.tools import ToolCallRequest, ToolPlan
from app.agents.state import RouteDecision
from app.api import auth as auth_mod
from app.api.routes import chat as chat_mod
from app.core.errors import AppError
from app.security import input_guard
from app.security.rbac import Role, UserContext
from app.tools import admin

ADMIN = UserContext(user_id="ada", name="Ada Admin", role=Role.admin, department="platform")
VIEWER = UserContext(user_id="vera", name="Vera Viewer", role=Role.viewer, department="retail")

ADMIN_Q = RouteDecision(
    intent="admin_action", agents=["tools"], search_query="reindex", reasoning="admin request"
)


def make_app() -> FastAPI:
    """Chat + auth on one app with the real graph replaced by an in-memory test graph."""
    api = FastAPI()

    @api.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=int(exc.status_code),
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    api.include_router(auth_mod.router)
    api.include_router(chat_mod.router)
    api.state.graph = graph_mod.build_graph(InMemorySaver())
    return api


async def admin_action_stubs(monkeypatch) -> list[dict]:
    """Stub every LLM touchpoint so an admin-question turn pauses on the approval interrupt."""
    admin_calls: list[dict] = []

    async def fake_route(_role, _schema, _messages, **_kw):
        return ADMIN_Q

    async def fake_plan(_role, _schema, _messages, **_kw):
        return ToolPlan(
            calls=[ToolCallRequest(tool="reindex_department", args={"department": "payments"})]
        )

    async def fake_stream(_role, _messages, **_kw):
        text = "Payments is being re-indexed."
        for word in text.split(" "):
            yield AIMessageChunk(content=word + " ")

    async def classifier_unavailable(_text):
        return None

    async def fake_execute(name: str, args: dict, *, user, approved: bool) -> dict:
        admin_calls.append({"name": name, "args": args, "approved": approved})
        return {"status": "reindexed"}

    monkeypatch.setattr(supervisor, "ainvoke_structured", fake_route)
    monkeypatch.setattr(tools_node_mod, "ainvoke_structured", fake_plan)
    monkeypatch.setattr(response, "astream_llm", fake_stream)
    monkeypatch.setattr(input_guard, "classify_with_model", classifier_unavailable)
    monkeypatch.setattr(admin, "execute_admin_tool", fake_execute)
    return admin_calls


async def post(api: FastAPI, path: str, token: str, json_body: dict) -> httpx.Response:
    async with httpx.AsyncClient(
        transport=ASGITransport(app=api), base_url="http://test"
    ) as client:
        return await client.post(path, json=json_body, headers={"Authorization": f"Bearer {token}"})


async def test_non_admin_resume_is_forbidden() -> None:
    token = auth_mod.create_access_token(VIEWER)
    response = await post(make_app(), "/chat/resume", token, {"thread_id": "t1", "approved": True})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


async def test_admin_resume_without_pending_interrupt_is_rejected() -> None:
    token = auth_mod.create_access_token(ADMIN)
    response = await post(
        make_app(), "/chat/resume", token, {"thread_id": "no-such-thread", "approved": True}
    )
    assert 400 <= response.status_code < 500
    # `detail` (the specific reason) goes to logs; clients see the typed code + safe message.
    assert response.json()["error"]["code"] == "invalid_request"


async def test_admin_chat_interrupts_then_resume_round_trip(monkeypatch) -> None:
    admin_calls = await admin_action_stubs(monkeypatch)
    api = make_app()
    token = auth_mod.create_access_token(ADMIN)

    # 1. The admin question pauses on the interrupt; the SSE announces it before `done`.
    first = await post(
        api, "/chat/stream", token, {"message": "reindex payments", "thread_id": "adm1"}
    )
    assert first.status_code == 200
    assert "event: approval" in first.text
    assert first.text.index("event: approval") < first.text.index("event: done")
    assert "reindex_department" in first.text
    assert admin_calls == []

    # 2. The resume decision executes the admin tool server-side and finishes the turn.
    second = await post(api, "/chat/resume", token, {"thread_id": "adm1", "approved": True})
    assert second.status_code == 200
    assert "event: done" in second.text
    assert not second.text.startswith("event: error")
    assert len(admin_calls) == 1
    assert admin_calls[0] == {
        "name": "reindex_department",
        "args": {"department": "payments"},
        "approved": True,
    }

    # 3. The thread is no longer interrupted: a second resume is now a 4xx.
    third = await post(api, "/chat/resume", token, {"thread_id": "adm1", "approved": True})
    assert 400 <= third.status_code < 500


async def test_resume_does_not_run_a_replanned_action(monkeypatch) -> None:
    admin_calls = await admin_action_stubs(monkeypatch)
    calls: list[int] = []

    async def drifting_plan(_role, _schema, _messages, **_kw):
        # First plan says payments (what the human approved); every re-plan says security.
        calls.append(1)
        department = "payments" if len(calls) == 1 else "security"
        return ToolPlan(
            calls=[ToolCallRequest(tool="reindex_department", args={"department": department})]
        )

    monkeypatch.setattr(tools_node_mod, "ainvoke_structured", drifting_plan)
    api = make_app()
    token = auth_mod.create_access_token(ADMIN)

    first = await post(
        api, "/chat/stream", token, {"message": "reindex payments", "thread_id": "drift"}
    )
    assert first.status_code == 200
    assert "event: approval" in first.text
    assert "payments" in first.text
    assert admin_calls == []

    second = await post(api, "/chat/resume", token, {"thread_id": "drift", "approved": True})
    assert second.status_code == 200
    assert admin_calls == [], "a re-planned action must never execute on a stale approval"
    assert "changed after approval" in second.text
    assert "event: done" in second.text
