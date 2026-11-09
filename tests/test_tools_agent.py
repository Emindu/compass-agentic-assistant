"""Tools agent wiring, with every LLM helper and MCP call stubbed (network-free).

Covers the RBAC re-binding, MCP containment and the human-approval interrupt round-trip:
the model only ever proposes calls, the graph re-checks them against the caller's role,
and admin actions pause on `interrupt` until a human resumes the thread.
"""

from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agents.graph import build_graph
from app.agents.nodes import response, retrieval, supervisor
from app.agents.nodes import tools as tools_node_mod
from app.agents.nodes.tools import ToolCallRequest, ToolPlan
from app.agents.state import RouteDecision
from app.api.routes.chat import _pending_interrupt
from app.retrieval.models import RetrievalResult
from app.security import input_guard
from app.security.rbac import Role, UserContext
from app.tools import admin
from app.tools.mcp_client import ToolUnavailableError

VIEWER = UserContext(user_id="vera", name="Vera Viewer", role=Role.viewer, department="retail")
ANALYST = UserContext(user_id="anil", name="Anil Analyst", role=Role.analyst, department="payments")
ADMIN = UserContext(user_id="ada", name="Ada Admin", role=Role.admin, department="platform")

DIRECTORY_Q = RouteDecision(
    intent="directory_lookup",
    agents=["tools"],
    search_query="who owns the payments runbook?",
    reasoning="a person lookup",
)
ADMIN_Q = RouteDecision(
    intent="admin_action",
    agents=["tools"],
    search_query="reindex payments",
    reasoning="destructive action",
)


def stub_llms(monkeypatch, *, answers: list[str]) -> list:
    """Patch all LLM touchpoints; returns the list the tools planner records its calls in."""
    planner_calls: list = []

    async def fake_route(_role, _schema, _messages, **_kw):
        return DIRECTORY_Q

    async def fake_search(_q):
        return RetrievalResult(chunks=[])

    async def fake_stream(_role, _messages, **_kw):
        for word in answers.pop(0).split(" "):
            yield AIMessageChunk(content=word + " ")

    async def classifier_unavailable(_text):
        return None  # the input guard fails open onto the (clean) pattern verdict

    monkeypatch.setattr(supervisor, "ainvoke_structured", fake_route)
    monkeypatch.setattr(retrieval, "hybrid_search", fake_search)
    monkeypatch.setattr(response, "astream_llm", fake_stream)
    monkeypatch.setattr(input_guard, "classify_with_model", classifier_unavailable)
    return planner_calls


def stub_plan(monkeypatch, planner_calls: list, plan: ToolPlan) -> None:
    async def fake_plan(_role, _schema, _messages, **_kw):
        planner_calls.append(_messages)
        return plan

    monkeypatch.setattr(tools_node_mod, "ainvoke_structured", fake_plan)


def stub_mcp(monkeypatch, calls: list[dict], *, error: Exception | None = None) -> None:
    async def fake_call_mcp(tool: str, args: dict, *, user):
        calls.append({"tool": tool, "args": args, "user": user})
        if error is not None:
            raise error
        return {"employee": "Priya Patel", "team": "payments"}

    monkeypatch.setattr(tools_node_mod, "call_mcp_tool", fake_call_mcp)


def stub_admin(monkeypatch, calls: list[dict]) -> None:
    async def fake_execute(name: str, args: dict, *, user, approved: bool) -> dict:
        calls.append({"name": name, "args": args, "approved": approved})
        return {"status": "reindexed", "department": args.get("department")}

    monkeypatch.setattr(admin, "execute_admin_tool", fake_execute)


async def run(graph, user: UserContext, text: str, thread: str = "t1") -> list[dict]:
    config = {"configurable": {"thread_id": thread}}
    inputs = {"messages": [HumanMessage(text)], "user": user}
    return [e async for e in graph.astream(inputs, config, stream_mode="custom")]


async def run_full(graph, user: UserContext, text: str, thread: str = "t1"):
    config = {"configurable": {"thread_id": thread}}
    events = await run(graph, user, text, thread)
    values = (await graph.aget_state(config)).values
    return values, events


async def resume_events(graph, config: dict, approved: bool) -> list[dict]:
    """Stream the graph with a resume Command, as POST /chat/resume does.

    Like the API, the decision is bound to the tool/args checkpointed in the pending
    interrupt: the node re-runs (and re-plans) on resume, so a bare approval must not
    authorize whatever the new plan contains.
    """
    pending = _pending_interrupt(await graph.aget_state(config)) or {}
    resume = {"approved": approved, "tool": pending.get("tool"), "args": pending.get("args")}
    return [e async for e in graph.astream(Command(resume=resume), config, stream_mode="custom")]


def tool_calls(events: list[dict]) -> list[dict]:
    return [e["data"] for e in events if e["type"] == "tool_call"]


async def test_analyst_directory_lookup_runs_tool_and_answers(monkeypatch):
    planner_calls = stub_llms(monkeypatch, answers=["Priya Patel is the payments runbook owner."])
    stub_plan(
        monkeypatch,
        planner_calls,
        ToolPlan(calls=[ToolCallRequest(tool="lookup_employee", args={"query": "Priya"})]),
    )
    mcp: list[dict] = []
    stub_mcp(monkeypatch, mcp)

    graph = build_graph(InMemorySaver())
    values, events = await run_full(graph, ANALYST, "who owns the payments runbook?")

    assert mcp == [{"tool": "lookup_employee", "args": {"query": "Priya"}, "user": ANALYST}], (
        "the tool agent must run exactly the planned call once"
    )
    assert planner_calls, "the tool planner LLM was invoked"
    statuses = tool_calls(events)
    assert [s["tool"] for s in statuses if s["status"] == "ok"] == ["lookup_employee"]
    assert values["tool_results"] == [
        {
            "tool": "lookup_employee",
            "args": {"query": "Priya"},
            "ok": True,
            "result": {"employee": "Priya Patel", "team": "payments"},
        }
    ]
    assert next(e for e in events if e["type"] == "answer")["data"]["text"].startswith(
        "Priya Patel"
    )
    assert values["errors"] == []


async def test_tool_unavailable_is_contained(monkeypatch):
    planner_calls = stub_llms(monkeypatch, answers=["The directory is unavailable right now."])
    stub_plan(
        monkeypatch,
        planner_calls,
        ToolPlan(calls=[ToolCallRequest(tool="lookup_employee", args={"query": "Priya"})]),
    )
    mcp: list[dict] = []
    stub_mcp(monkeypatch, mcp, error=ToolUnavailableError("MCP server did not answer"))

    graph = build_graph(InMemorySaver())
    values, events = await run_full(graph, ANALYST, "who owns the payments runbook?")

    # The failure is contained: no error event, an AgentError in state, and an answer exists.
    assert not any(e["type"] == "error" for e in events)
    assert [(e.node, e.code) for e in values["errors"]] == [("tools", "mcp_unavailable")]
    assert values["tool_results"][0]["ok"] is False
    assert "unavailable" in values["tool_results"][0]["error"]
    assert [d["status"] for d in tool_calls(events)] == ["running", "failed"]
    assert any(e["type"] == "answer" for e in events)
    assert values["messages"][-1].type == "ai"


async def test_viewer_directory_question_denied_routes_to_retrieval(monkeypatch):
    planner_calls = stub_llms(
        monkeypatch, answers=["I couldn't find information about that person."]
    )
    stub_plan(
        monkeypatch,
        planner_calls,
        ToolPlan(calls=[ToolCallRequest(tool="lookup_employee", args={"query": "Priya"})]),
    )
    mcp: list[dict] = []
    stub_mcp(monkeypatch, mcp)

    graph = build_graph(InMemorySaver())
    values, events = await run_full(graph, VIEWER, "who owns the payments runbook?")

    denied = [d for d in tool_calls(events) if d.get("status") == "denied"]
    assert denied and denied[0]["tool"] == "tools"
    assert "may not use directory tools" in denied[0]["reason"]
    # The planner never ran: the tools node was never reached.
    assert planner_calls == [] and mcp == []
    assert any(e["type"] == "retrieval" for e in events), "the viewer falls back to documents"
    assert not values.get("tool_results")
    assert values["route"].intent == "directory_lookup"


def admin_approval_setup(monkeypatch) -> tuple[list[dict], list[dict]]:
    planner_calls = stub_llms(monkeypatch, answers=["The department is being re-indexed."])
    stub_plan(
        monkeypatch,
        planner_calls,
        ToolPlan(
            calls=[
                ToolCallRequest(
                    tool="reindex_department", args={"department": "payments"}, why="requested"
                )
            ]
        ),
    )
    mcp: list[dict] = []
    admin_calls: list[dict] = []
    stub_admin(monkeypatch, admin_calls)
    return mcp, admin_calls


def pending_interrupt(snapshot) -> dict | None:
    """First interrupt payload on a state snapshot, tolerant of langgraph revisions."""
    interrupts = list(getattr(snapshot, "interrupts", None) or ())
    for task in getattr(snapshot, "tasks", None) or ():
        interrupts.extend(getattr(task, "interrupts", None) or ())
    for intr in interrupts:
        value = intr.get("value", intr) if isinstance(intr, dict) else getattr(intr, "value", None)
        if isinstance(value, dict):
            return value
    return None


async def test_admin_action_interrupts_until_approved(monkeypatch):
    mcp, admin_calls = admin_approval_setup(monkeypatch)
    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "approve1"}}
    inputs = {"messages": [HumanMessage("reindex payments")], "user": ADMIN}

    # First run pauses at the interrupt: a pending_approval event, no ok, no execution yet.
    events = [e async for e in graph.astream(inputs, config, stream_mode="custom")]
    statuses = [d["status"] for d in tool_calls(events) if d["tool"] == "reindex_department"]
    assert "pending_approval" in statuses and "ok" not in statuses
    assert not any(e["type"] == "answer" for e in events)
    assert admin_calls == []

    snapshot = await graph.aget_state(config)
    assert snapshot.next, "the graph must be paused with a pending task"
    payload = pending_interrupt(snapshot)
    assert payload is not None and payload["tool"] == "reindex_department"
    assert payload["args"] == {"department": "payments"}

    # Resume with the approval: the admin tool runs exactly once and the turn completes.
    events = await resume_events(graph, config, approved=True)
    assert events, "the resumed stream must still produce activity events"
    resumed = [d for d in tool_calls(events) if d["tool"] == "reindex_department"]
    assert resumed[-1]["status"] == "ok"
    assert len(admin_calls) == 1
    assert admin_calls[0] == {
        "name": "reindex_department",
        "args": {"department": "payments"},
        "approved": True,
    }
    values = (await graph.aget_state(config)).values
    assert values["tool_results"][0]["ok"] is True
    assert values["messages"][-1].type == "ai"
    assert not (await graph.aget_state(config)).next


async def test_admin_action_rejected_never_executes(monkeypatch):
    mcp, admin_calls = admin_approval_setup(monkeypatch)
    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "reject1"}}
    inputs = {"messages": [HumanMessage("reindex payments")], "user": ADMIN}

    events = [e async for e in graph.astream(inputs, config, stream_mode="custom")]
    assert "pending_approval" in [d["status"] for d in tool_calls(events)]

    events = await resume_events(graph, config, approved=False)
    rejected = [d for d in tool_calls(events) if d["tool"] == "reindex_department"]
    assert rejected[-1]["status"] == "rejected"
    assert admin_calls == [], "a rejected action must never execute"
    values = (await graph.aget_state(config)).values
    assert values["tool_results"][0]["ok"] is False
    assert "Rejected" in values["tool_results"][0]["error"]
    assert any(e["type"] == "answer" for e in events), "rejection still ends with an answer"


async def test_unbound_approval_does_not_execute(monkeypatch):
    mcp, admin_calls = admin_approval_setup(monkeypatch)
    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "unbound1"}}
    inputs = {"messages": [HumanMessage("reindex payments")], "user": ADMIN}

    events = [e async for e in graph.astream(inputs, config, stream_mode="custom")]
    assert "pending_approval" in [d["status"] for d in tool_calls(events)]

    # A bare approval carries no tool/args, so it must not bind to (and run) the plan.
    events = [
        e
        async for e in graph.astream(
            Command(resume={"approved": True}), config, stream_mode="custom"
        )
    ]
    assert admin_calls == [], "an unbound approval must never execute the admin tool"
    assert any(d.get("status") == "failed" for d in tool_calls(events))
    values = (await graph.aget_state(config)).values
    assert [(e.node, e.code) for e in values["errors"]] == [("tools", "approval_mismatch")]


async def test_tool_outside_role_in_plan_is_skipped(monkeypatch):
    planner_calls = stub_llms(monkeypatch, answers=["Priya Patel is the runbook owner."])
    stub_plan(
        monkeypatch,
        planner_calls,
        ToolPlan(
            calls=[
                ToolCallRequest(tool="lookup_employee", args={"query": "Priya"}),
                ToolCallRequest(
                    tool="reindex_department", args={"department": "payments"}, why="hijack"
                ),
            ]
        ),
    )
    mcp: list[dict] = []
    admin_calls: list[dict] = []
    stub_admin(monkeypatch, admin_calls)
    stub_mcp(monkeypatch, mcp)

    graph = build_graph(InMemorySaver())
    values, events = await run_full(graph, ANALYST, "who owns the payments runbook?")

    # The planner proposed an admin tool; the execution-time role check skipped it.
    denied = [d for d in tool_calls(events) if d["status"] == "denied"]
    assert [d["tool"] for d in denied] == ["reindex_department"]
    assert "not available to role" in denied[0]["reason"]
    assert [r["tool"] for r in values["tool_results"]] == ["lookup_employee"]
    assert [c["tool"] for c in mcp] == ["lookup_employee"]
    assert admin_calls == []
    assert any(e["type"] == "answer" for e in events)
