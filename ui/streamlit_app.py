"""Streamlit entry point: chat plus a live agent-activity panel.

Deliberately thin — it only wires session state and layout; all real logic lives in
`ui.client` (transport) and `ui.activity` (event formatting), which stay testable.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1]))

import streamlit as st

from ui import activity
from ui.activity import Step, TurnState
from ui.client import DEMO_USERS, ApiError, CompassClient

RENDER_INTERVAL_S = 0.15
ACTIVITY_HEIGHT_PX = 640
LIVE_EVENTS = {"node_start", "node_end", "answer", "done"}


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "messages": [],  # chat history: {"role", "content", "citations", "notes", "errors"}
        "turns": [],  # completed TurnState objects, one per assistant answer
        "thread_id": None,
        "base_url": None,
        "client": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_client(base_url: str) -> CompassClient:
    """Reuse the client across reruns, but rebuild it when the API URL changes."""
    client = st.session_state.get("client")
    if not isinstance(client, CompassClient) or st.session_state.base_url != base_url:
        client = CompassClient(base_url)
        st.session_state.client = client
        st.session_state.base_url = base_url
    return client


def _reset_conversation() -> None:
    """Drop the thread so the next question starts a fresh, memory-free conversation."""
    st.session_state.thread_id = None
    st.session_state.messages = []
    st.session_state.turns = []


def _render_sidebar() -> CompassClient:
    base_url = st.sidebar.text_input(
        "API URL", value=os.environ.get("COMPASS_API_URL", "http://localhost:8000")
    ).strip()
    client = _get_client(base_url)
    st.sidebar.markdown("🧭 **Compass**")

    if client.health():
        st.sidebar.caption("🟢 API online")
    else:
        st.sidebar.caption("🔴 API offline")

    if client.user:
        user = client.user
        st.sidebar.markdown(f"**{user.get('name', 'user')}**")
        badge = activity.format_status_badge(str(user.get("role", "")))
        st.sidebar.caption(f"{badge} · {user.get('department', '-')}")
        if st.sidebar.button("New conversation"):
            _reset_conversation()
            st.rerun()
        if st.sidebar.button("Sign out"):
            client.logout()
            _reset_conversation()
            st.rerun()
    else:
        username = st.sidebar.selectbox(
            "Demo user", options=list(DEMO_USERS), format_func=lambda u: DEMO_USERS[u][1]
        )
        if st.sidebar.button("Sign in"):
            password, _label = DEMO_USERS[username]
            try:
                client.login(username, password)
                st.rerun()
            except ApiError as exc:
                st.sidebar.error(exc.message)

    if st.session_state.thread_id:
        st.sidebar.caption(f"Thread: {st.session_state.thread_id}")
    return client


def _render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] != "assistant":
            return
        if message.get("citations"):
            with st.expander("Sources"):
                st.markdown(activity.render_citations(message["citations"]))
        for note in message.get("notes", []):
            st.caption(f"ℹ️ {note}")
        for error in message.get("errors", []):
            st.warning(f"{error.get('node') or 'error'}: {error.get('message', '')}")


def _render_steps(steps: list[Step]) -> None:
    for step in steps:
        # Titles from ui.activity usually start with their icon already; don't print it twice.
        title = step.title if step.title.startswith(step.icon) else f"{step.icon} {step.title}"
        st.markdown(title)
        if step.detail:
            st.caption(step.detail)
        if step.code:
            with st.expander("Plan code"):
                st.code(step.code, language="python")
        if step.table:
            st.dataframe(step.table, hide_index=True)


def _show_api_error(exc: ApiError, base_url: str) -> None:
    if exc.code == "unreachable":
        st.error(f"Cannot reach the Compass API at {base_url}. Is the server running?")
    elif exc.status == 429:
        wait = exc.retry_after if exc.retry_after is not None else 30.0
        st.warning(f"Rate limit reached — retry in {wait:.0f} s")
    elif exc.status == 401 or exc.code == "not_authenticated":
        st.warning("Your session ended — please sign in again.")
    else:
        st.error(exc.message or "The request failed.")


def _run_turn(
    client: CompassClient,
    prompt: str,
    base_url: str,
    answer_col: Any,
    live_panel: Any,
) -> None:
    """Stream one turn, live-updating both the chat and the activity panel."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    answer_col.chat_message("user").markdown(prompt)

    # st.empty() holds ONE element, so the panel is re-rendered through .container() each time.
    with answer_col.chat_message("assistant"):
        answer_box = st.empty()

    state = TurnState()
    last_render = 0.0
    try:
        for event in client.stream_chat(prompt, st.session_state.thread_id):
            state = activity.apply(state, event)
            cursor = "" if state.done or event.type == "answer" else " ▌"
            answer_box.markdown(state.answer_text + cursor)
            now = time.monotonic()
            if event.type in LIVE_EVENTS or now - last_render >= RENDER_INTERVAL_S:
                with live_panel.container():
                    _render_steps(state.steps)
                last_render = now
    except ApiError as exc:
        _show_api_error(exc, base_url)
    finally:
        if state.steps:
            answer_box.markdown(state.answer_text or "*(no answer)*")
            with live_panel.container():
                _render_steps(state.steps)
            st.session_state.turns.append(state)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": state.answer_text,
                    "citations": state.citations,
                    "notes": state.notes,
                    "errors": state.errors,
                }
            )
            if state.thread_id:
                st.session_state.thread_id = state.thread_id
            # Resync layout (past-turn expanders, thread caption) now that state is stored.
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Compass", layout="wide")
    _init_state()
    client = _render_sidebar()

    chat_col, activity_col = st.columns([3, 2])

    with chat_col:
        st.subheader("Chat")
        for message in st.session_state.messages:
            _render_message(message)

    with activity_col:
        st.subheader("Agent activity")
        # Fixed height + own scrollbar: a long RLM trace must not push the chat off screen.
        panel_box = st.container(height=ACTIVITY_HEIGHT_PX)
        turns = st.session_state.turns
        with panel_box:
            for index, turn in enumerate(turns[:-1], start=1):
                with st.expander(f"Turn {index}"):
                    _render_steps(turn.steps)
        live_panel = panel_box.empty()
        if turns:  # keep the last completed turn visible between questions
            with live_panel.container():
                _render_steps(turns[-1].steps)

    prompt = st.chat_input("Ask Compass…")
    if prompt:
        if not client.user:
            st.warning("Please sign in first")
        else:
            _run_turn(
                client,
                prompt,
                st.session_state.base_url or "http://localhost:8000",
                chat_col,
                live_panel,
            )


main()
