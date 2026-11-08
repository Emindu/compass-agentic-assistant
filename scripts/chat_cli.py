"""Terminal client for the chat API: logs in, streams one or more turns, prints every event.

Run (API must be running on :8000):
  uv run python -m scripts.chat_cli --user vera "How do I rotate the card switch certificate?" \
      "What did the older version of that runbook say?"
"""

import argparse
import json

import httpx
from httpx_sse import connect_sse

DEMO_PASSWORDS = {"vera": "viewer-demo", "anil": "analyst-demo", "ada": "admin-demo"}


def show(event: str, payload: dict) -> None:
    node, data = payload.get("node"), payload.get("data", {})
    if event == "token":
        print(data["text"], end="", flush=True)
    elif event == "node_start":
        print(f"\n  ▶ {node}", flush=True)
    elif event == "node_end":
        print(f"\n  ■ {node} {data.get('status')} {data.get('duration_ms')} ms", flush=True)
    elif event == "retrieval" and data.get("hits") is not None:
        print(f"  retrieval {data['status']} notes={data.get('notes')}")
        for h in data["hits"]:
            print(f"     {h['fused']:.3f} d={h['dense']} s={h['sparse']} {h['citation']}")
    elif event == "answer":
        cites = [c["id"] for c in data.get("citations", [])]
        print(f"\n  ✔ answer citations={cites} errors={data.get('errors')}")
    else:
        print(f"  {event}: {json.dumps(data)[:300]}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("messages", nargs="+")
    p.add_argument("--user", default="vera", choices=DEMO_PASSWORDS)
    p.add_argument("--base", default="http://localhost:8000")
    p.add_argument("--thread", default=None)
    args = p.parse_args()

    with httpx.Client(base_url=args.base, timeout=180) as client:
        login = client.post(
            "/auth/login", json={"username": args.user, "password": DEMO_PASSWORDS[args.user]}
        )
        login.raise_for_status()
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        thread = args.thread
        for message in args.messages:
            print(f"\n=== {args.user}: {message}")
            body = {"message": message, "thread_id": thread}
            with connect_sse(client, "POST", "/chat/stream", json=body, headers=headers) as source:
                if source.response.status_code != 200:
                    print(source.response.status_code, source.response.read().decode())
                    return
                for sse in source.iter_sse():
                    payload = json.loads(sse.data)
                    if sse.event == "done":
                        thread = payload["data"]["thread_id"]
                        print(f"\n  done thread={thread}")
                    else:
                        show(sse.event, payload)


if __name__ == "__main__":
    main()
