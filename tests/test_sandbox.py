"""Tests for the plan sandbox: real child-process runs, no network required.

Each `run_plan` call spawns a genuine `python -I` child, so these cover the
actual protocol, env, and isolation plumbing rather than mocks of them.
"""

import asyncio
import textwrap
import time

import pytest

from app.rlm import sandbox
from app.rlm.sandbox import PlanRejected, validate_plan


class FakeHandler:
    """Records (name, kwargs) per call; optionally fails on a given primitive."""

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail_on = fail_on

    async def __call__(self, name: str, kwargs: dict) -> object:
        self.calls.append((name, dict(kwargs)))
        if name == self.fail_on:
            raise ValueError(f"primitive {name} failed")
        return [{"doc_id": "A", "n": kwargs.get("x", 0)}]


COUNTER_PLAN = textwrap.dedent(
    """    import collections
    hits = search(query="compass")
    counts = collections.Counter(hit["doc_id"] for hit in hits)
    best = None
    for doc_id, n in counts.most_common():
        best = doc_id
        break
    result = {"total": len(hits), "by_doc": dict(counts), "best": best}
    """
)


async def test_plan_runs_and_returns_result():
    handler = FakeHandler()
    res = await sandbox.run_plan(COUNTER_PLAN, handler, ["search"])

    assert res.ok is True, res.error
    assert res.result == {"total": 1, "by_doc": {"A": 1}, "best": "A"}
    assert handler.calls == [("search", {"query": "compass"})]
    assert len(res.calls) == 1
    assert res.calls[0].name == "search"
    assert res.calls[0].kwargs == {"query": "compass"}
    assert res.calls[0].ok is True
    assert res.calls[0].error is None
    assert res.duration_ms > 0


async def test_print_captured_not_mixed_into_protocol():
    handler = FakeHandler()
    res = await sandbox.run_plan('print("hello sandbox")\nresult = "done"', handler, ["search"])

    assert res.ok is True, res.error
    assert res.stdout == "hello sandbox\n"
    assert res.result == "done"


async def test_stdout_is_truncated():
    plan = 'print("x" * 10_000)\nresult = 1'
    res = await sandbox.run_plan(plan, FakeHandler(), [])

    assert res.ok is True, res.error
    assert len(res.stdout) == 4000


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "import subprocess",
        "from os import path",
        "().__class__.__bases__",
        'open("/etc/passwd")',
        'eval("1")',
        'getattr(x, "y")',
        "def f():\n    global counter\n",
        'result = __import__("os")\n',
        "x = vars()\n",
    ],
)
def test_validate_plan_rejects(code: str):
    with pytest.raises(PlanRejected):
        validate_plan(code)


def test_validate_plan_rejects_overlong_code():
    with pytest.raises(PlanRejected, match="characters"):
        validate_plan("#" * 8001)


def test_validate_plan_accepts_normal_plan():
    ok_plan = textwrap.dedent(
        """        import re
        words = [w for w in re.findall(r"\\w+", "compass rocks") if w]
        counts = {w: len(w) for w in words}
        label = f"found={len(counts)}"
        try:
            total = sum(len(word) for word in words)
        except ValueError:
            total = -1
        result = {"counts": counts, "label": label, "total": total}
        """
    )
    assert validate_plan(ok_plan) is None


async def test_run_plan_validates_before_spawning():
    handler = FakeHandler()
    res = await sandbox.run_plan("import os\nresult = os.getpid()", handler, ["search"])

    assert res.ok is False
    assert "import" in (res.error or "")
    assert "os" in (res.error or "")
    assert handler.calls == []
    assert res.calls == []


async def test_infinite_loop_times_out():
    handler = FakeHandler()
    started = time.monotonic()
    res = await sandbox.run_plan("while True:\n    pass\n", handler, ["search"], timeout_s=2.0)
    elapsed = time.monotonic() - started

    assert res.ok is False
    assert "timed out" in (res.error or "")
    assert elapsed < 5.0
    assert handler.calls == []


async def test_unknown_primitive_errors_inside_plan():
    handler = FakeHandler()
    plan = textwrap.dedent(
        """        try:
            missing_primitive(query="x")
            result = "not-caught"
        except Exception as exc:
            result = "caught:" + str(exc)
        """
    )
    res = await sandbox.run_plan(plan, handler, ["search"])

    assert res.ok is True, res.error
    assert isinstance(res.result, str)
    assert res.result.startswith("caught:")
    assert "missing_primitive" in res.result
    assert handler.calls == []


async def test_call_limit_is_enforced():
    handler = FakeHandler()
    plan = textwrap.dedent(
        """        seen = []
        for i in range(5):
            seen.append(search(query="q"))
        result = len(seen)
        """
    )
    res = await sandbox.run_plan(plan, handler, ["search"], max_calls=2)

    assert res.ok is False
    assert "limit" in (res.error or "")
    assert len(handler.calls) == 2
    assert len(res.calls) == 3
    assert res.calls[-1].ok is False


def test_child_env_has_no_parent_vars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SECRET_TEST_VAR", "leak")

    env = sandbox._child_env()

    assert "SECRET_TEST_VAR" not in env
    assert env == {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"}


async def test_spawned_child_env_has_no_parent_vars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SECRET_TEST_VAR", "leak")
    captured: dict[str, str] = {}
    real_exec = asyncio.create_subprocess_exec

    async def spy_exec(*args: object, **kwargs: object) -> object:
        captured.update(kwargs.get("env") or {})
        return await real_exec(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy_exec)
    res = await sandbox.run_plan("result = 1", FakeHandler(), [])

    assert res.ok is True, res.error
    assert "SECRET_TEST_VAR" not in captured
    assert captured == {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"}


async def test_handler_exception_is_catchable_in_plan():
    handler = FakeHandler(fail_on="search")
    plan = textwrap.dedent(
        """        try:
            search(query="x")
            result = "not-caught"
        except Exception as exc:
            result = "caught:" + str(exc)
        """
    )
    res = await sandbox.run_plan(plan, handler, ["search"])

    assert res.ok is True, res.error
    assert "primitive search failed" in str(res.result)
    assert len(res.calls) == 1
    assert res.calls[0].ok is False
    assert "primitive search failed" in (res.calls[0].error or "")


async def test_uncaught_handler_error_fails_the_result():
    handler = FakeHandler(fail_on="search")
    res = await sandbox.run_plan('result = search(query="x")', handler, ["search"])

    assert res.ok is False
    # The child stub raises RuntimeError, so the class name appears in done/error.
    assert "RuntimeError" in (res.error or "")
    assert "primitive search failed" in (res.error or "")


@pytest.mark.parametrize(
    "code",
    [
        "g = (x for x in [1])\nf = g.gi_frame.f_back\nresult = f.f_globals",
        "result = '{0.gi_frame}'.format((x for x in []))",
        "result = list_documents._name",
    ],
)
def test_frame_and_format_walks_are_rejected(code):
    with pytest.raises(PlanRejected):
        validate_plan(code)


async def test_large_result_crosses_the_pipe():
    async def handler(_name, _kwargs):
        return None

    out = await sandbox.run_plan("result = 'x' * 150000", handler, [])
    assert out.ok, out.error
    assert len(out.result) == 150000
