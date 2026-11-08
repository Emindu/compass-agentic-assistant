"""Sandboxed execution of LLM-written "search plans".

Why a subprocess: the RLM planner emits arbitrary Python, and Python-level
restriction alone is bypassable (object traversal via `__subclasses__`,
smuggled `__import__`, C-level escapes), so a plan runs in a disposable child
interpreter that the parent can kill outright. Defence in depth stacks further
layers on top of that process boundary:

1. a static AST allow-list (`validate_plan`) so the worst untrusted code never
   even spawns a child (length cap, import allow-list, dunder-name rejection,
   banned callables, no global/nonlocal);
2. restricted builtins inside the child, with `__import__` re-checking the
   allow-list at runtime — checked twice, on purpose, in case one layer is
   bypassed;
3. a minimal child environment: no API keys or other parent env vars leak in,
   and RLIMIT_CPU/AS/NOFILE/FSIZE bound runaways at the kernel level;
4. `unshare -rn` (when the kernel allows it) opens a user+network namespace so
   the child has no network stack at all.

Access control deliberately does NOT live here: primitives run in the parent
process with the calling user's clearance, so sandboxed code can never widen
access beyond what its user already has. This module is the cheap in-process
fallback; production hardening would place the child in gVisor / Firecracker /
E2B instead.
"""

import ast
import asyncio
import functools
import json
import os
import resource
import subprocess
import sys
import tempfile
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any

from pydantic import BaseModel

from app.core.logging import get_logger

logger = get_logger(__name__)

ALLOWED_IMPORTS = frozenset(
    {"json", "re", "math", "statistics", "collections", "itertools", "datetime", "functools"}
)

# Banned callables: execution/introspection builtins can defeat a Python-level
# sandbox or reach data the plan should not see, so none may be called at all.
_BANNED_CALLS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
        "memoryview",
        "help",
    }
)

# Frame, code and traceback attributes (gen.gi_frame.f_back.f_globals, ...) walk back to the
# child bootstrap's globals, where the real __import__ and sys live, without any dunder.
_BANNED_ATTR_PREFIXES = ("_", "gi_", "cr_", "ag_", "f_", "tb_", "co_")
_BANNED_ATTRS = frozenset({"format", "format_map", "mro"})

_PLAN_MAX_CHARS = 8000
_STDOUT_MAX_CHARS = 4000
_CHILD_CPU_SECONDS = 20
_CHILD_AS_BYTES = 512 * 1024 * 1024
_CHILD_NOFILE = 32
_CHILD_FSIZE_BYTES = 1024 * 1024

PrimitiveHandler = Callable[[str, dict[str, Any]], Awaitable[Any]]


class PlanRejected(ValueError):
    """Raised by `validate_plan` when code violates the sandbox policy.

    A ValueError subclass so API layers can map it to a 400 without inspecting
    the reason string.
    """


class PrimitiveCall(BaseModel):
    """One brokered primitive call: what ran, with which kwargs, and whether it failed."""

    name: str
    kwargs: dict[str, Any]
    duration_ms: float
    ok: bool
    error: str | None = None


class SandboxResult(BaseModel):
    """Outcome of one sandboxed plan run.

    `error` and `stdout` surface to the planner/LLM, so they carry short
    messages only — never parent-side tracebacks or environment details.
    """

    ok: bool
    result: Any = None
    stdout: str = ""
    error: str | None = None
    calls: list[PrimitiveCall] = []
    duration_ms: float = 0.0


def _leading_package(module: str) -> str:
    return module.split(".", 1)[0]


def validate_plan(code: str) -> None:
    """Statically vet planner code before it ever reaches a child process.

    Why AST-level allow-listing: it is deterministic, auditable and costs no
    subprocess — the worst untrusted code is rejected before it can run at all.
    Raises `PlanRejected` with a concrete reason for every policy violation.
    """
    if len(code) > _PLAN_MAX_CHARS:
        raise PlanRejected(f"plan is longer than {_PLAN_MAX_CHARS} characters")
    try:
        tree = ast.parse(code, filename="<plan>")
    except SyntaxError as exc:
        raise PlanRejected(f"plan does not parse: {exc.msg} (line {exc.lineno})") from exc
    except ValueError as exc:
        # ast.parse raises plain ValueError for some inputs (e.g. null bytes).
        raise PlanRejected(f"plan does not parse: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                package = _leading_package(alias.name)
                if package not in ALLOWED_IMPORTS:
                    raise PlanRejected(f"import of module {package!r} is not allowed in plans")
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            if module is None or _leading_package(module) not in ALLOWED_IMPORTS:
                reason = (
                    "relative imports are not allowed in plans"
                    if module is None
                    else (f"import from module {module!r} is not allowed in plans")
                )
                raise PlanRejected(reason)
        elif isinstance(node, ast.Name):
            # Blocks the live-object escape hatches: __class__, __subclasses__,
            # __globals__, __builtins__, __import__ and friends.
            if node.id.startswith("__"):
                raise PlanRejected(f"dunder name {node.id!r} is not allowed in plans")
        elif isinstance(node, ast.Attribute):
            # Private, frame/code/traceback and str.format attributes are all object-graph
            # walks a search plan never needs.
            if node.attr.startswith(_BANNED_ATTR_PREFIXES) or node.attr in _BANNED_ATTRS:
                raise PlanRejected(f"attribute {node.attr!r} is not allowed in plans")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise PlanRejected("global/nonlocal statements are not allowed in plans")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BANNED_CALLS:
                raise PlanRejected(f"call to {func.id!r} is not allowed in plans")
            elif isinstance(func, ast.Attribute) and func.attr in _BANNED_CALLS:
                raise PlanRejected(f"call to {func.attr!r} is not allowed in plans")


def _child_env() -> dict[str, str]:
    """Env for the child process, deliberately NOT derived from os.environ.

    The parent may hold API keys and tokens; building the dict literal from
    scratch means no future change to os.environ starts leaking into children
    by accident.
    """
    return {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"}


@functools.lru_cache(maxsize=1)
def network_isolation_available() -> bool:
    """One-time probe: can this kernel give a child a user+network namespace?

    Why probe instead of trusting os.path.exists alone: many hosts disable
    unprivileged user namespaces, and a silent failure here would leave children
    WITH network while everyone believes otherwise. Cached so each process pays
    the probe cost once.
    """
    if not os.path.exists("/usr/bin/unshare"):
        return False
    try:
        probe = subprocess.run(
            ["unshare", "-rn", "true"], capture_output=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


@functools.lru_cache(maxsize=1)
def _warn_no_network_isolation() -> None:
    """Called once per process so operators see the degraded mode without log spam."""
    logger.warning("sandbox_no_network_isolation")


def _apply_rlimits() -> None:
    """Run in the freshly forked child, before exec: bound the blast radius.

    CPU stops tight loops even if the parent's event loop is wedged, address
    space stops memory bombs, NOFILE stops fd exhaustion, and FSIZE caps any
    file the child tries to create. Limits live in the kernel, so they hold even
    if the plan corrupts interpreter state.
    """
    resource.setrlimit(resource.RLIMIT_CPU, (_CHILD_CPU_SECONDS, _CHILD_CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (_CHILD_AS_BYTES, _CHILD_AS_BYTES))
    resource.setrlimit(resource.RLIMIT_NOFILE, (_CHILD_NOFILE, _CHILD_NOFILE))
    resource.setrlimit(resource.RLIMIT_FSIZE, (_CHILD_FSIZE_BYTES, _CHILD_FSIZE_BYTES))


# The child bootstrap, executed as `sys.executable -I -c`. Kept as a plain
# string constant (not a shared module) so the child cannot import app code or
# anything beside the Python standard library.
CHILD_SOURCE = """\\
import io
import json
import sys

# The parent pipe: protocol lines only, never plan output.
_protocol_out = sys.__stdout__
_real_import = __import__


def _send(message):
    try:
        line = json.dumps(message, default=str)
    except (TypeError, ValueError):
        line = json.dumps({
            "type": "done",
            "ok": False,
            "error": "sandbox result was not serialisable",
            "stdout": "",
        })
    _protocol_out.write(line + "\\n")
    _protocol_out.flush()


def _read():
    line = sys.stdin.readline()
    if not line:
        raise RuntimeError("sandbox parent closed stdin")
    return json.loads(line)


init = _read()
_allowed_modules = set(init["allowed_imports"])


def _safe_import(name, *args, **kwargs):
    # Runtime re-check of the same allow-list validate_plan enforces statically:
    # belt and braces, in case one layer is bypassed.
    if name.split(".")[0] not in _allowed_modules:
        raise ImportError("module {!r} is not allowed in plans".format(name))
    return _real_import(name, *args, **kwargs)


_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float,
    "frozenset": frozenset, "int": int, "isinstance": isinstance, "len": len,
    "list": list, "map": map, "max": max, "min": min, "print": print,
    "range": range, "reversed": reversed, "round": round, "set": set,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "Exception": Exception, "ValueError": ValueError, "KeyError": KeyError,
    "TypeError": TypeError, "__import__": _safe_import,
}

_plan_globals = {"__builtins__": _SAFE_BUILTINS}
for _primitive in init["primitives"]:
    def _stub(*args, _name=_primitive, **kwargs):
        _send({"type": "call", "name": _name, "args": list(args), "kwargs": kwargs})
        reply = _read()
        if reply.get("ok"):
            return reply["value"]
        raise RuntimeError(reply.get("error") or "primitive call failed")
    _plan_globals[_primitive] = _stub

# Plan print() lands in this buffer; sys.__stdout__ stays protocol-only.
_stdout_buffer = io.StringIO()
_previous_stdout = sys.stdout
sys.stdout = _stdout_buffer
try:
    exec(compile(init["code"], "<plan>", "exec"), _plan_globals)
    _done = {
        "type": "done",
        "ok": True,
        "result": _plan_globals.get("result"),
        "stdout": _stdout_buffer.getvalue()[:4000],
    }
except BaseException as exc:
    _done = {
        "type": "done",
        "ok": False,
        "error": "{}: {}".format(type(exc).__name__, exc),
        "stdout": _stdout_buffer.getvalue()[:4000],
    }
finally:
    sys.stdout = _previous_stdout
_send(_done)
"""


async def _handle_call(
    message: dict[str, Any],
    *,
    allowed: set[str],
    handler: PrimitiveHandler,
    calls: list[PrimitiveCall],
    max_calls: int,
) -> dict[str, Any]:
    """Broker one primitive RPC. Runs in the parent, where access control lives."""
    name = str(message.get("name", ""))
    args = list(message.get("args") or [])
    kwargs = dict(message.get("kwargs") or {})
    # Plans are keyword-first; positional args travel under the reserved
    # "_args" key so handlers decide how to bind them.
    payload = {"_args": args, **kwargs} if args else kwargs

    started = perf_counter()
    error: str | None = None
    value: Any = None
    if name not in allowed:
        # Defence in depth: the child only has stubs for allowed names, so this
        # only fires if the protocol is tampered with.
        error = f"unknown primitive {name!r}"
    elif len(calls) >= max_calls:
        error = f"primitive call limit of {max_calls} exceeded"
    else:
        try:
            value = await handler(name, payload)
        except Exception as exc:
            # A failing primitive is the plan's problem, not the parent's:
            # report it down the pipe instead of tearing down the sandbox.
            error = str(exc) or type(exc).__name__

    calls.append(
        PrimitiveCall(
            name=name,
            kwargs=payload,
            duration_ms=(perf_counter() - started) * 1000,
            ok=error is None,
            error=error,
        )
    )
    return {"ok": False, "error": error} if error is not None else {"ok": True, "value": value}


def _finish(done: dict[str, Any], max_result_bytes: int) -> SandboxResult:
    """Turn the child's done-line into a SandboxResult, enforcing the size cap."""
    stdout = str(done.get("stdout") or "")[:_STDOUT_MAX_CHARS]
    if not done.get("ok"):
        return SandboxResult(ok=False, error=str(done.get("error") or "plan failed"), stdout=stdout)
    size = len(json.dumps(done.get("result"), default=str))
    if size > max_result_bytes:
        return SandboxResult(
            ok=False, error=f"plan result is too large ({size} bytes)", stdout=stdout
        )
    return SandboxResult(ok=True, result=done.get("result"), stdout=stdout)


async def _child_died(proc: asyncio.subprocess.Process) -> SandboxResult:
    """The child exited without a done line: report the last stderr chars."""
    await proc.wait()
    stderr_bytes = b""
    if proc.stderr is not None:
        stderr_bytes = await proc.stderr.read()
    tail = stderr_bytes.decode("utf-8", errors="replace")[-500:]
    return SandboxResult(ok=False, error=f"sandbox child exited: {tail}".rstrip())


async def _drive(
    proc: asyncio.subprocess.Process,
    *,
    code: str,
    primitives: list[str],
    handler: PrimitiveHandler,
    calls: list[PrimitiveCall],
    max_calls: int,
    max_result_bytes: int,
) -> SandboxResult:
    """Serve the child's stdin/stdout protocol until done or EOF."""
    assert proc.stdin is not None and proc.stdout is not None  # always spawned with pipes
    init = {
        "code": code,
        "primitives": sorted(set(primitives)),
        "allowed_imports": sorted(ALLOWED_IMPORTS),
    }
    try:
        proc.stdin.write((json.dumps(init) + "\n").encode())
        await proc.stdin.drain()
        allowed = set(primitives)
        while True:
            line = await proc.stdout.readline()
            if not line:
                return await _child_died(proc)
            try:
                message = json.loads(line)
            except ValueError:
                return SandboxResult(ok=False, error="sandbox sent a malformed protocol line")
            kind = message.get("type")
            if kind == "call":
                reply = await _handle_call(
                    message,
                    allowed=allowed,
                    handler=handler,
                    calls=calls,
                    max_calls=max_calls,
                )
                try:
                    payload = (json.dumps(reply, default=str) + "\n").encode()
                except (TypeError, ValueError):
                    payload = (
                        json.dumps(
                            {"ok": False, "error": "primitive returned a non-serialisable value"}
                        )
                        + "\n"
                    ).encode()
                proc.stdin.write(payload)
                await proc.stdin.drain()
            elif kind == "done":
                return _finish(message, max_result_bytes)
            else:
                return SandboxResult(ok=False, error=f"sandbox sent unknown message {kind!r}")
    except (BrokenPipeError, ConnectionResetError):
        return await _child_died(proc)


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    """Kill and reap the child unconditionally; an unreaped sandbox process lingers."""
    try:
        if proc.returncode is None:
            proc.kill()
    except ProcessLookupError:
        pass  # died and was reaped between the check and the signal
    if proc.stdin is not None:
        proc.stdin.close()
    # Reader pipes have no close(); their transports close on EOF when the child dies.
    await proc.wait()


def _finalize(outcome: SandboxResult, calls: list[PrimitiveCall], started: float) -> SandboxResult:
    outcome.calls = calls
    outcome.duration_ms = (perf_counter() - started) * 1000
    logger.info(
        "sandbox_run",
        ok=outcome.ok,
        calls=len(calls),
        duration_ms=round(outcome.duration_ms, 3),
        error=outcome.error,
    )
    return outcome


async def run_plan(
    code: str,
    handler: PrimitiveHandler,
    primitives: list[str],
    *,
    timeout_s: float = 30.0,
    max_calls: int = 40,
    max_result_bytes: int = 200_000,
) -> SandboxResult:
    """Validate and execute one plan in an isolated child process.

    The plan calls `primitives` which are brokered to `handler` in this
    process. Every failure mode (reject, crash, timeout, oversize result)
    yields `ok=False` with a short message instead of raising or leaking
    parent-side tracebacks, so callers can feed the failure straight back to
    the planner.
    """
    started = perf_counter()
    calls: list[PrimitiveCall] = []
    try:
        validate_plan(code)
    except PlanRejected as exc:
        return _finalize(SandboxResult(ok=False, error=str(exc)), calls, started)

    with tempfile.TemporaryDirectory() as cwd:
        # Throwaway cwd: the plan has no business reading the app tree or anything
        # in it.
        command = [sys.executable, "-I", "-c", CHILD_SOURCE]
        if network_isolation_available():
            command = ["unshare", "-rn", *command]
        else:
            _warn_no_network_isolation()
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=_child_env(),
            cwd=cwd,
            preexec_fn=_apply_rlimits,
            # Protocol lines carry whole results; asyncio's 64 KiB default would turn a
            # legitimate large result into a plumbing error.
            limit=max_result_bytes * 2 + 65_536,
        )
        try:
            async with asyncio.timeout(timeout_s):
                outcome = await _drive(
                    proc,
                    code=code,
                    primitives=primitives,
                    handler=handler,
                    calls=calls,
                    max_calls=max_calls,
                    max_result_bytes=max_result_bytes,
                )
        except TimeoutError:
            outcome = SandboxResult(
                ok=False, error=f"Plan timed out after {timeout_s}s", calls=list(calls)
            )
        except Exception as exc:
            # Parent-side plumbing failure: keep the message short, no tracebacks.
            outcome = SandboxResult(ok=False, error=f"sandbox failed: {exc}", calls=list(calls))
        finally:
            await _terminate(proc)

    return _finalize(outcome, calls, started)
