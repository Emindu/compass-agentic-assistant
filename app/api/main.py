"""FastAPI entrypoint: lifespan (graph + checkpointer), middleware, error handling, routes."""

import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.errors import AppError, RateLimitError
from app.core.logging import configure_logging, get_logger, request_id_var

# LangSmith and the LLM SDKs read os.environ directly, so export .env before anything else.
load_dotenv()

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402

from app.agents.graph import build_graph  # noqa: E402
from app.api import auth  # noqa: E402
from app.api.routes import chat  # noqa: E402
from app.retrieval.pinecone_store import get_store  # noqa: E402

settings = get_settings()
configure_logging(settings.log_level)
log = get_logger(__name__)

CHECKPOINT_DB = Path("data/checkpoints/compass.sqlite")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the checkpointer for the app's lifetime and warm Pinecone connections, so the first
    user request doesn't pay about 4 s of connection setup."""
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        app.state.graph = build_graph(checkpointer)
        try:
            store = get_store()
            for index in (settings.pinecone_dense_index, settings.pinecone_sparse_index):
                await store.search(index, "warm up", ["corporate"], 1, {"access_rank": 0})
            log.info("pinecone_warm")
        except Exception as exc:  # degraded mode is still usable; don't block start-up
            log.warning("pinecone_warmup_failed", error=repr(exc))
        yield
        try:
            await get_store().aclose()
        except Exception:
            pass


app = FastAPI(title=f"{settings.app_name} API", version="0.1.0", lifespan=lifespan)
app.include_router(auth.router)
app.include_router(chat.router)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Assign a request id (or honour an inbound one) and log one line per request."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        log.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        return response
    finally:
        request_id_var.reset(token)


def _error_body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message, "request_id": request_id_var.get()}}


@app.exception_handler(AppError)
async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    log.warning("app_error", code=exc.code, detail=exc.detail, retryable=exc.retryable)
    headers = {}
    if isinstance(exc, RateLimitError):
        headers["Retry-After"] = str(max(1, round(exc.retry_after_s)))
    return JSONResponse(
        status_code=int(exc.status_code),
        content=_error_body(exc.code, exc.message),
        headers=headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    log.info("invalid_request", errors=exc.errors())
    return JSONResponse(
        status_code=400, content=_error_body("invalid_request", "The request was not valid.")
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    # Never leak internals to the client; the full trace goes to the structured log.
    log.exception("unhandled_error", error_type=type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content=_error_body("internal_error", "Something went wrong. Please try again."),
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "app": settings.app_name, "environment": settings.environment}
