"""Typed application errors.

Why typed errors: each failure class maps to one HTTP status and one user-safe message, and graph
nodes can record them in state by `code` rather than leaking stack traces to the user. The
`retryable` flag tells callers (and the supervisor) whether a degraded retry path makes sense.
"""

from http import HTTPStatus


class AppError(Exception):
    """Base class. `message` is safe to show to end users; details go to logs only."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    code: str = "internal_error"
    message: str = "Something went wrong. Please try again."
    retryable: bool = False

    def __init__(self, detail: str | None = None, *, message: str | None = None) -> None:
        self.detail = detail
        if message is not None:
            self.message = message
        super().__init__(detail or self.message)


class InvalidRequestError(AppError):
    status_code = HTTPStatus.BAD_REQUEST
    code = "invalid_request"
    message = "The request was not valid."


class AuthenticationError(AppError):
    status_code = HTTPStatus.UNAUTHORIZED
    code = "unauthenticated"
    message = "Please sign in again."


class AuthorizationError(AppError):
    status_code = HTTPStatus.FORBIDDEN
    code = "forbidden"
    message = "Your role does not have access to that action."


class RateLimitError(AppError):
    status_code = HTTPStatus.TOO_MANY_REQUESTS
    code = "rate_limited"
    message = "You're sending requests too quickly. Please wait a moment."
    retryable = True

    def __init__(self, retry_after_s: float, detail: str | None = None) -> None:
        self.retry_after_s = retry_after_s
        super().__init__(detail)


class GuardrailViolation(AppError):
    status_code = HTTPStatus.BAD_REQUEST
    code = "guardrail_violation"
    message = "I can't help with that request."


class LLMError(AppError):
    status_code = HTTPStatus.BAD_GATEWAY
    code = "llm_unavailable"
    message = "The language model is temporarily unavailable."
    retryable = True


class VectorStoreError(AppError):
    status_code = HTTPStatus.BAD_GATEWAY
    code = "vector_store_unavailable"
    message = "Document search is temporarily degraded."
    retryable = True


class MCPError(AppError):
    status_code = HTTPStatus.BAD_GATEWAY
    code = "mcp_unavailable"
    message = "An enterprise data service is temporarily unavailable."
    retryable = True


class ToolTimeoutError(AppError):
    status_code = HTTPStatus.GATEWAY_TIMEOUT
    code = "tool_timeout"
    message = "A tool took too long to respond."
    retryable = True
