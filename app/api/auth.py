"""Authentication for the POC: three hardcoded users, HS256 JWTs.

POC only: production authenticates against the bank's IdP (Keycloak) via OIDC; only this
module changes, the RBAC code does not. Passwords are stored as salted PBKDF2-SHA256
hashes so the plaintext demo passwords appear exactly once (here, at import time), and
login failures return one identical error so an attacker cannot probe which usernames
exist by timing or distinguishing responses.
"""

import hashlib
import hmac
from datetime import UTC, datetime, timedelta

import jwt
import structlog
from fastapi import APIRouter, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.errors import AuthenticationError
from app.core.logging import get_logger
from app.security.rbac import Role, UserContext

log = get_logger(__name__)
_bearer = HTTPBearer(auto_error=False)

_PBKDF2_ITERATIONS = 100_000


def _hash_password(password: str, salt: str) -> bytes:
    """Salted PBKDF2-SHA256. Slow by design so a leaked hash table is expensive to crack."""
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _PBKDF2_ITERATIONS)


# Fixed per-user salts are acceptable for a POC with public demo credentials; production
# salts are random per password and stored alongside the hash in the IdP.
_DEMO_USERS: dict[str, dict] = {
    "vera": {
        "name": "Vera Viewer",
        "role": Role.viewer,
        "department": "retail",
        "salt": "vera-salt",
        "hash": _hash_password("viewer-demo", "vera-salt"),
    },
    "anil": {
        "name": "Anil Analyst",
        "role": Role.analyst,
        "department": "payments",
        "salt": "anil-salt",
        "hash": _hash_password("analyst-demo", "anil-salt"),
    },
    "ada": {
        "name": "Ada Admin",
        "role": Role.admin,
        "department": "platform",
        "salt": "ada-salt",
        "hash": _hash_password("admin-demo", "ada-salt"),
    },
}


class LoginRequest(BaseModel):
    username: str = Field(max_length=100)
    password: str = Field(max_length=100)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserContext


def authenticate(username: str, password: str) -> UserContext:
    """Return the user's context if the credentials match, else raise AuthenticationError.

    The same error (and message) is used for unknown users and wrong passwords so the
    endpoint does not leak which username exists.
    """
    record = _DEMO_USERS.get(username)
    if record is None or not hmac.compare_digest(
        _hash_password(password, record["salt"]), record["hash"]
    ):
        log.info("login_failed")  # no password material, and no hint about the reason
        raise AuthenticationError("Invalid username or password")
    return UserContext(
        user_id=username, name=record["name"], role=record["role"], department=record["department"]
    )


def create_access_token(user: UserContext) -> str:
    """Mint an HS256 JWT. Claims mirror UserContext so the token is the single identity source."""
    settings = get_settings()
    now = datetime.now(UTC)
    claims = {
        "sub": user.user_id,
        "name": user.name,
        "role": user.role,
        "department": user.department,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_ttl_minutes),
    }
    secret = settings.jwt_secret.get_secret_value()
    return jwt.encode(claims, secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> UserContext:
    """Verify a JWT and rebuild the UserContext; any failure is one AuthenticationError.

    Signature/expiry problems and unknown roles all collapse to the same error so callers
    never need to branch on *why* the token was rejected.
    """
    settings = get_settings()
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
        return UserContext(
            user_id=claims["sub"],
            name=claims["name"],
            role=Role(claims["role"]),
            department=claims["department"],
        )
    except (jwt.InvalidTokenError, KeyError, ValueError) as exc:
        raise AuthenticationError("Invalid or expired token") from exc


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> UserContext:
    """Resolve the bearer token to a UserContext, or reject the request as unauthenticated.

    Binding user_id and role into the structlog contextvars means every log line emitted
    while serving this request (including inside graph nodes) carries the caller's
    identity without threading it through arguments.
    """
    if credentials is None:
        raise AuthenticationError("Missing bearer token")
    user = decode_access_token(credentials.credentials)
    structlog.contextvars.bind_contextvars(user_id=user.user_id, role=user.role)
    return user


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest) -> TokenResponse:
    user = authenticate(request.username, request.password)
    return TokenResponse(access_token=create_access_token(user), user=user)


@router.get("/me", response_model=UserContext)
async def me(user: UserContext = Depends(get_current_user)) -> UserContext:
    return user


@router.get("/demo-users")
async def demo_users() -> list[dict]:
    """Demo identities for the Streamlit login dropdown. Password material is never exposed."""
    return [
        {"username": username, "name": r["name"], "role": r["role"], "department": r["department"]}
        for username, r in _DEMO_USERS.items()
    ]
