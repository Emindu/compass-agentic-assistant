"""Tests for hardcoded-user authentication and JWT round-trips. No network access needed."""

import jwt
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.api.auth import (
    _DEMO_USERS,
    authenticate,
    decode_access_token,
    router,
)
from app.core.config import get_settings
from app.core.errors import AppError, AuthenticationError
from app.security.rbac import Role

client_app = FastAPI()


@client_app.exception_handler(AppError)
async def _app_error_handler(_request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=int(exc.status_code), content={"code": exc.code})


client_app.include_router(router)
client = TestClient(client_app)


def _login(username: str, password: str):
    return client.post("/auth/login", json={"username": username, "password": password})


def test_each_demo_user_gets_a_token_with_the_right_role():
    expected = {
        "vera": (Role.viewer, "viewer-demo"),
        "anil": (Role.analyst, "analyst-demo"),
        "ada": (Role.admin, "admin-demo"),
    }
    for username, (role, password) in expected.items():
        response = _login(username, password)
        assert response.status_code == 200
        body = response.json()
        assert body["token_type"] == "bearer"
        decoded = decode_access_token(body["access_token"])
        assert decoded.user_id == username
        assert decoded.role == role
        assert decoded.name == body["user"]["name"] == _DEMO_USERS[username]["name"]
        assert decoded.department == body["user"]["department"]


def test_wrong_password_and_unknown_user_return_identical_401s():
    wrong = _login("vera", "analyst-demo")
    unknown = _login("nobody", "viewer-demo")
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_me_requires_a_valid_token():
    no_token = client.get("/auth/me")
    assert no_token.status_code == 401

    tampered = _login("vera", "viewer-demo").json()["access_token"][:-2] + "xx"
    headers = {"Authorization": f"Bearer {tampered}"}
    assert client.get("/auth/me", headers=headers).status_code == 401

    token = _login("ada", "admin-demo").json()["access_token"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["role"] == "admin"


def test_expired_token_is_rejected_by_decode_access_token():
    settings = get_settings()
    user = authenticate("anil", "analyst-demo")
    expired = jwt.encode(
        {
            "sub": user.user_id,
            "name": user.name,
            "role": user.role,
            "department": user.department,
            "iat": 0,
            "exp": 1,
        },
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(AuthenticationError):
        decode_access_token(expired)


def test_demo_users_endpoint_leaks_no_password_material():
    response = client.get("/auth/demo-users")
    assert response.status_code == 200
    for entry in response.json():
        assert "password" not in entry
        assert "hash" not in entry
