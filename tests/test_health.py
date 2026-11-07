from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health_ok_and_has_request_id():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.headers["x-request-id"]


def test_inbound_request_id_is_echoed():
    resp = client.get("/health", headers={"x-request-id": "abc123"})
    assert resp.headers["x-request-id"] == "abc123"
