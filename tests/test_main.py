import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.main import app

TEST_API_KEY = "test-secret-key"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("APP_SECRET_KEY", TEST_API_KEY)
    mock_service = MagicMock()
    mock_service.ask_question.return_value = "Test answer."
    with patch("src.main.QAService", return_value=mock_service):
        with TestClient(app) as c:
            yield c


@pytest.fixture
def client_with_broken_service(monkeypatch):
    """Client where QAService fails to initialize."""
    monkeypatch.setenv("APP_SECRET_KEY", TEST_API_KEY)
    with patch("src.main.QAService", side_effect=Exception("Init failed")):
        with TestClient(app) as c:
            yield c


def test_ask_returns_403_without_api_key(client):
    response = client.post("/ask", json={"question": "Hello?"})
    assert response.status_code == 403


def test_ask_returns_403_with_wrong_api_key(client):
    response = client.post("/ask", json={"question": "Hello?"}, headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 403


def test_ask_returns_400_on_empty_question(client):
    response = client.post("/ask", json={"question": "   "}, headers={"X-API-Key": TEST_API_KEY})
    assert response.status_code == 400


def test_ask_returns_200_with_valid_request(client):
    response = client.post("/ask", json={"question": "What is RAG?"}, headers={"X-API-Key": TEST_API_KEY})
    assert response.status_code == 200
    assert response.json()["answer"] == "Test answer."


def test_ask_returns_503_when_service_unavailable(client_with_broken_service):
    response = client_with_broken_service.post(
        "/ask",
        json={"question": "What is RAG?"},
        headers={"X-API-Key": TEST_API_KEY},
    )
    assert response.status_code == 503


def test_app_refuses_to_start_without_app_secret_key(monkeypatch):
    monkeypatch.delenv("APP_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="APP_SECRET_KEY"):
        with TestClient(app):
            pass
