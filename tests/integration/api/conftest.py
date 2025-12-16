"""Pytest fixtures for API integration tests."""

import os
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app


@pytest.fixture
def api_key() -> str:
    """Fixture providing API key for tests.

    Returns:
        str: API key.
    """
    return "test_api_key_123"


@pytest.fixture
def client(api_key: str) -> Generator[TestClient, None, None]:
    """Fixture providing FastAPI test client.

    Args:
        api_key (str): API key.

    Returns:
        Generator[TestClient, None, None]: Test client.
    """
    os.environ["API_KEY"] = api_key
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    if "API_KEY" in os.environ:
        del os.environ["API_KEY"]


@pytest.fixture
def client_no_auth() -> Generator[TestClient, None, None]:
    """Fixture providing FastAPI test client without authentication.

    Returns:
        Generator[TestClient, None, None]: Test client.
    """
    old_key = os.environ.get("API_KEY")
    if "API_KEY" in os.environ:
        del os.environ["API_KEY"]
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    if old_key:
        os.environ["API_KEY"] = old_key
