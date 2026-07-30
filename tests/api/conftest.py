import pytest
from fastapi.testclient import TestClient

from app import web_app


@pytest.fixture
def client():
    """A test client for the API."""
    return TestClient(web_app)
