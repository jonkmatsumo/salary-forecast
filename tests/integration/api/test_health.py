"""Tests for health check endpoint."""


def test_health_check(client):
    """Test health check endpoint. Args: client: Test client."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_health_check_no_auth(client_no_auth):
    """Test health check works without authentication. Args: client_no_auth: Test client without auth."""
    response = client_no_auth.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
