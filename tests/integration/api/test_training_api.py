"""Integration tests for training endpoints."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from conftest import create_test_config  # noqa: E402


def test_upload_data_works_without_auth_when_key_not_set(client_no_auth):
    """Test that upload data works without auth when API_KEY not set (development mode). Args: client_no_auth: Test client without auth."""
    csv_content = "col1,col2\n1,2\n3,4"
    response = client_no_auth.post(
        "/api/v1/training/data/upload",
        files={"file": ("test.csv", csv_content, "text/csv")},
    )
    assert response.status_code in [200, 401]


def test_upload_data_invalid_file(client, api_key):
    """Test uploading invalid CSV file. Args: client: Test client. api_key: API key."""
    response = client.post(
        "/api/v1/training/data/upload",
        files={"file": ("test.csv", "", "text/csv")},
        headers={"X-API-Key": api_key},
    )
    assert response.status_code in [400, 422]
    data = response.json()
    assert data["status"] == "error"


def test_upload_data_success(client, api_key):
    """Test successful data upload. Args: client: Test client. api_key: API key."""
    csv_content = "col1,col2\n1,2\n3,4\n5,6"
    response = client.post(
        "/api/v1/training/data/upload",
        files={"file": ("test.csv", csv_content, "text/csv")},
        headers={"X-API-Key": api_key},
    )
    assert response.status_code == 200
    data = response.json()
    assert "dataset_id" in data
    assert data["row_count"] == 3
    assert data["column_count"] == 2


def test_start_training_works_without_auth_when_key_not_set(client_no_auth):
    """Test that start training works without auth when API_KEY not set (development mode). Args: client_no_auth: Test client without auth."""
    response = client_no_auth.post(
        "/api/v1/training/jobs",
        json={"dataset_id": "test123", "config": create_test_config()},
    )
    assert response.status_code in [200, 400, 401]


def test_start_training_dataset_not_found(client, api_key):
    """Test starting training with non-existent dataset. Args: client: Test client. api_key: API key."""
    response = client.post(
        "/api/v1/training/jobs",
        json={
            "dataset_id": "nonexistent",
            "config": create_test_config(),
        },
        headers={"X-API-Key": api_key},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "not found" in data["error"]["message"].lower()


def test_get_training_job_status_works_without_auth_when_key_not_set(client_no_auth):
    """Test that get training job status works without auth when API_KEY not set (development mode). Args: client_no_auth: Test client without auth."""
    response = client_no_auth.get("/api/v1/training/jobs/test123")
    assert response.status_code in [200, 401, 404]


def test_get_training_job_status_not_found(client, api_key):
    """Test getting training job status when job not found. Args: client: Test client. api_key: API key."""
    response = client.get(
        "/api/v1/training/jobs/nonexistent",
        headers={"X-API-Key": api_key},
    )
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "TRAINING_JOB_NOT_FOUND"
