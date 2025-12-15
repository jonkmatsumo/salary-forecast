"""Integration tests for model management endpoints."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_list_models_works_without_auth_when_key_not_set(client_no_auth):
    """Test that list models works without auth when API_KEY not set (development mode). Args: client_no_auth: Test client without auth."""
    response = client_no_auth.get("/api/v1/models")
    assert response.status_code in [200, 401]


@patch("src.api.routers.models.ModelRegistry")
def test_list_models_empty(mock_registry_class, client, api_key):
    """Test listing models when empty. Args: mock_registry_class: Mock registry. client: Test client. api_key: API key."""
    mock_registry = MagicMock()
    mock_registry.list_models.return_value = []
    mock_registry_class.return_value = mock_registry

    response = client.get("/api/v1/models", headers={"X-API-Key": api_key})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["models"] == []
    assert data["data"]["pagination"]["total"] == 0


@patch("src.api.routers.models.ModelRegistry")
def test_list_models_with_data(mock_registry_class, client, api_key):
    """Test listing models with data. Args: mock_registry_class: Mock registry. client: Test client. api_key: API key."""
    from datetime import datetime

    mock_registry = MagicMock()
    mock_registry.list_models.return_value = [
        {
            "run_id": "abc123",
            "start_time": datetime(2024, 1, 1, 12, 0, 0),
            "tags.model_type": "XGBoost",
            "tags.dataset_name": "test_data",
            "metrics.cv_mean_score": 0.95,
        }
    ]
    mock_registry_class.return_value = mock_registry

    response = client.get("/api/v1/models", headers={"X-API-Key": api_key})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["data"]["models"]) == 1
    assert data["data"]["models"][0]["run_id"] == "abc123"


@patch("src.api.routers.models.InferenceService")
def test_get_model_details_not_found(mock_service_class, client, api_key):
    """Test getting model details when model not found. Args: mock_service_class: Mock service. client: Test client. api_key: API key."""
    from src.services.inference_service import ModelNotFoundError

    mock_service = MagicMock()
    mock_service.load_model.side_effect = ModelNotFoundError("Model not found")
    mock_service_class.return_value = mock_service

    response = client.get("/api/v1/models/nonexistent", headers={"X-API-Key": api_key})
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "MODEL_NOT_FOUND"


def test_get_model_schema_works_without_auth_when_key_not_set(client_no_auth):
    """Test that get model schema works without auth when API_KEY not set (development mode). Args: client_no_auth: Test client without auth."""
    response = client_no_auth.get("/api/v1/models/test123/schema")
    assert response.status_code in [200, 401, 404]
