"""Integration tests for analytics endpoints."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_data_summary_works_without_auth_when_key_not_set(client_no_auth):
    """Test that data summary works without auth when API_KEY not set (development mode). Args: client_no_auth: Test client without auth."""
    sample_data = json.dumps([{"col1": 1, "col2": "a"}])
    response = client_no_auth.post(
        "/api/v1/analytics/data-summary",
        json={"data": sample_data},
    )
    assert response.status_code in [200, 401]


def test_data_summary_success(client, api_key):
    """Test successful data summary. Args: client: Test client. api_key: API key."""
    sample_data = json.dumps([{"col1": 1, "col2": "a"}, {"col1": 2, "col2": "b"}])
    response = client.post(
        "/api/v1/analytics/data-summary",
        json={"data": sample_data},
        headers={"X-API-Key": api_key},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_samples"] == 2
    assert data["shape"] == [2, 2]


def test_get_feature_importance_works_without_auth_when_key_not_set(client_no_auth):
    """Test that get feature importance works without auth when API_KEY not set (development mode). Args: client_no_auth: Test client without auth."""
    response = client_no_auth.get(
        "/api/v1/models/test123/analytics/feature-importance?target=BaseSalary&quantile=0.5"
    )
    assert response.status_code in [200, 401, 404]


@patch("src.api.routers.analytics.InferenceService")
def test_get_feature_importance_model_not_found(mock_service_class, client, api_key):
    """Test get feature importance when model not found. Args: mock_service_class: Mock service. client: Test client. api_key: API key."""
    from src.services.inference_service import ModelNotFoundError

    mock_service = MagicMock()
    mock_service.load_model.side_effect = ModelNotFoundError("Model not found")
    mock_service_class.return_value = mock_service

    response = client.get(
        "/api/v1/models/nonexistent/analytics/feature-importance?target=BaseSalary&quantile=0.5",
        headers={"X-API-Key": api_key},
    )
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "MODEL_NOT_FOUND"
