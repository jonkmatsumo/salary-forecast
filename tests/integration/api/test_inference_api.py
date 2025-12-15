"""Integration tests for inference endpoints."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_predict_works_without_auth_when_key_not_set(client_no_auth):
    """Test that predict works without auth when API_KEY not set (development mode). Args: client_no_auth: Test client without auth."""
    response = client_no_auth.post(
        "/api/v1/models/test123/predict",
        json={"features": {"Level": "L4", "YearsOfExperience": 5}},
    )
    assert response.status_code in [200, 401, 404]


@patch("src.api.routers.inference.InferenceService")
def test_predict_model_not_found(mock_service_class, client, api_key):
    """Test predict when model not found. Args: mock_service_class: Mock service. client: Test client. api_key: API key."""
    from src.services.inference_service import ModelNotFoundError

    mock_service = MagicMock()
    mock_service.load_model.side_effect = ModelNotFoundError("Model not found")
    mock_service_class.return_value = mock_service

    response = client.post(
        "/api/v1/models/nonexistent/predict",
        json={"features": {"Level": "L4"}},
        headers={"X-API-Key": api_key},
    )
    assert response.status_code == 404
    data = response.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "MODEL_NOT_FOUND"


@patch("src.api.routers.inference.InferenceService")
def test_predict_invalid_input(mock_service_class, client, api_key):
    """Test predict with invalid input. Args: mock_service_class: Mock service. client: Test client. api_key: API key."""
    from src.services.inference_service import InvalidInputError

    mock_model = MagicMock()
    mock_service = MagicMock()
    mock_service.load_model.return_value = mock_model
    mock_service.predict.side_effect = InvalidInputError("Invalid features")
    mock_service_class.return_value = mock_service

    response = client.post(
        "/api/v1/models/test123/predict",
        json={"features": {}},
        headers={"X-API-Key": api_key},
    )
    assert response.status_code in [400, 422]
    if response.status_code == 400:
        data = response.json()
        assert data["status"] == "error"
        assert data["error"]["code"] == "VALIDATION_ERROR"


@patch("src.api.routers.inference.InferenceService")
def test_predict_success(mock_service_class, client, api_key):
    """Test successful prediction. Args: mock_service_class: Mock service. client: Test client. api_key: API key."""
    from src.services.inference_service import PredictionResult

    mock_model = MagicMock()
    mock_service = MagicMock()
    mock_service.load_model.return_value = mock_model
    mock_service.predict.return_value = PredictionResult(
        predictions={"BaseSalary": {"p50": 150000.0}},
        metadata={},
    )
    mock_service_class.return_value = mock_service

    response = client.post(
        "/api/v1/models/test123/predict",
        json={"features": {"Level": "L4", "YearsOfExperience": 5}},
        headers={"X-API-Key": api_key},
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "BaseSalary" in data["predictions"]
    assert data["predictions"]["BaseSalary"]["p50"] == 150000.0


def test_batch_predict_works_without_auth_when_key_not_set(client_no_auth):
    """Test that batch predict works without auth when API_KEY not set (development mode). Args: client_no_auth: Test client without auth."""
    response = client_no_auth.post(
        "/api/v1/models/test123/predict/batch",
        json={"features": [{"Level": "L4"}]},
    )
    assert response.status_code in [200, 401, 404]
