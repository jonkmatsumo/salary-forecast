"""Unit tests for models router endpoints."""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.api.exceptions import ModelNotFoundError as APIModelNotFoundError
from src.api.routers.models import get_model_details, get_model_schema, list_models
from src.services.inference_service import InferenceService, ModelNotFoundError, ModelSchema
from src.services.model_registry import ModelRegistry


class TestListModels:
    """Tests for list_models endpoint."""

    def test_list_models_with_experiment_filter(self):
        """Test filtering by experiment_name."""
        registry = MagicMock(spec=ModelRegistry)
        registry.list_models.return_value = [
            {
                "run_id": "run1",
                "start_time": datetime(2023, 1, 1),
                "tags.experiment_name": "exp1",
                "tags.model_type": "XGBoost",
                "metrics.cv_mean_score": 0.95,
                "tags.dataset_name": "dataset1",
            },
            {
                "run_id": "run2",
                "start_time": datetime(2023, 1, 2),
                "tags.experiment_name": "exp2",
                "tags.model_type": "XGBoost",
                "metrics.cv_mean_score": 0.90,
                "tags.dataset_name": "dataset2",
            },
        ]
        
        response = asyncio.run(list_models(
            limit=50,
            offset=0,
            experiment_name="exp1",
            user="test_user",
            registry=registry,
        ))
        
        assert response.status == "success"
        assert len(response.data["models"]) == 1
        assert response.data["models"][0]["run_id"] == "run1"

    def test_list_models_without_filter(self):
        """Test list_models without experiment_name filter."""
        registry = MagicMock(spec=ModelRegistry)
        registry.list_models.return_value = [
            {
                "run_id": "run1",
                "start_time": datetime(2023, 1, 1),
                "tags.model_type": "XGBoost",
                "tags.dataset_name": "dataset1",
            },
        ]
        
        response = asyncio.run(list_models(
            limit=50,
            offset=0,
            experiment_name=None,
            user="test_user",
            registry=registry,
        ))
        
        assert response.status == "success"
        assert len(response.data["models"]) == 1


class TestGetModelDetails:
    """Tests for get_model_details endpoint."""

    @patch("src.api.routers.models.ModelRegistry")
    def test_get_model_details_success(self, mock_registry_class):
        """Test successful model details retrieval."""
        mock_model = MagicMock()
        mock_model.ranked_encoders = {
            "Level": MagicMock(mapping={"E3": 0, "E4": 1, "E5": 2}),
        }
        mock_model.proximity_encoders = {
            "Location": MagicMock(),
        }
        mock_model.feature_names = ["Level_Enc", "Location_Enc", "YearsOfExperience"]
        mock_model.targets = ["BaseSalary"]
        mock_model.quantiles = [0.5, 0.75, 0.9]
        
        inference_service = MagicMock(spec=InferenceService)
        schema = ModelSchema(mock_model)
        inference_service.load_model.return_value = mock_model
        inference_service.get_model_schema.return_value = schema
        
        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.list_models.return_value = [
            {
                "run_id": "test123",
                "start_time": datetime(2023, 1, 1),
                "tags.model_type": "XGBoost",
                "metrics.cv_mean_score": 0.95,
                "tags.dataset_name": "test_dataset",
                "tags.additional_tag": "test_tag",
            },
        ]
        mock_registry_class.return_value = mock_registry
        
        response = asyncio.run(get_model_details("test123", user="test_user", inference_service=inference_service))
        
        assert response.run_id == "test123"
        assert response.metadata.run_id == "test123"
        assert len(response.model_schema.ranked_features) == 1
        assert response.model_schema.ranked_features[0].name == "Level"
        assert set(response.model_schema.ranked_features[0].levels) == {"E3", "E4", "E5"}
        assert len(response.model_schema.proximity_features) == 1
        assert response.model_schema.proximity_features[0].name == "Location"

    @patch("src.api.routers.models.ModelRegistry")
    def test_get_model_details_model_not_found_in_registry(self, mock_registry_class):
        """Test model not found in registry after loading."""
        mock_model = MagicMock()
        inference_service = MagicMock(spec=InferenceService)
        schema = ModelSchema(mock_model)
        inference_service.load_model.return_value = mock_model
        inference_service.get_model_schema.return_value = schema
        
        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.list_models.return_value = [
            {
                "run_id": "other_run",
                "start_time": datetime(2023, 1, 1),
            },
        ]
        mock_registry_class.return_value = mock_registry
        
        with pytest.raises(APIModelNotFoundError) as exc_info:
            asyncio.run(get_model_details("test123", user="test_user", inference_service=inference_service))
        
        assert "test123" in str(exc_info.value.message)

    def test_get_model_details_model_not_found_error(self):
        """Test ModelNotFoundError from inference service."""
        inference_service = MagicMock(spec=InferenceService)
        inference_service.load_model.side_effect = ModelNotFoundError("Model not found")
        
        with pytest.raises(APIModelNotFoundError) as exc_info:
            asyncio.run(get_model_details("nonexistent", user="test_user", inference_service=inference_service))
        
        assert "nonexistent" in str(exc_info.value.message)
        assert exc_info.value.__cause__ is not None


class TestGetModelSchema:
    """Tests for get_model_schema endpoint."""

    def test_get_model_schema_success(self):
        """Test successful schema retrieval."""
        mock_model = MagicMock()
        mock_model.ranked_encoders = {
            "Level": MagicMock(mapping={"E3": 0, "E4": 1}),
        }
        mock_model.proximity_encoders = {
            "Location": MagicMock(),
        }
        mock_model.feature_names = ["Level_Enc", "Location_Enc", "YearsOfExperience"]
        
        inference_service = MagicMock(spec=InferenceService)
        schema = ModelSchema(mock_model)
        inference_service.load_model.return_value = mock_model
        inference_service.get_model_schema.return_value = schema
        
        response = asyncio.run(get_model_schema("test123", user="test_user", inference_service=inference_service))
        
        assert response.run_id == "test123"
        assert len(response.model_schema.ranked_features) == 1
        assert len(response.model_schema.proximity_features) == 1
        assert len(response.model_schema.numerical_features) > 0

    def test_get_model_schema_model_not_found_error(self):
        """Test ModelNotFoundError raises APIModelNotFoundError."""
        inference_service = MagicMock(spec=InferenceService)
        inference_service.load_model.side_effect = ModelNotFoundError("Model not found")
        
        with pytest.raises(APIModelNotFoundError) as exc_info:
            asyncio.run(get_model_schema("nonexistent", user="test_user", inference_service=inference_service))
        
        assert "nonexistent" in str(exc_info.value.message)
        assert exc_info.value.__cause__ is not None

    def test_get_model_schema_empty_encoders(self):
        """Test schema retrieval with empty encoders."""
        mock_model = MagicMock()
        mock_model.ranked_encoders = {}
        mock_model.proximity_encoders = {}
        mock_model.feature_names = ["YearsOfExperience"]
        
        inference_service = MagicMock(spec=InferenceService)
        schema = ModelSchema(mock_model)
        inference_service.load_model.return_value = mock_model
        inference_service.get_model_schema.return_value = schema
        
        response = asyncio.run(get_model_schema("test123", user="test_user", inference_service=inference_service))
        
        assert len(response.model_schema.ranked_features) == 0
        assert len(response.model_schema.proximity_features) == 0
        assert len(response.model_schema.numerical_features) > 0

