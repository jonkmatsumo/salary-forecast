"""Unit tests for inference router endpoints."""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.api.dto.inference import BatchPredictionRequest, PredictionRequest
from src.api.exceptions import InvalidInputError, ModelNotFoundError as APIModelNotFoundError
from src.api.routers.inference import predict, predict_batch
from src.services.inference_service import (
    InferenceService,
    InvalidInputError as ServiceInvalidInputError,
    ModelNotFoundError,
    PredictionResult,
)


class TestPredict:
    """Tests for predict endpoint."""

    def test_predict_model_not_found(self):
        """Test ModelNotFoundError raises APIModelNotFoundError."""
        inference_service = MagicMock(spec=InferenceService)
        inference_service.load_model.side_effect = ModelNotFoundError("Model not found")
        
        request = PredictionRequest(features={"Level": "L4"})
        
        with pytest.raises(APIModelNotFoundError) as exc_info:
            asyncio.run(predict("nonexistent", request, user="test_user", inference_service=inference_service))
        
        assert "nonexistent" in str(exc_info.value.message)
        assert exc_info.value.__cause__ is not None

    def test_predict_service_invalid_input_error(self):
        """Test ServiceInvalidInputError raises InvalidInputError."""
        mock_model = MagicMock()
        inference_service = MagicMock(spec=InferenceService)
        inference_service.load_model.return_value = mock_model
        inference_service.predict.side_effect = ServiceInvalidInputError("Invalid features")
        
        request = PredictionRequest(features={"Level": "L4"})
        
        with pytest.raises(InvalidInputError) as exc_info:
            asyncio.run(predict("test123", request, user="test_user", inference_service=inference_service))
        
        assert "Invalid features" in str(exc_info.value.message)
        assert exc_info.value.__cause__ is not None

    def test_predict_success(self):
        """Test successful prediction."""
        mock_model = MagicMock()
        inference_service = MagicMock(spec=InferenceService)
        inference_service.load_model.return_value = mock_model
        inference_service.predict.return_value = PredictionResult(
            predictions={"BaseSalary": {"p50": 150000.0}},
            metadata={"location_zone": "Zone1"},
        )
        
        request = PredictionRequest(features={"Level": "L4", "YearsOfExperience": 5})
        
        response = asyncio.run(predict("test123", request, user="test_user", inference_service=inference_service))
        
        assert response.predictions == {"BaseSalary": {"p50": 150000.0}}
        assert response.metadata.model_run_id == "test123"
        assert response.metadata.location_zone == "Zone1"
        assert isinstance(response.metadata.prediction_timestamp, datetime)


class TestPredictBatch:
    """Tests for predict_batch endpoint."""

    def test_predict_batch_multiple_feature_sets(self):
        """Test predict_batch with multiple feature sets."""
        mock_model = MagicMock()
        inference_service = MagicMock(spec=InferenceService)
        inference_service.load_model.return_value = mock_model
        inference_service.predict.side_effect = [
            PredictionResult(predictions={"BaseSalary": {"p50": 100000.0}}, metadata={}),
            PredictionResult(predictions={"BaseSalary": {"p50": 110000.0}}, metadata={}),
            PredictionResult(predictions={"BaseSalary": {"p50": 120000.0}}, metadata={}),
        ]
        
        request = BatchPredictionRequest(features=[
            {"Level": "L3"},
            {"Level": "L4"},
            {"Level": "L5"},
        ])
        
        response = asyncio.run(predict_batch("test123", request, user="test_user", inference_service=inference_service))
        
        assert len(response.predictions) == 3
        assert response.total == 3
        assert response.predictions[0].predictions == {"BaseSalary": {"p50": 100000.0}}
        assert response.predictions[1].predictions == {"BaseSalary": {"p50": 110000.0}}
        assert response.predictions[2].predictions == {"BaseSalary": {"p50": 120000.0}}

    def test_predict_batch_model_not_found(self):
        """Test batch prediction with ModelNotFoundError."""
        inference_service = MagicMock(spec=InferenceService)
        inference_service.load_model.side_effect = ModelNotFoundError("Model not found")
        
        request = BatchPredictionRequest(features=[{"Level": "L4"}])
        
        with pytest.raises(APIModelNotFoundError) as exc_info:
            asyncio.run(predict_batch("nonexistent", request, user="test_user", inference_service=inference_service))
        
        assert "nonexistent" in str(exc_info.value.message)

    def test_predict_batch_service_invalid_input_error(self):
        """Test batch prediction with ServiceInvalidInputError."""
        mock_model = MagicMock()
        inference_service = MagicMock(spec=InferenceService)
        inference_service.load_model.return_value = mock_model
        inference_service.predict.side_effect = ServiceInvalidInputError("Invalid features")
        
        request = BatchPredictionRequest(features=[{"Level": "L4"}])
        
        with pytest.raises(InvalidInputError) as exc_info:
            asyncio.run(predict_batch("test123", request, user="test_user", inference_service=inference_service))
        
        assert "Invalid features" in str(exc_info.value.message)

    def test_predict_batch_single_feature(self):
        """Test batch prediction with single feature set."""
        mock_model = MagicMock()
        inference_service = MagicMock(spec=InferenceService)
        inference_service.load_model.return_value = mock_model
        inference_service.predict.return_value = PredictionResult(
            predictions={"BaseSalary": {"p50": 100000.0}},
            metadata={},
        )
        
        request = BatchPredictionRequest(features=[{"Level": "L4"}])
        
        response = asyncio.run(predict_batch("test123", request, user="test_user", inference_service=inference_service))
        
        assert len(response.predictions) == 1
        assert response.total == 1

    def test_predict_batch_with_metadata(self):
        """Test batch prediction includes correct metadata."""
        mock_model = MagicMock()
        inference_service = MagicMock(spec=InferenceService)
        inference_service.load_model.return_value = mock_model
        inference_service.predict.return_value = PredictionResult(
            predictions={"BaseSalary": {"p50": 150000.0}},
            metadata={"location_zone": "Zone2"},
        )
        
        request = BatchPredictionRequest(features=[
            {"Level": "L4"},
            {"Level": "L5"},
        ])
        
        response = asyncio.run(predict_batch("test123", request, user="test_user", inference_service=inference_service))
        
        assert len(response.predictions) == 2
        assert response.predictions[0].metadata.model_run_id == "test123"
        assert response.predictions[0].metadata.location_zone == "Zone2"
        assert response.predictions[1].metadata.model_run_id == "test123"

