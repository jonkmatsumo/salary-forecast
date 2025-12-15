"""Unit tests for analytics router endpoints."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.api.dto.analytics import DataSummaryRequest
from src.api.exceptions import InvalidInputError, ModelNotFoundError as APIModelNotFoundError
from src.api.routers.analytics import get_data_summary, get_feature_importance
from src.services.analytics_service import AnalyticsService
from src.services.inference_service import InferenceService, ModelNotFoundError


class TestGetDataSummary:
    """Tests for get_data_summary endpoint."""

    @patch("pandas.read_json")
    def test_get_data_summary_exception_during_json_parsing(self, mock_read_json):
        """Test exception during JSON parsing raises InvalidInputError."""
        mock_read_json.side_effect = ValueError("Invalid JSON format")
        
        request = DataSummaryRequest(data=json.dumps([{"col1": 1}]))
        analytics_service = MagicMock(spec=AnalyticsService)
        
        with pytest.raises(InvalidInputError) as exc_info:
            asyncio.run(get_data_summary(request, user="test_user", analytics_service=analytics_service))
        
        assert "Failed to parse data" in str(exc_info.value.message)
        assert "Invalid JSON format" in str(exc_info.value.message)
        assert exc_info.value.__cause__ is not None

    @patch("pandas.read_json")
    def test_get_data_summary_json_decode_error(self, mock_read_json):
        """Test JSONDecodeError during parsing raises InvalidInputError."""
        from json import JSONDecodeError
        
        mock_read_json.side_effect = JSONDecodeError("Expecting value", "", 0)
        
        request = DataSummaryRequest(data="invalid json")
        analytics_service = MagicMock(spec=AnalyticsService)
        
        with pytest.raises(InvalidInputError) as exc_info:
            asyncio.run(get_data_summary(request, user="test_user", analytics_service=analytics_service))
        
        assert "Failed to parse data" in str(exc_info.value.message)

    @patch("pandas.read_json")
    def test_get_data_summary_success(self, mock_read_json):
        """Test successful data summary."""
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        mock_read_json.return_value = df
        
        analytics_service = MagicMock(spec=AnalyticsService)
        analytics_service.get_data_summary.return_value = {
            "total_samples": 2,
            "shape": (2, 2),
            "unique_col2": 2,
        }
        
        request = DataSummaryRequest(data=json.dumps([{"col1": 1, "col2": "a"}]))
        
        response = asyncio.run(get_data_summary(request, user="test_user", analytics_service=analytics_service))
        
        assert response.total_samples == 2
        assert response.shape == (2, 2)
        assert "col2" in response.unique_counts
        assert response.unique_counts["col2"] == 2


class TestGetFeatureImportance:
    """Tests for get_feature_importance endpoint."""

    def test_get_feature_importance_df_imp_is_none(self):
        """Test df_imp is None raises InvalidInputError."""
        mock_model = MagicMock()
        inference_service = MagicMock(spec=InferenceService)
        inference_service.load_model.return_value = mock_model
        
        analytics_service = MagicMock(spec=AnalyticsService)
        analytics_service.get_feature_importance.return_value = None
        
        with pytest.raises(InvalidInputError) as exc_info:
            asyncio.run(get_feature_importance(
                run_id="test123",
                target="BaseSalary",
                quantile=0.5,
                user="test_user",
                inference_service=inference_service,
                analytics_service=analytics_service,
            ))
        
        assert "No feature importance found" in str(exc_info.value.message)
        assert "BaseSalary" in str(exc_info.value.message)
        assert "0.5" in str(exc_info.value.message)

    def test_get_feature_importance_model_not_found(self):
        """Test ModelNotFoundError from service raises APIModelNotFoundError."""
        inference_service = MagicMock(spec=InferenceService)
        inference_service.load_model.side_effect = ModelNotFoundError("Model not found")
        
        analytics_service = MagicMock(spec=AnalyticsService)
        
        with pytest.raises(APIModelNotFoundError) as exc_info:
            asyncio.run(get_feature_importance(
                run_id="nonexistent",
                target="BaseSalary",
                quantile=0.5,
                user="test_user",
                inference_service=inference_service,
                analytics_service=analytics_service,
            ))
        
        assert "nonexistent" in str(exc_info.value.message)
        assert exc_info.value.__cause__ is not None

    def test_get_feature_importance_empty_dataframe(self):
        """Test empty feature importance DataFrame."""
        mock_model = MagicMock()
        inference_service = MagicMock(spec=InferenceService)
        inference_service.load_model.return_value = mock_model
        
        analytics_service = MagicMock(spec=AnalyticsService)
        analytics_service.get_feature_importance.return_value = pd.DataFrame(columns=["Feature", "Gain"])
        
        response = asyncio.run(get_feature_importance(
            run_id="test123",
            target="BaseSalary",
            quantile=0.5,
            user="test_user",
            inference_service=inference_service,
            analytics_service=analytics_service,
        ))
        
        assert len(response.features) == 0

    def test_get_feature_importance_success(self):
        """Test successful feature importance retrieval."""
        mock_model = MagicMock()
        inference_service = MagicMock(spec=InferenceService)
        inference_service.load_model.return_value = mock_model
        
        df_imp = pd.DataFrame({
            "Feature": ["feat1", "feat2"],
            "Gain": [0.5, 0.3],
        })
        
        analytics_service = MagicMock(spec=AnalyticsService)
        analytics_service.get_feature_importance.return_value = df_imp
        
        response = asyncio.run(get_feature_importance(
            run_id="test123",
            target="BaseSalary",
            quantile=0.5,
            user="test_user",
            inference_service=inference_service,
            analytics_service=analytics_service,
        ))
        
        assert len(response.features) == 2
        assert response.features[0].name == "feat1"
        assert response.features[0].gain == 0.5
        assert response.features[1].name == "feat2"
        assert response.features[1].gain == 0.3

