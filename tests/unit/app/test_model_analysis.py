"""
Tests for the standalone model_analysis module.

Note: The model_analysis functionality has been integrated into the Inference tab
as a collapsible "Model Analysis" section. These tests verify the standalone module
still works correctly for backward compatibility.
"""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import streamlit as st
from datetime import datetime
from src.app.model_analysis import render_model_analysis_ui

@pytest.fixture
def mock_streamlit():
    with patch("src.app.model_analysis.st") as mock_st:
        # Defaults
        mock_st.selectbox.return_value = None
        yield mock_st

@pytest.fixture
def mock_registry():
    with patch("src.app.model_analysis.ModelRegistry") as MockReg:
        yield MockReg.return_value

@pytest.fixture
def mock_analytics():
    with patch("src.app.model_analysis.AnalyticsService") as MockAn:
        yield MockAn.return_value

def test_no_models_shows_warning(mock_streamlit, mock_registry):
    mock_registry.list_models.return_value = []
    render_model_analysis_ui()
    mock_streamlit.warning.assert_called_with("No models found in MLflow. Please train a new model.")

def test_load_valid_model(mock_streamlit, mock_registry, mock_analytics):
    run_data = {
        "run_id": "run123",
        "start_time": datetime(2023,1,1,12,0),
        "metrics.cv_mean_score": 0.99
    }
    mock_registry.list_models.return_value = [run_data]
    
    # Construct expected label
    expected_label = f"2023-01-01 12:00 | CV:0.9900 | ID:run123"
    
    # User selects the label
    mock_streamlit.selectbox.side_effect = [expected_label, "BaseSalary", 0.5] 
    
    # Actual Forecaster
    mock_forecaster = MagicMock()
    # Mock Forecaster wrapper returned by MLflow
    mock_wrapper = MagicMock()
    mock_wrapper.unwrap_python_model.return_value = mock_forecaster
    # Actual Forecaster
    mock_forecaster = MagicMock()
    # Mock Analytics Responses
    mock_analytics.get_available_targets.return_value = ["BaseSalary"]
    mock_analytics.get_available_quantiles.return_value = [0.5]
    
    # Registry returns the wrapper first? No, our code calls unwrap.
    # registry.load_model -> returns unwrap_python_model().
    # So we just need registry.load_model to return the forecaster directly 
    # IF the registry method does the unwrapping internally (which it does).
    # Wait, let's check code:
    # return mlflow.pyfunc.load_model(model_uri).unwrap_python_model()
    # So the mock need to act as mlflow. But here we are mocking registry.load_model directly.
    # Ah! The test mocks `src.app.model_analysis.ModelRegistry`.
    # So `mock_registry.load_model.return_value` should be the FORECASTER object itself.
    
    mock_registry.load_model.return_value = mock_forecaster
    mock_analytics.get_available_quantiles.return_value = [0.5]
    
    # Mock Feature Importance
    df_imp = pd.DataFrame({"Feature": ["A"], "Gain": [10.0]})
    mock_analytics.get_feature_importance.return_value = df_imp
    
    render_model_analysis_ui()
    
    # Verify success loading
    mock_streamlit.success.assert_called()
    
    # Verify plotting
    # Verify plotting
    # Assuming get_feature_importance returns valid DF, it plots
    mock_streamlit.pyplot.assert_called()

def test_empty_importance(mock_streamlit, mock_registry, mock_analytics):
    # Construct expected label based on default mock behavior?  
    # Better to define specific run data
    run_data = {
        "run_id": "run123",
        "start_time": datetime(2023,1,1,12,0),
        "metrics.cv_mean_score": 0.99
    }
    mock_registry.list_models.return_value = [run_data]
    
    mock_streamlit.selectbox.side_effect = [f"2023-01-01 12:00 | CV:0.9900 | ID:run123", "BaseSalary", 0.5]
    
    mock_forecaster = MagicMock()
    mock_registry.load_model.return_value = mock_forecaster
    
    mock_analytics.get_available_targets.return_value = ["BaseSalary"]
    mock_analytics.get_available_quantiles.return_value = [0.5]
    
    # Return empty importance
    mock_analytics.get_feature_importance.return_value = pd.DataFrame()
    
    render_model_analysis_ui()
        
    mock_streamlit.warning.assert_called()
    assert "No feature importance scores found" in mock_streamlit.warning.call_args_list[-1][0][0]
