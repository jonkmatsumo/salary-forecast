"""
Tests for the standalone model_analysis module.

Note: The model_analysis functionality has been integrated into the Inference tab
as a collapsible "Model Analysis" section. These tests verify the standalone module
still works correctly for backward compatibility.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

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
    with patch("src.app.model_analysis.get_analytics_service") as mock_get_analytics:
        mock_instance = MagicMock()
        mock_get_analytics.return_value = mock_instance
        yield mock_instance


def test_no_models_shows_warning(mock_streamlit, mock_registry):
    mock_registry.list_models.return_value = []
    render_model_analysis_ui()
    mock_streamlit.warning.assert_called_with(
        "No models found in MLflow. Please train a new model."
    )


@patch("src.app.model_analysis.get_inference_service")
def test_load_valid_model(
    mock_get_inference_service, mock_streamlit, mock_registry, mock_analytics
):
    from src.services.inference_service import ModelSchema

    run_data = {
        "run_id": "run123",
        "start_time": datetime(2023, 1, 1, 12, 0),
        "metrics.cv_mean_score": 0.99,
    }
    mock_registry.list_models.return_value = [run_data]

    # Construct expected label
    expected_label = "2023-01-01 12:00 | CV:0.9900 | ID:run123"

    # User selects the label
    mock_streamlit.selectbox.side_effect = [expected_label, "BaseSalary", 0.5]

    # Mock model and schema
    mock_forecaster = MagicMock()
    mock_schema = MagicMock(spec=ModelSchema)
    mock_schema.targets = ["BaseSalary"]
    mock_schema.quantiles = [0.5]

    mock_inference_service = mock_get_inference_service.return_value
    mock_inference_service.load_model.return_value = mock_forecaster
    mock_inference_service.get_model_schema.return_value = mock_schema

    # Mock Feature Importance
    df_imp = pd.DataFrame({"Feature": ["A"], "Gain": [10.0]})
    mock_analytics.get_feature_importance.return_value = df_imp

    render_model_analysis_ui()

    # Verify success loading
    mock_streamlit.success.assert_called()

    # Verify plotting
    mock_streamlit.pyplot.assert_called()


@patch("src.app.model_analysis.get_inference_service")
def test_empty_importance(
    mock_get_inference_service, mock_streamlit, mock_registry, mock_analytics
):
    from src.services.inference_service import ModelSchema

    run_data = {
        "run_id": "run123",
        "start_time": datetime(2023, 1, 1, 12, 0),
        "metrics.cv_mean_score": 0.99,
    }
    mock_registry.list_models.return_value = [run_data]

    mock_streamlit.selectbox.side_effect = [
        "2023-01-01 12:00 | CV:0.9900 | ID:run123",
        "BaseSalary",
        0.5,
    ]

    mock_forecaster = MagicMock()
    mock_schema = MagicMock(spec=ModelSchema)
    mock_schema.targets = ["BaseSalary"]
    mock_schema.quantiles = [0.5]

    mock_inference_service = mock_get_inference_service.return_value
    mock_inference_service.load_model.return_value = mock_forecaster
    mock_inference_service.get_model_schema.return_value = mock_schema

    # Return empty importance
    mock_analytics.get_feature_importance.return_value = pd.DataFrame()

    render_model_analysis_ui()

    mock_streamlit.warning.assert_called()
    assert "No feature importance scores found" in mock_streamlit.warning.call_args_list[-1][0][0]


def test_fmt_score_value_error(mock_streamlit, mock_registry):
    """Test fmt_score handles ValueError (non-numeric CV score)."""
    run_data = {
        "run_id": "run123",
        "start_time": datetime(2023, 1, 1, 12, 0),
        "metrics.cv_mean_score": "invalid",
    }
    mock_registry.list_models.return_value = [run_data]

    mock_streamlit.selectbox.return_value = None

    render_model_analysis_ui()

    assert mock_streamlit.selectbox.called


def test_fmt_score_type_error(mock_streamlit, mock_registry):
    """Test fmt_score handles TypeError (None CV score)."""
    run_data = {
        "run_id": "run123",
        "start_time": datetime(2023, 1, 1, 12, 0),
        "metrics.cv_mean_score": None,
    }
    mock_registry.list_models.return_value = [run_data]

    mock_streamlit.selectbox.return_value = None

    render_model_analysis_ui()

    assert mock_streamlit.selectbox.called


def test_empty_selected_label_returns_early(mock_streamlit, mock_registry):
    """Test that function returns early when selected_label is None."""
    run_data = {
        "run_id": "run123",
        "start_time": datetime(2023, 1, 1, 12, 0),
        "metrics.cv_mean_score": 0.99,
    }
    mock_registry.list_models.return_value = [run_data]

    mock_streamlit.selectbox.return_value = None

    render_model_analysis_ui()

    mock_registry.load_model.assert_not_called()


@patch("src.app.model_analysis.get_inference_service")
def test_no_targets_shows_error(
    mock_get_inference_service, mock_streamlit, mock_registry, mock_analytics
):
    """Test that function shows error and returns when no targets found."""
    from src.services.inference_service import ModelSchema

    run_data = {
        "run_id": "run123",
        "start_time": datetime(2023, 1, 1, 12, 0),
        "metrics.cv_mean_score": 0.99,
    }
    mock_registry.list_models.return_value = [run_data]

    expected_label = "2023-01-01 12:00 | CV:0.9900 | ID:run123"
    mock_streamlit.selectbox.return_value = expected_label

    mock_forecaster = MagicMock()
    mock_schema = MagicMock(spec=ModelSchema)
    mock_schema.targets = []  # No targets

    mock_inference_service = mock_get_inference_service.return_value
    mock_inference_service.load_model.return_value = mock_forecaster
    mock_inference_service.get_model_schema.return_value = mock_schema

    render_model_analysis_ui()

    mock_streamlit.error.assert_called_with(
        "This model file does not appear to contain trained models."
    )


@patch("src.app.model_analysis.get_inference_service")
def test_exception_handling_displays_traceback(
    mock_get_inference_service, mock_streamlit, mock_registry
):
    """Test that exceptions are caught and traceback is displayed."""
    run_data = {
        "run_id": "run123",
        "start_time": datetime(2023, 1, 1, 12, 0),
        "metrics.cv_mean_score": 0.99,
    }
    mock_registry.list_models.return_value = [run_data]

    expected_label = "2023-01-01 12:00 | CV:0.9900 | ID:run123"
    mock_streamlit.selectbox.return_value = expected_label

    mock_inference_service = mock_get_inference_service.return_value
    mock_inference_service.load_model.side_effect = ValueError("Test error")

    render_model_analysis_ui()

    mock_streamlit.error.assert_called()
    mock_streamlit.code.assert_called()
    call_args = mock_streamlit.code.call_args[0][0]
    assert "ValueError" in call_args or "Test error" in call_args
