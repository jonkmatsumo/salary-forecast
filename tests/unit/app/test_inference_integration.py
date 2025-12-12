"""
Integration tests for Inference tab features including Model Analysis.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.app.inference_ui import render_inference_ui, render_model_information


@pytest.fixture
def mock_streamlit():
    with patch("src.app.inference_ui.st") as mock_st:
        mock_st.session_state = {}

        # Mock columns to return proper number of mock objects
        def columns_side_effect(n):
            if isinstance(n, int):
                return [MagicMock() for _ in range(n)]
            elif isinstance(n, list):
                return [MagicMock() for _ in range(len(n))]
            return []

        mock_st.columns.side_effect = columns_side_effect
        yield mock_st


@pytest.fixture
def mock_registry():
    with patch("src.app.inference_ui.ModelRegistry") as MockReg:
        mock_instance = MagicMock()
        run_data = {
            "run_id": "test_run_123",
            "start_time": pd.Timestamp("2023-01-01 12:00:00"),
            "tags.model_type": "XGBoost",
            "tags.dataset_name": "test_dataset",
            "metrics.cv_mean_score": 0.95,
            "tags.additional_tag": "test_tag",
        }
        mock_instance.list_models.return_value = [run_data]
        MockReg.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_forecaster():
    forecaster = MagicMock()
    forecaster.ranked_encoders = {"Level": MagicMock()}
    forecaster.ranked_encoders["Level"].mapping = {"E3": 1, "E4": 2}
    forecaster.proximity_encoders = {"Location": MagicMock()}
    forecaster.feature_names = ["Level", "Location", "YearsOfExperience"]
    return forecaster


@pytest.fixture
def mock_analytics_service():
    with patch("src.app.inference_ui.AnalyticsService") as MockAn:
        mock_instance = MagicMock()
        mock_instance.get_available_targets.return_value = ["BaseSalary", "TotalComp"]
        mock_instance.get_available_quantiles.return_value = [0.1, 0.5, 0.9]
        mock_instance.get_feature_importance.return_value = pd.DataFrame(
            {"Feature": ["Level", "Location", "YearsOfExperience"], "Gain": [10.0, 8.0, 5.0]}
        )
        MockAn.return_value = mock_instance
        yield mock_instance


def test_model_information_display(mock_streamlit, mock_registry, mock_forecaster):
    """Test that model information is displayed correctly."""
    runs = [
        {
            "run_id": "test_run_123",
            "start_time": pd.Timestamp("2023-01-01 12:00:00"),
            "tags.model_type": "XGBoost",
            "tags.dataset_name": "test_dataset",
            "metrics.cv_mean_score": 0.95,
        }
    ]

    # Mock expander
    mock_expander = MagicMock()
    mock_expander.__enter__.return_value = MagicMock()
    mock_expander.__exit__.return_value = None
    mock_streamlit.expander.return_value = mock_expander

    render_model_information(mock_forecaster, "test_run_123", runs)

    # Should call expander for Model Metadata and Feature Information
    expander_calls = [call[0][0] for call in mock_streamlit.expander.call_args_list]
    assert "Model Metadata" in expander_calls
    assert "Feature Information" in expander_calls


def test_model_analysis_section(
    mock_streamlit, mock_registry, mock_forecaster, mock_analytics_service
):
    """Test that Model Analysis section appears in Inference tab."""
    mock_streamlit.session_state = {"forecaster": mock_forecaster, "current_run_id": "test_run_123"}

    # Mock registry
    runs = [
        {
            "run_id": "test_run_123",
            "start_time": pd.Timestamp("2023-01-01 12:00:00"),
            "tags.model_type": "XGBoost",
            "tags.dataset_name": "test_dataset",
            "metrics.cv_mean_score": 0.95,
        }
    ]
    mock_registry.list_models.return_value = runs

    # Mock selectbox for model selection and analysis
    # Need to handle multiple calls - use a function that returns values in order
    selectbox_calls = [
        "2023-01-01 12:00 | XGBoost | test_dataset | CV:0.9500 | ID:test_run",  # Model selection
        "Feature Importance",  # Visualization selection
        "BaseSalary",  # Target selection
        0.5,  # Quantile selection
    ]
    call_count = [0]

    def selectbox_side_effect(*args, **kwargs):
        if call_count[0] < len(selectbox_calls):
            result = selectbox_calls[call_count[0]]
            call_count[0] += 1
            return result
        return None

    mock_streamlit.selectbox.side_effect = selectbox_side_effect

    # Mock expander
    mock_expander = MagicMock()
    mock_expander_context = MagicMock()
    mock_expander.__enter__.return_value = mock_expander_context
    mock_expander.__exit__.return_value = None
    mock_streamlit.expander.return_value = mock_expander

    # Mock form and form_submit_button to prevent form submission
    mock_form = MagicMock()
    mock_streamlit.form.return_value.__enter__ = MagicMock(return_value=mock_form)
    mock_streamlit.form.return_value.__exit__ = MagicMock(return_value=None)
    mock_streamlit.form_submit_button.return_value = False

    # Mock selectbox and text_input for form inputs
    mock_streamlit.text_input.return_value = "New York"
    mock_streamlit.number_input.return_value = 5

    render_inference_ui()

    # Should call expander for Model Analysis
    expander_calls = [call[0][0] for call in mock_streamlit.expander.call_args_list]
    assert "Model Analysis" in expander_calls, "Model Analysis expander should be called"

    # Should call AnalyticsService methods
    mock_analytics_service.get_available_targets.assert_called()
    mock_analytics_service.get_available_quantiles.assert_called()


def test_model_analysis_feature_importance(
    mock_streamlit, mock_registry, mock_forecaster, mock_analytics_service
):
    """Test feature importance visualization in Model Analysis."""
    mock_streamlit.session_state = {"forecaster": mock_forecaster, "current_run_id": "test_run_123"}

    runs = [
        {
            "run_id": "test_run_123",
            "start_time": pd.Timestamp("2023-01-01 12:00:00"),
            "tags.model_type": "XGBoost",
            "tags.dataset_name": "test_dataset",
            "metrics.cv_mean_score": 0.95,
        }
    ]
    mock_registry.list_models.return_value = runs

    # Mock UI elements
    selectbox_calls = [
        "2023-01-01 12:00 | XGBoost | test_dataset | CV:0.9500 | ID:test_run",
        "Feature Importance",
        "BaseSalary",
        0.5,
    ]
    call_count = [0]

    def selectbox_side_effect(*args, **kwargs):
        if call_count[0] < len(selectbox_calls):
            result = selectbox_calls[call_count[0]]
            call_count[0] += 1
            return result
        return None

    mock_streamlit.selectbox.side_effect = selectbox_side_effect

    mock_expander = MagicMock()
    mock_expander_context = MagicMock()
    mock_expander.__enter__.return_value = mock_expander_context
    mock_expander.__exit__.return_value = None
    mock_streamlit.expander.return_value = mock_expander

    # Mock form and form_submit_button to prevent form submission
    mock_form = MagicMock()
    mock_streamlit.form.return_value.__enter__ = MagicMock(return_value=mock_form)
    mock_streamlit.form.return_value.__exit__ = MagicMock(return_value=None)
    mock_streamlit.form_submit_button.return_value = False

    # Mock selectbox and text_input for form inputs
    mock_streamlit.text_input.return_value = "New York"
    mock_streamlit.number_input.return_value = 5

    render_inference_ui()

    # Should call get_feature_importance
    mock_analytics_service.get_feature_importance.assert_called_with(
        mock_forecaster, "BaseSalary", 0.5
    )

    # Should display dataframe and plot
    mock_streamlit.dataframe.assert_called()
    mock_streamlit.pyplot.assert_called()
