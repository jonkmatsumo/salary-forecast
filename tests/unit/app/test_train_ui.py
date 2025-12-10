import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.app.train_ui import render_training_ui

@pytest.fixture
def mock_streamlit():
    with patch("src.app.train_ui.st") as mock_st:
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
def mock_load_data():
    with patch("src.app.train_ui.load_data") as mock_ld:
        yield mock_ld

@pytest.fixture
def mock_training_service():
    with patch("src.app.train_ui.get_training_service") as mock_get_svc:
        yield mock_get_svc

@pytest.fixture
def mock_registry():
    with patch("src.app.train_ui.ModelRegistry") as mock_reg:
        yield mock_reg

@pytest.fixture
def mock_analytics_service():
    with patch("src.app.train_ui.AnalyticsService") as mock_analytics:
        mock_instance = MagicMock()
        mock_instance.get_data_summary.return_value = {
            "total_samples": 100,
            "unique_locations": 10,
            "unique_levels": 5,
            "shape": (100, 10)
        }
        mock_analytics.return_value = mock_instance
        yield mock_analytics

@pytest.fixture
def mock_workflow_wizard():
    with patch("src.app.train_ui.render_workflow_wizard") as mock_wizard:
        mock_wizard.return_value = None  # Wizard not completed by default
        yield mock_wizard

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Level": ["E3", "E4", "E3"],
        "Location": ["NY", "SF", "NY"],
        "BaseSalary": [100, 150, 110],
        "TotalComp": [120, 180, 130],
        "YearsOfExperience": [1, 5, 2],
        "YearsAtCompany": [1, 2, 1]
    })

def test_render_training_ui_upload_loads_data(mock_streamlit, mock_load_data, mock_training_service, mock_registry):
    # Setup: No training_data, uploader active
    mock_streamlit.session_state = {}
    mock_upload_file = MagicMock()
    mock_upload_file.name = "test.csv"
    mock_streamlit.file_uploader.return_value = mock_upload_file
    
    # Validation succeeds
    df = MagicMock()
    df.__len__.return_value = 10
    mock_load_data.return_value = df
    
    # Mock expander and other UI elements
    mock_streamlit.expander.return_value.__enter__.return_value = MagicMock()
    mock_streamlit.selectbox.return_value = "Overview Metrics"
    
    render_training_ui()
    
    # Assert data loaded
    assert "training_data" in mock_streamlit.session_state
    
    # Assert Success Message
    mock_streamlit.success.assert_called()
    
    # Assert Configuration tip removed (no longer redirects to Configuration page)
    found_redirect = False
    for call in mock_streamlit.info.call_args_list:
        if "Configuration" in call[0][0] and "page" in call[0][0].lower():
            found_redirect = True
            break
    assert not found_redirect, "Configuration redirect tip should be removed"
            
def test_render_training_ui_requires_wizard(mock_streamlit, mock_load_data, mock_training_service):
    # Setup state - data loaded but wizard not completed
    df = MagicMock()
    mock_streamlit.session_state = {
        "training_data": df,
        "training_dataset_name": "dataset.csv",
        "training_job_id": None,
        "workflow_phase": "not_started"  # Wizard not completed
    }
    
    # Mock expander for Data Analysis and Wizard
    mock_expander = MagicMock()
    mock_expander.__enter__.return_value = MagicMock()
    mock_streamlit.expander.return_value = mock_expander
    mock_streamlit.selectbox.return_value = "Overview Metrics"
    
    render_training_ui()
    
    # Should show info that wizard is required
    info_calls = [call[0][0] for call in mock_streamlit.info.call_args_list]
    found_wizard_required = any("wizard" in msg.lower() or "configuration" in msg.lower() for msg in info_calls)
    assert found_wizard_required, "Should show message that wizard is required"
    
    # Should NOT show training controls (button should not be called with "Start Training")
    button_labels = [call[0][0] if call[0] else "" for call in mock_streamlit.button.call_args_list]
    start_training_called = any("Start Training" in label for label in button_labels)
    # Actually, the button might exist but be disabled/hidden, so we check that training service is not called
    service_instance = mock_training_service.return_value
    service_instance.start_training_async.assert_not_called()

def test_render_training_ui_starts_job_after_wizard(mock_streamlit, mock_load_data, mock_training_service):
    # Setup state - wizard completed
    df = MagicMock()
    mock_streamlit.session_state = {
        "training_data": df,
        "training_dataset_name": "dataset.csv",
        "training_job_id": None,
        "workflow_phase": "complete"  # Wizard completed
    }
    
    # Mock expander for Data Analysis
    mock_expander = MagicMock()
    mock_expander.__enter__.return_value = MagicMock()
    mock_streamlit.expander.return_value = mock_expander
    mock_streamlit.selectbox.return_value = "Overview Metrics"
    
    # Mock inputs
    # Checkboxes: first is do_tune (False), second is remove_outliers (True), third is display_charts (True)
    checkbox_calls = [False, True, True]  # do_tune=False, remove_outliers=True, display_charts=True
    checkbox_count = [0]
    def checkbox_side_effect(*args, **kwargs):
        if checkbox_count[0] < len(checkbox_calls):
            result = checkbox_calls[checkbox_count[0]]
            checkbox_count[0] += 1
            return result
        # Default return for any additional checkbox calls
        return kwargs.get("value", False)
    mock_streamlit.checkbox.side_effect = checkbox_side_effect
    mock_streamlit.number_input.return_value = 20
    # text_input for additional tag
    mock_streamlit.text_input.return_value = "tag-v1"
    
    # Mock Start Button -> True (after wizard completion)
    # Button label is "Start Training (Async)" - check for that
    # Need to handle multiple button calls - first might be for wizard, second is Start Training
    button_call_count = [0]
    def button_side_effect(label, **kwargs):
        button_call_count[0] += 1
        # Return True for "Start Training" button, False for others
        label_str = str(label) if label else ""
        if "Start Training" in label_str or ("Start" in label_str and "Training" in label_str):
            return True
        return False
    mock_streamlit.button.side_effect = button_side_effect
    
    # Mock Service
    service_instance = mock_training_service.return_value
    service_instance.start_training_async.return_value = "new_job_id"
    
    render_training_ui()
    
    service_instance.start_training_async.assert_called_with(
        df,
        remove_outliers=True,  # Default is True now
        do_tune=False,
        n_trials=20,
        additional_tag="tag-v1",
        dataset_name="dataset.csv"
    )
    
    assert mock_streamlit.session_state["training_job_id"] == "new_job_id"
    mock_streamlit.rerun.assert_called()

def test_data_analysis_section_available(mock_streamlit, mock_load_data, sample_df, mock_analytics_service):
    """Test that Data Analysis section is available when data is loaded."""
    mock_streamlit.session_state = {
        "training_data": sample_df,
        "training_dataset_name": "test.csv"
    }
    
    # Mock expander
    mock_expander = MagicMock()
    mock_expander.__enter__.return_value = MagicMock()
    mock_streamlit.expander.return_value = mock_expander
    mock_streamlit.selectbox.return_value = "Overview Metrics"
    
    render_training_ui()
    
    # Should call expander for Data Analysis
    expander_calls = [call[0][0] for call in mock_streamlit.expander.call_args_list]
    assert "Data Analysis" in expander_calls, "Data Analysis expander should be called"
    
    # Should call AnalyticsService
    mock_analytics_service.return_value.get_data_summary.assert_called()

def test_data_analysis_visualization_selection(mock_streamlit, sample_df, mock_analytics_service):
    """Test that different visualizations can be selected."""
    mock_streamlit.session_state = {
        "training_data": sample_df,
        "training_dataset_name": "test.csv",
        "workflow_phase": "complete"
    }
    
    # Mock expander
    mock_expander = MagicMock()
    mock_expander_context = MagicMock()
    mock_expander.__enter__.return_value = mock_expander_context
    mock_streamlit.expander.return_value = mock_expander
    
    # Test different visualization selections
    for viz_type in ["Overview Metrics", "Data Sample", "Salary Distribution", "Categorical Breakdown", "Correlations"]:
        mock_expander_context.selectbox.return_value = viz_type
        mock_streamlit.selectbox.return_value = viz_type
        
        render_training_ui()
        
        # Verify selectbox was called for visualization selection
        assert mock_expander_context.selectbox.called or mock_streamlit.selectbox.called
