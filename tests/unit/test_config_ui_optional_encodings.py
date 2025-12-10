"""Tests for optional encodings in phase 2 of the workflow."""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.app.config_ui import _render_encoding_phase
from src.services.workflow_service import WorkflowService


@pytest.fixture
def mock_workflow_service():
    """Create a mock workflow service."""
    service = MagicMock(spec=WorkflowService)
    service.workflow = MagicMock()
    service.workflow.current_state = {
        "optional_encodings": {},
        "column_types": {"Location": "location", "Date": "datetime"}
    }
    return service


@pytest.fixture
def sample_encoding_result():
    """Sample encoding phase result."""
    return {
        "data": {
            "encodings": {
                "Location": {"type": "proximity", "reasoning": "Location column"},
                "Date": {"type": "numeric", "reasoning": "Date column"}
            },
            "summary": "Encoding summary"
        }
    }


def test_optional_encodings_shown_for_location_columns(mock_workflow_service, sample_encoding_result):
    """Test that optional encodings are shown for location columns in phase 2."""
    with patch("src.app.config_ui.st") as mock_st:
        mock_st.session_state = {"training_data": pd.DataFrame({
            "Location": ["New York", "Austin"],
            "Date": pd.to_datetime(["2023-01-01", "2023-02-01"])
        })}
        
        # Mock data editor to return encoding table
        mock_enc_df = pd.DataFrame([
            {"Column": "Location", "Encoding": "proximity", "Mapping": "", "Notes": ""}
        ])
        mock_st.data_editor.return_value = mock_enc_df
        
        # Mock selectbox for optional encoding
        mock_st.selectbox.return_value = "None"
        
        # Mock columns for action buttons
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_col3 = MagicMock()
        mock_st.columns.return_value = (mock_col1, mock_col2, mock_col3)
        mock_st.button.return_value = False
        
        result = _render_encoding_phase(mock_workflow_service, sample_encoding_result)
        
        # Verify that selectbox was called for location encoding
        selectbox_calls = [call for call in mock_st.selectbox.call_args_list 
                          if "Location" in str(call)]
        assert len(selectbox_calls) > 0, "Optional encoding selectbox should be shown for location columns"


def test_optional_encodings_shown_for_date_columns(mock_workflow_service, sample_encoding_result):
    """Test that optional encodings are shown for date columns in phase 2."""
    with patch("src.app.config_ui.st") as mock_st:
        mock_st.session_state = {"training_data": pd.DataFrame({
            "Location": ["New York", "Austin"],
            "Date": pd.to_datetime(["2023-01-01", "2023-02-01"])
        })}
        
        # Mock data editor to return encoding table
        mock_enc_df = pd.DataFrame([
            {"Column": "Date", "Encoding": "numeric", "Mapping": "", "Notes": ""}
        ])
        mock_st.data_editor.return_value = mock_enc_df
        
        # Mock selectbox for optional encoding
        mock_st.selectbox.return_value = "None"
        
        # Mock columns for action buttons
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_col3 = MagicMock()
        mock_st.columns.return_value = (mock_col1, mock_col2, mock_col3)
        mock_st.button.return_value = False
        
        result = _render_encoding_phase(mock_workflow_service, sample_encoding_result)
        
        # Verify that selectbox was called for date encoding
        selectbox_calls = [call for call in mock_st.selectbox.call_args_list 
                          if "Date" in str(call) and "opt_enc_date" in str(call)]
        assert len(selectbox_calls) > 0, "Optional encoding selectbox should be shown for date columns"


def test_optional_encodings_not_in_classification_phase():
    """Test that optional encodings are NOT shown in classification phase."""
    # This test verifies that the classification phase doesn't have optional encodings
    # The actual implementation should have removed the optional encodings section
    # We can verify this by checking that the classification phase UI doesn't include
    # the optional encodings expanders
    
    # Import the classification phase function
    from src.app.config_ui import _render_classification_phase
    
    with patch("src.app.config_ui.st") as mock_st:
        mock_service = MagicMock()
        mock_service.workflow = MagicMock()
        mock_service.workflow.current_state = {
            "column_types": {"Location": "location"}
        }
        
        result = {
            "data": {
                "targets": ["Salary"],
                "features": ["Level"],
                "ignore": []
            }
        }
        
        df = pd.DataFrame({
            "Location": ["New York"],
            "Salary": [100000]
        })
        
        # Mock data editor
        mock_st.data_editor.return_value = pd.DataFrame([
            {"Column": "Location", "Role": "Feature", "Dtype": "string (location)"}
        ])
        
        # Mock columns for action buttons
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_col3 = MagicMock()
        mock_st.columns.return_value = (mock_col1, mock_col2, mock_col3)
        mock_st.button.return_value = False
        
        _render_classification_phase(mock_service, result, df)
        
        # Verify that no optional encoding expanders were created
        expander_calls = [call for call in mock_st.expander.call_args_list 
                         if "Encodings" in str(call)]
        assert len(expander_calls) == 0, "Optional encodings should not appear in classification phase"

