import pytest
from unittest.mock import patch, MagicMock
from src.app.app import render_training_ui

@pytest.fixture
def mock_streamlit():
    with patch("src.app.app.st") as mock_st:
        mock_st.session_state = {}
        yield mock_st

@pytest.fixture
def mock_load_data():
    with patch("src.app.app.load_data") as mock_ld:
        yield mock_ld

@pytest.fixture
def mock_training_service():
    with patch("src.app.app.get_training_service") as mock_get_svc:
        yield mock_get_svc

@pytest.fixture
def mock_registry():
    with patch("src.app.app.ModelRegistry") as mock_reg:
        yield mock_reg

def test_render_training_ui_upload_redirect(mock_streamlit, mock_load_data, mock_training_service, mock_registry):
    # Setup: No training_data, uploader active
    mock_streamlit.session_state = {}
    mock_upload_file = MagicMock()
    mock_streamlit.file_uploader.return_value = mock_upload_file
    
    # Validation succeeds
    df = MagicMock()
    df.__len__.return_value = 10
    mock_load_data.return_value = df
    
    render_training_ui()
    
    # Assert data loaded
    assert "training_data" in mock_streamlit.session_state
    
    # Assert Success Message
    mock_streamlit.success.assert_called()
    
    # Assert Redirect Info (Tip)
    found_redirect = False
    for call in mock_streamlit.info.call_args_list:
        if "Tip" in call[0][0] and "Configuration" in call[0][0]:
            found_redirect = True
            break
    assert found_redirect, "Redirect tip message not found in Training UI"
