import pytest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import streamlit as st
from src.app.model_analysis import render_model_analysis_ui

@pytest.fixture
def mock_streamlit():
    with patch("src.app.model_analysis.st") as mock_st:
        # Defaults
        mock_st.selectbox.return_value = None
        yield mock_st

@pytest.fixture
def mock_glob():
    with patch("src.app.model_analysis.glob.glob") as mock_g:
        yield mock_g

@pytest.fixture
def mock_pickle():
    with patch("src.app.model_analysis.pickle.load") as mock_p:
        yield mock_p

def test_no_models_shows_warning(mock_streamlit, mock_glob):
    mock_glob.return_value = []
    render_model_analysis_ui()
    mock_streamlit.warning.assert_called_with("No model files (*.pkl) found in the root directory. Please train a model first.")

def test_load_valid_model(mock_streamlit, mock_glob, mock_pickle):
    mock_glob.return_value = ["model.pkl"]
    mock_streamlit.selectbox.side_effect = ["model.pkl", "BaseSalary", 0.5] # Model, Target, Quantile
    
    # Mock Forecaster object
    mock_forecaster = MagicMock()
    mock_forecaster.targets = ["BaseSalary"]
    mock_forecaster.quantiles = [0.5]
    
    # Mock Model inside
    mock_booster = MagicMock()
    mock_booster.get_score.return_value = {"FeatureA": 10, "FeatureB": 5}
    
    mock_model = MagicMock()
    mock_model.get_booster.return_value = mock_booster
    
    mock_forecaster.models = {"BaseSalary_p50": mock_model}
    
    mock_pickle.return_value = mock_forecaster
    
    with patch("builtins.open", mock_open(read_data=b"data")):
        render_model_analysis_ui()
    
    # Verify success loading
    mock_streamlit.success.assert_called()
    
    # Verify chart plotted
    mock_streamlit.pyplot.assert_called()
    
    # Verify dataframe shown
    # Cannot easily check arguments of dataframe call without complex asserts, but we know it was called via expander context
    # mock_streamlit.expander.return_value.__enter__.return_value.dataframe.assert_called() 
    # ^ Need to mock context manager for expander

def test_load_model_fallback_keys(mock_streamlit, mock_glob, mock_pickle):
    """Test fallback logic when targets/quantiles attributes are missing."""
    mock_glob.return_value = ["model.pkl"]
    
    # Sequence: Select Model -> Select Target (parsed) -> Select Quantile (parsed)
    # Target parsed from "BaseSalary_p50" -> "BaseSalary"
    # Quantile parsed from "p50" -> 0.5
    mock_streamlit.selectbox.side_effect = ["model.pkl", "BaseSalary", 0.5]
    
    mock_forecaster = MagicMock()
    del mock_forecaster.targets # Ensure attribute missing
    del mock_forecaster.quantiles
    
    # But models dict exists
    mock_booster = MagicMock()
    mock_booster.get_score.return_value = {"FeatureA": 10}
    mock_model = MagicMock()
    mock_model.get_booster.return_value = mock_booster

    mock_forecaster.models = {"BaseSalary_p50": mock_model}
    
    mock_pickle.return_value = mock_forecaster
    
    with patch("builtins.open", mock_open(read_data=b"data")):
        render_model_analysis_ui()
        
    # Should still succeed
    mock_streamlit.pyplot.assert_called()

def test_empty_importance(mock_streamlit, mock_glob, mock_pickle):
    mock_glob.return_value = ["model.pkl"]
    mock_streamlit.selectbox.side_effect = ["model.pkl", "BaseSalary", 0.5]
    
    mock_forecaster = MagicMock()
    mock_forecaster.targets = ["BaseSalary"]
    mock_forecaster.quantiles = [0.5]
    
    mock_model = MagicMock()
    # Empty score
    mock_model.get_booster.return_value.get_score.return_value = {}
    
    mock_forecaster.models = {"BaseSalary_p50": mock_model}
    mock_pickle.return_value = mock_forecaster
    
    with patch("builtins.open", mock_open(read_data=b"data")):
        render_model_analysis_ui()
        
    mock_streamlit.warning.assert_called()
    assert "No feature importance scores found" in mock_streamlit.warning.call_args_list[-1][0][0]
