import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd
import json
import io
from src.cli.inference_cli import collect_user_data, main

def test_collect_user_data():
    # Mock input to return: E5, New York, 5, 2
    inputs = ["E5", "New York", "5", "2"]
    with patch('builtins.input', side_effect=inputs):
        df = collect_user_data()
        
        assert len(df) == 1
        assert df.iloc[0]["Level"] == "E5"
        assert df.iloc[0]["Location"] == "New York"
        assert df.iloc[0]["YearsOfExperience"] == 5
        assert df.iloc[0]["YearsAtCompany"] == 2

def test_collect_user_data_invalid_level():
    # Mock input: Invalid, then E5, New York, 5, 2
    with patch('builtins.input', side_effect=["Invalid", "E5", "New York", "5", "2"]):
        df = collect_user_data()
        assert df.iloc[0]["Level"] == "E5"

@pytest.fixture
def mock_model():
    m = MagicMock()
    m.quantiles = [0.5]
    m.predict.return_value = {"BaseSalary": {"p50": [120000]}}
    return m

@pytest.fixture
def mock_input():
    with patch("builtins.input") as m:
        yield m

@pytest.fixture
def mock_console():
    with patch("src.cli.inference_cli.Console") as m:
        yield m.return_value

@patch("src.cli.inference_cli.ModelRegistry")
def test_main_interactive(MockRegistry, mock_model, mock_input, mock_console):
    # Setup mocks
    mock_registry_instance = MockRegistry.return_value
    mock_registry_instance.list_models.return_value = [{"run_id": "run123", "start_time": datetime.now(), "metrics.cv_mean_score": 0.9}]
    mock_registry_instance.load_model.return_value = mock_model
    
    # Run with empty args
    with patch("sys.argv", ["script_name"]):
        # Mock interactive inputs:
        # 1. select_model -> "1"
        # 2. collect_user_data -> "E5", "New York", "5", "2"
        # 3. loop break -> "n"
        mock_input.side_effect = ["1", "E5", "New York", "5", "2", "n"]
        main()
        
    mock_registry_instance.load_model.assert_called_with("run123")
    mock_model.predict.assert_called()

@patch("src.cli.inference_cli.ModelRegistry")
def test_main_non_interactive_json(MockRegistry, mock_model, capsys):
    mock_registry_instance = MockRegistry.return_value
    mock_registry_instance.load_model.return_value = mock_model
    
    with patch("sys.argv", ["script_name", "--run-id", "runAUTO", "--level", "E5", "--location", "NY", "--yoe", "5", "--yac", "2", "--json"]):
        main()
        
    mock_registry_instance.load_model.assert_called_with("runAUTO")
    captured = capsys.readouterr()
    assert '"BaseSalary":' in captured.out
    
@patch("src.cli.inference_cli.ModelRegistry")
def test_main_partial_args_error(MockRegistry, mock_console):
    # Should exit if partial non-interactive args
    mock_registry_instance = MockRegistry.return_value
    mock_registry_instance.list_models.return_value = []
    
    with patch("sys.argv", ["script_name", "--level", "E5"]): # Missing others
         with pytest.raises(SystemExit):
             main()
    mock_console.print.assert_called()
