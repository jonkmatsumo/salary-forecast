import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from cli import collect_user_data, main

def test_collect_user_data():
    # Mock input to return: E5, New York, 5, 2
    with patch('builtins.input', side_effect=["E5", "New York", "5", "2"]):
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

def test_main_flow():
    # Mock load_model to return a mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = {
        "BaseSalary": {"p25": [100000], "p50": [120000], "p75": [140000]}
    }
    
    # Mock inputs: 
    # 1. E5 (Level)
    # 2. New York (Location)
    # 3. 5 (YOE)
    # 4. 2 (YAC)
    # 5. n (Stop)
    inputs = ["E5", "New York", "5", "2", "n"]
    
    with patch('cli.load_model', return_value=mock_model), \
         patch('builtins.input', side_effect=inputs), \
         patch('builtins.print') as mock_print:
        
        main()
        
        # Verify predict was called
        mock_model.predict.assert_called_once()
        
        # Verify output contains expected strings (checking args passed to print)
        # We can just check if "BaseSalary" was printed
        printed_text = " ".join([str(call.args[0]) for call in mock_print.call_args_list])
        assert "BaseSalary" in printed_text
        assert "$120,000" in printed_text
