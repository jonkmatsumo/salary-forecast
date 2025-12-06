import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.cli import collect_user_data, main

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
    
    with patch('src.cli.load_model', return_value=mock_model), \
         patch('builtins.input', side_effect=inputs), \
         patch('src.cli.Console') as MockConsole:
        
        mock_console_instance = MockConsole.return_value
        
        main()
        
        # Verify predict was called
        mock_model.predict.assert_called_once()
        
        # Verify Console was instantiated
        MockConsole.assert_called_once()
        
        # Verify print was called on the console instance
        # We expect multiple calls: welcome, calculating, table, goodbye
        assert mock_console_instance.print.call_count >= 1
        
        # Verify that a Table object was printed
        # One of the calls to print should have a Table argument
        table_printed = False
        for call in mock_console_instance.print.call_args_list:
            args, _ = call
            if len(args) > 0 and "Table" in str(type(args[0])):
                table_printed = True
                break
        
        # Note: In the test environment, checking the type directly might be tricky if imports differ,
        # but checking the string representation of the type usually works or checking if it has 'add_row'.
        # Actually, let's just check if any arg has a 'title' attribute equal to "Prediction Results"
        # or just rely on the fact that we passed *something* to print.
        
        # A more robust check:
        # Check if any call argument was a Table. 
        # Since we mock Console, we can't easily check isinstance(arg, Table) unless we import Table here too.
        from rich.table import Table
        table_printed = any(isinstance(call.args[0], Table) for call in mock_console_instance.print.call_args_list)
        assert table_printed, "A rich Table should have been printed"
