import pytest
from unittest.mock import patch, MagicMock
from src.cli.train_cli import main, train_workflow

def test_train_cli_main():
    # Mock inputs: 
    # 1. input.csv (CSV)
    # 2. config.json (Config)
    # 3. model.pkl (Output)
    inputs = ["input.csv", "config.json", "model.pkl"]
    
    with patch('src.cli.train_cli.Console') as MockConsole, \
         patch('os.path.exists', return_value=True), \
         patch('src.cli.train_cli.train_workflow') as mock_train_workflow:
        
        mock_console = MockConsole.return_value
        mock_console.input.side_effect = inputs
        
        main()
        
        # Verify train_workflow was called with correct args
        # Note: we also pass console now
        mock_train_workflow.assert_called_once()
        args, kwargs = mock_train_workflow.call_args
        assert args[0] == "input.csv"
        assert args[1] == "config.json"
        assert args[2] == "model.pkl"

def test_train_cli_defaults():
    # Mock inputs: Enter (default), Enter (default), Enter (default)
    inputs = ["", "", ""]
    
    with patch('src.cli.train_cli.Console') as MockConsole:
        mock_console = MockConsole.return_value
        # Mock input method of console
        mock_console.input.side_effect = inputs
        
        with patch('os.path.exists', return_value=True), \
             patch('src.cli.train_cli.train_workflow') as mock_train_workflow:
            
            main()
            
            # Verify defaults
            mock_train_workflow.assert_called_once()
            args, kwargs = mock_train_workflow.call_args
            assert args[0] == "salaries-list.csv"
            assert args[1] == "config.json"
            assert args[2] == "salary_model.pkl"

def test_train_cli_file_not_found():
    # Mock inputs: missing.csv, ...
    inputs = ["missing.csv", "config.json", "model.pkl"]
    
    with patch('src.cli.train_cli.Console') as MockConsole:
        mock_console = MockConsole.return_value
        mock_console.input.side_effect = inputs
        
        # Mock os.path.exists to return False for the first file
        with patch('os.path.exists', side_effect=[False, True]): 
            
            main()
            
            # Verify error message printed
            # Check if any print call contains "Error"
            error_printed = any("Error" in str(call) for call in mock_console.print.call_args_list)
            assert error_printed

def test_train_workflow():
    # Mock dependencies
    with patch('src.cli.train_cli.load_data') as mock_load_data, \
         patch('src.cli.train_cli.SalaryForecaster') as MockForecaster, \
         patch('src.cli.train_cli.load_config') as mock_load_config, \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', new_callable=MagicMock), \
         patch('pickle.dump') as mock_pickle_dump: # Mock pickle dump
        
        # Setup mocks
        mock_df = MagicMock()
        mock_df.__len__.return_value = 100 # Simulate loaded data
        mock_load_data.return_value = mock_df
        
        mock_model = MockForecaster.return_value
        # Mock predict to return dictionary logic for output printing
        mock_model.quantiles = [0.10, 0.50, 0.90]
        mock_model.predict.return_value = {
            "BaseSalary": {"p10": [150000], "p50": [200000], "p90": [250000]}
        }

        mock_console = MagicMock()

        # Run train_workflow
        train_workflow(csv_path="test.csv", config_path="test_config.json", output_path="test_model.pkl", console=mock_console)
        
        # Verify interactions
        mock_load_data.assert_called_once_with("test.csv")
        MockForecaster.assert_called_once()
        mock_model.train.assert_called_once_with(mock_df, console=mock_console)
        
        # Verify pickle.dump was called
        mock_pickle_dump.assert_called_once()
        
        # Verify inference was attempted (predict called)
        assert mock_model.predict.call_count >= 1

def test_train_workflow_calls_load_config():
    with patch('src.cli.train_cli.load_config') as mock_load_config, \
         patch('src.cli.train_cli.load_data'), \
         patch('src.cli.train_cli.SalaryForecaster'), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open'), \
         patch('pickle.dump'):
        
        mock_console = MagicMock()
        train_workflow(csv_path="test.csv", config_path="test_config.json", output_path="model.pkl", console=mock_console)
        
        mock_load_config.assert_called_once_with("test_config.json")
