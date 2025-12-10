import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.services.training_service import TrainingService
from src.xgboost.model import SalaryForecaster

class TestTrainingService(unittest.TestCase):
    def setUp(self):
        self.service = TrainingService()
        self.df = pd.DataFrame({"col": [1, 2, 3]})

    @patch("src.services.training_service.SalaryForecaster")
    def test_train_model(self, MockForecaster):
        # Setup mock instance
        mock_instance = MockForecaster.return_value
        
        callback = MagicMock()
        
        model = self.service.train_model(self.df, remove_outliers=True, callback=callback)
        
        # Verify Forecaster was instantiated
        MockForecaster.assert_called_once()
        
        # Verify train was called
        mock_instance.train.assert_called_with(self.df, callback=callback, remove_outliers=True)
        
        # Verify callback initial call
        callback.assert_any_call("Starting training...", None)
        
        self.assertEqual(model, mock_instance)

    @patch("src.services.training_service.SalaryForecaster")
    def test_start_training_async(self, MockForecaster):
        job_id = self.service.start_training_async(self.df)
        self.assertIsNotNone(job_id)
        
        status = self.service.get_job_status(job_id)
        self.assertIsNotNone(status)
        self.assertIn(status["status"], ["QUEUED", "RUNNING", "COMPLETED"])
        self.assertIn(status["status"], ["QUEUED", "RUNNING", "COMPLETED"])
        self.assertIsInstance(status["history"], list)

    def test_run_async_job_logs_tags(self):
        import src.services.training_service as ts
        
        with patch.object(ts, "mlflow") as mock_mlflow, \
             patch.object(ts, "SalaryForecaster") as MockForecaster:
            
            job_id = "test_job"
            self.service._jobs[job_id] = {
                "status": "QUEUED",
                "logs": [], "history": [], "scores": [],
                "result": None
            }
            
            # Mock Context Manager for start_run
            mock_run = MagicMock()
            mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
            
            self.service._run_async_job(
                job_id, self.df, True, False, 10, 
                additional_tag="experimental",
                dataset_name="test.csv"
            )
            
            mock_mlflow.set_tags.assert_called_with({
                "model_type": "XGBoost",
                "dataset_name": "test.csv",
                "additional_tag": "experimental"
            })
            
            # Verify run_name logic
            mock_mlflow.start_run.assert_called_with(run_name="experimental")
            
            # Verify MockForecaster was used
            MockForecaster.return_value.train.assert_called()

    def test_get_job_status_invalid(self):
        status = self.service.get_job_status("invalid_id")
        self.assertIsNone(status)
    
    @patch("src.services.training_service.SalaryForecaster")
    def test_train_with_optional_encodings(self, MockForecaster):
        """Test training with optional encodings in config."""
        mock_instance = MockForecaster.return_value
        
        # Create config with optional encodings
        config = {
            "model": {
                "targets": ["Salary"],
                "features": [{"name": "Location_CostOfLiving", "monotone_constraint": 0}],
                "quantiles": [0.5]
            },
            "optional_encodings": {
                "Location": {"type": "cost_of_living", "params": {}}
            }
        }
        
        df = pd.DataFrame({
            "Location": ["New York", "San Francisco"],
            "Salary": [100000, 150000]
        })
        
        model = self.service.train_model(df, config=config)
        
        # Verify Forecaster was instantiated with config
        MockForecaster.assert_called_once()
        call_kwargs = MockForecaster.call_args[1]
        self.assertIn("optional_encodings", call_kwargs["config"])
        self.assertEqual(call_kwargs["config"]["optional_encodings"]["Location"]["type"], "cost_of_living")
        
        # Verify train was called
        mock_instance.train.assert_called_once()
    
    @patch("src.services.training_service.SalaryForecaster")
    def test_train_without_optional_encodings(self, MockForecaster):
        """Test training without optional encodings (backward compatibility)."""
        mock_instance = MockForecaster.return_value
        
        # Create config without optional encodings
        config = {
            "model": {
                "targets": ["Salary"],
                "features": [{"name": "Level", "monotone_constraint": 1}],
                "quantiles": [0.5]
            }
            # No optional_encodings field
        }
        
        df = pd.DataFrame({
            "Level": ["L3", "L4"],
            "Salary": [100000, 150000]
        })
        
        model = self.service.train_model(df, config=config)
        
        # Should still work
        MockForecaster.assert_called_once()
        mock_instance.train.assert_called_once()