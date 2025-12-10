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
