import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from src.services.model_registry import ModelRegistry


class TestModelRegistry(unittest.TestCase):
    @patch("src.services.model_registry.mlflow")
    @patch("src.services.model_registry.MlflowClient")
    def setUp(self, MockClient, mock_mlflow):
        self.registry = ModelRegistry()

    @patch("src.services.model_registry.mlflow.search_runs")
    def test_list_models(self, mock_search):
        # Mock dataframe return from search_runs
        mock_df = pd.DataFrame(
            {
                "run_id": ["run1", "run2"],
                "start_time": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
                "status": ["FINISHED", "FINISHED"],
                "status": ["FINISHED", "FINISHED"],
                "metrics.cv_mean_score": [0.95, 0.96],
                "tags.dataset_name": ["d1", "d2"],
            }
        )
        mock_search.return_value = mock_df

        models = self.registry.list_models()
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]["run_id"], "run1")
        self.assertEqual(models[0]["tags.dataset_name"], "d1")

    @patch("src.services.model_registry.mlflow.search_runs")
    def test_list_models_missing_col(self, mock_search):
        # Mock dataframe WITHOUT metric column
        mock_df = pd.DataFrame(
            {"run_id": ["run1"], "start_time": [datetime(2023, 1, 1)], "status": ["FINISHED"]}
        )
        mock_search.return_value = mock_df

        models = self.registry.list_models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["run_id"], "run1")
        # Ensure it didn't crash and metric key is absent
        self.assertNotIn("metrics.cv_mean_score", models[0])
        self.assertNotIn("tags.dataset_name", models[0])

    @patch("src.services.model_registry.mlflow.pyfunc.load_model")
    def test_load_model(self, mock_load):
        mock_model_wrapper = MagicMock()
        # First unwrap returns the SalaryForecasterWrapper instance
        # Second unwrap (called on Wrapper) returns the Inner Model

        # We simulate this chain
        mock_pyfunc_model = MagicMock()
        mock_wrapper = MagicMock()
        mock_wrapper.unwrap_python_model.return_value = "RealModel"

        mock_pyfunc_model.unwrap_python_model.return_value = mock_wrapper
        mock_load.return_value = mock_pyfunc_model

        model = self.registry.load_model("run123")
        self.assertEqual(model, "RealModel")
        mock_load.assert_called_with("runs:/run123/model")

    def test_save_model_deprecated(self):
        # Just ensure it doesn't crash on dummy call
        self.registry.save_model(None, "test")

    @patch("src.services.model_registry.mlflow.pyfunc.load_model")
    def test_load_model_error(self, mock_load):
        """Test load_model handles errors gracefully."""
        mock_load.side_effect = Exception("Model not found")

        with self.assertRaises(Exception) as cm:
            self.registry.load_model("invalid_run")

        self.assertIn("Model not found", str(cm.exception))

    @patch("src.services.model_registry.mlflow.search_runs")
    def test_list_models_empty(self, mock_search):
        """Test list_models with no runs."""
        mock_search.return_value = pd.DataFrame()

        models = self.registry.list_models()

        self.assertEqual(len(models), 0)
        self.assertIsInstance(models, list)

    @patch("src.services.model_registry.mlflow.search_runs")
    def test_list_models_search_error(self, mock_search):
        """Test list_models handles non-corruption search errors."""
        mock_search.side_effect = ValueError("MLflow connection error")

        models = self.registry.list_models()

        self.assertEqual(len(models), 0)
        self.assertIsInstance(models, list)

    @patch("src.services.model_registry.mlflow.search_runs")
    @patch.object(ModelRegistry, "_list_models_fallback")
    def test_list_models_corrupted_metadata_fallback_success(self, mock_fallback, mock_search):
        """Test list_models uses fallback when encountering corrupted metadata."""
        corrupted_error = AttributeError("'NoneType' object has no attribute 'copy'")
        mock_search.side_effect = corrupted_error

        mock_fallback_df = pd.DataFrame(
            {
                "run_id": ["run1"],
                "start_time": [datetime(2023, 1, 1)],
                "tags.dataset_name": ["d1"],
                "metrics.cv_mean_score": [0.95],
            }
        )
        mock_fallback.return_value = mock_fallback_df

        models = self.registry.list_models()

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["run_id"], "run1")
        mock_fallback.assert_called_once()

    @patch("src.services.model_registry.mlflow.search_runs")
    @patch.object(ModelRegistry, "_list_models_fallback")
    def test_list_models_corrupted_metadata_fallback_fails(self, mock_fallback, mock_search):
        """Test list_models returns empty list when fallback also fails."""
        corrupted_error = AttributeError("'NoneType' object has no attribute 'copy'")
        mock_search.side_effect = corrupted_error
        mock_fallback.side_effect = Exception("Fallback failed")

        models = self.registry.list_models()

        self.assertEqual(len(models), 0)
        self.assertIsInstance(models, list)
        mock_fallback.assert_called_once()

    @patch("src.services.model_registry.mlflow.search_runs")
    def test_list_models_corrupted_metadata_other_attribute_error(self, mock_search):
        """Test list_models handles other AttributeErrors differently."""
        other_error = AttributeError("'str' object has no attribute 'xyz'")
        mock_search.side_effect = other_error

        models = self.registry.list_models()

        self.assertEqual(len(models), 0)
        self.assertIsInstance(models, list)

    def test_list_models_fallback_success(self):
        """Test fallback method successfully lists runs."""
        mock_run1 = MagicMock()
        mock_run1.info.run_id = "run1"
        mock_run1.info.start_time = 1672531200000
        mock_run1.data.tags = {"dataset_name": "d1"}
        mock_run1.data.metrics = {"cv_mean_score": 0.95}

        mock_run2 = MagicMock()
        mock_run2.info.run_id = "run2"
        mock_run2.info.start_time = 1672617600000
        mock_run2.data.tags = {"dataset_name": "d2"}
        mock_run2.data.metrics = {"cv_mean_score": 0.96}

        self.registry.client.search_runs = MagicMock(return_value=[mock_run1, mock_run2])

        result = self.registry._list_models_fallback()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn("run_id", result.columns)
        self.assertIn("start_time", result.columns)
        self.assertIn("tags.dataset_name", result.columns)
        self.assertIn("metrics.cv_mean_score", result.columns)
        self.assertEqual(result.iloc[0]["run_id"], "run2")
        self.assertEqual(result.iloc[1]["run_id"], "run1")

    def test_list_models_fallback_skips_corrupted_runs(self):
        """Test fallback method skips individual corrupted runs."""
        from unittest.mock import PropertyMock

        mock_run1 = MagicMock()
        mock_run1.info.run_id = "run1"
        mock_run1.info.start_time = 1672531200000
        mock_run1.data.tags = {}
        mock_run1.data.metrics = {}

        mock_run_corrupted = MagicMock()
        type(mock_run_corrupted.info).run_id = PropertyMock(
            side_effect=Exception("Corrupted run metadata")
        )

        mock_run2 = MagicMock()
        mock_run2.info.run_id = "run2"
        mock_run2.info.start_time = 1672617600000
        mock_run2.data.tags = {}
        mock_run2.data.metrics = {}

        self.registry.client.search_runs = MagicMock(
            return_value=[mock_run1, mock_run_corrupted, mock_run2]
        )

        result = self.registry._list_models_fallback()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn("run1", result["run_id"].values)
        self.assertIn("run2", result["run_id"].values)

    def test_list_models_fallback_search_runs_fails(self):
        """Test fallback method handles search_runs failure."""
        corrupted_error = AttributeError("'NoneType' object has no attribute 'copy'")
        self.registry.client.search_runs = MagicMock(side_effect=corrupted_error)

        result = self.registry._list_models_fallback()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)

    def test_list_models_fallback_search_runs_other_error(self):
        """Test fallback method handles non-corruption errors from search_runs gracefully."""
        other_error = ValueError("Connection error")
        self.registry.client.search_runs = MagicMock(side_effect=other_error)

        result = self.registry._list_models_fallback()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)

    def test_list_models_fallback_empty_result(self):
        """Test fallback method returns empty DataFrame when no runs found."""
        self.registry.client.search_runs = MagicMock(return_value=[])

        result = self.registry._list_models_fallback()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)

    def test_list_models_fallback_complete_failure(self):
        """Test fallback method handles complete failure gracefully."""
        self.registry.client.search_runs = MagicMock(side_effect=Exception("Complete failure"))

        result = self.registry._list_models_fallback()

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)
