import os
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd
from src.app.inference_ui import render_model_information, render_inference_ui

class TestRenderModelInformation(unittest.TestCase):
    """Tests for render_model_information function."""
    
    @patch("src.app.inference_ui.st")
    def test_render_model_information_with_run(self, mock_st):
        """Verify model information displays correctly when run data is available."""
        mock_forecaster = MagicMock()
        mock_forecaster.ranked_encoders = {
            "Level": MagicMock(mapping={"E3": 0, "E4": 1, "E5": 2})
        }
        mock_forecaster.proximity_encoders = {
            "Location": MagicMock()
        }
        mock_forecaster.feature_names = ["Level_Enc", "Location_Enc", "YearsOfExperience"]
        
        run_id = "test_run_12345"
        runs = [{
            "run_id": run_id,
            "start_time": datetime(2023, 1, 1, 12, 0),
            "tags.model_type": "XGBoost",
            "metrics.cv_mean_score": 0.95,
            "tags.dataset_name": "Test Dataset",
            "tags.additional_tag": "test_tag"
        }]
        
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2]
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=None)
        
        render_model_information(mock_forecaster, run_id, runs)
        
        mock_st.subheader.assert_called_with("Model Information")
        
        markdown_calls = [call[0][0] for call in mock_st.markdown.call_args_list]
        self.assertTrue(any("Run ID" in call for call in markdown_calls))
    
    @patch("src.app.inference_ui.st")
    def test_render_model_information_no_run(self, mock_st):
        """Verify graceful handling when run metadata is missing."""
        mock_forecaster = MagicMock()
        mock_forecaster.ranked_encoders = {}
        mock_forecaster.proximity_encoders = {}
        mock_forecaster.feature_names = []
        
        run_id = "missing_run"
        runs = [{"run_id": "other_run", "start_time": datetime(2023, 1, 1)}]
        
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=None)
        
        render_model_information(mock_forecaster, run_id, runs)
        
        mock_st.info.assert_called_with("Metadata not available")
    
    @patch("src.app.inference_ui.st")
    def test_render_model_information_feature_info(self, mock_st):
        """Verify feature information is displayed for model transparency."""
        mock_forecaster = MagicMock()
        mock_forecaster.ranked_encoders = {
            "Level": MagicMock(mapping={"E3": 0, "E4": 1, "E5": 2, "E6": 3, "E7": 4, "E8": 5})
        }
        mock_forecaster.proximity_encoders = {"Location": MagicMock()}
        mock_forecaster.feature_names = ["Level_Enc", "Location_Enc", "YearsOfExperience"]
        
        run_id = "test_run"
        runs = [{"run_id": run_id, "start_time": datetime(2023, 1, 1)}]
        
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2]
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=None)
        
        render_model_information(mock_forecaster, run_id, runs)
        
        markdown_calls = [call[0][0] for call in mock_st.markdown.call_args_list]
        self.assertTrue(any("Ranked Features" in call for call in markdown_calls))
        self.assertTrue(any("Proximity Features" in call for call in markdown_calls))
        self.assertTrue(any("Total Features" in call for call in markdown_calls))


class TestRenderInferenceUI(unittest.TestCase):
    """Tests for render_inference_ui function."""
    
    @patch("src.app.inference_ui.st")
    @patch("src.app.inference_ui.ModelRegistry")
    def test_render_inference_ui_no_models(self, mock_registry_class, mock_st):
        """Verify user is informed when no models are available."""
        mock_registry = mock_registry_class.return_value
        mock_registry.list_models.return_value = []
        
        render_inference_ui()
        
        mock_st.warning.assert_called_with("No trained models found in MLflow. Please train a new model.")
        mock_st.selectbox.assert_not_called()
    
    @patch("src.app.inference_ui.st")
    @patch("src.app.inference_ui.ModelRegistry")
    @patch("src.app.inference_ui.render_model_information")
    def test_render_inference_ui_model_loading_error(self, mock_render_info, mock_registry_class, mock_st):
        """Verify graceful error handling when model loading fails."""
        mock_registry = mock_registry_class.return_value
        mock_registry.list_models.return_value = [{
            "run_id": "test_run",
            "start_time": datetime(2023, 1, 1, 12, 0),
            "tags.model_type": "XGBoost",
            "metrics.cv_mean_score": 0.95,
            "tags.dataset_name": "Test Dataset"
        }]
        
        mock_st.selectbox.return_value = "2023-01-01 12:00 | XGBoost | Test Dataset | CV:0.9500 | ID:test_run"
        
        mock_st.session_state = {}
        
        mock_registry.load_model.side_effect = Exception("Model loading failed")
        
        render_inference_ui()
        
        # Should show error message
        mock_st.error.assert_called()
        error_call = mock_st.error.call_args[0][0]
        self.assertIn("Failed to load model", error_call)
    
    @patch("src.app.inference_ui.st")
    @patch("src.app.inference_ui.ModelRegistry")
    @patch("src.app.inference_ui.render_model_information")
    @patch("src.app.inference_ui.AnalyticsService")
    def test_render_inference_ui_success(self, mock_analytics_class, mock_render_info, mock_registry_class, mock_st):
        """Test render_inference_ui with successful model loading."""
        mock_registry = mock_registry_class.return_value
        mock_registry.list_models.return_value = [{
            "run_id": "test_run",
            "start_time": datetime(2023, 1, 1, 12, 0),
            "tags.model_type": "XGBoost",
            "metrics.cv_mean_score": 0.95,
            "tags.dataset_name": "Test Dataset"
        }]
        
        # Mock forecaster
        mock_forecaster = MagicMock()
        mock_forecaster.ranked_encoders = {"Level": MagicMock(mapping={"E3": 0, "E4": 1})}
        mock_forecaster.proximity_encoders = {}
        mock_forecaster.feature_names = ["Level_Enc", "YearsOfExperience"]
        
        mock_registry.load_model.return_value = mock_forecaster
        
        # Mock selectbox
        mock_st.selectbox.return_value = "2023-01-01 12:00 | XGBoost | Test Dataset | CV:0.9500 | ID:test_run"
        
        # Mock session state
        mock_st.session_state = {}
        
        # Mock form and columns
        mock_form = MagicMock()
        mock_st.form.return_value.__enter__ = MagicMock(return_value=mock_form)
        mock_st.form.return_value.__exit__ = MagicMock(return_value=None)
        # Mock form_submit_button to return False (form not submitted)
        # Note: form_submit_button is called on st, not on the form context
        mock_st.form_submit_button.return_value = False
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2]
        
        # Mock analytics service
        mock_analytics = mock_analytics_class.return_value
        mock_analytics.get_available_targets.return_value = ["BaseSalary"]
        mock_analytics.get_available_quantiles.return_value = [0.5]
        mock_analytics.get_feature_importance.return_value = pd.DataFrame()
        
        # Mock expander for Model Analysis
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=MagicMock())
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander
        
        render_inference_ui()
        
        # Verify model information was rendered
        mock_render_info.assert_called_once()
        
        # Verify header was set
        mock_st.header.assert_called_with("Salary Inference")

