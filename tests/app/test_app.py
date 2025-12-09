import os
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.app.app import render_model_information, render_inference_ui, main

class TestStreamlitApp(unittest.TestCase):
    def setUp(self):
        self.app_path = "src/app/app.py"
        
        if not os.path.exists(self.app_path):
            self.fail(f"App file not found at {self.app_path}")

    def test_app_smoke(self):
        """Verify the app launches without error."""
        at = AppTest.from_file(self.app_path)
        at.run()
        
        self.assertTrue(len(at.header) > 0)
        self.assertEqual(at.header[0].value, "Model Training")
        
        self.assertTrue(at.sidebar.title[0].value == "Navigation")

    def test_navigation_training(self):
        """Verify navigation to Training page."""
        at = AppTest.from_file(self.app_path)
        at.run()
        
        at.sidebar.radio[0].set_value("Training").run()
        
        self.assertEqual(at.header[0].value, "Model Training")
        
        if len(at.checkbox) > 0:
            self.assertTrue(len(at.checkbox) >= 2, "Should have at least 2 checkboxes (Outliers, Tune)")

    def test_navigation_inference(self):
        """Verify navigation to Inference page."""
        at = AppTest.from_file(self.app_path)
        at.run()
        
        at.sidebar.radio[0].set_value("Inference").run()
        
        self.assertEqual(at.header[0].value, "Salary Inference")
        
        has_warning = len(at.warning) > 0
        has_selectbox = len(at.selectbox) > 0
        
        self.assertTrue(has_warning or has_selectbox or len(at.header) > 0, 
                       "Should show header (and optionally warning or model selector)")
        
        if has_selectbox:
            subheaders = [sh.value for sh in at.subheader]
            self.assertTrue(len(subheaders) > 0, "Should have subheaders when model is loaded")

    def test_navigation_configuration_removed(self):
        """Verify Configuration tab has been removed."""
        at = AppTest.from_file(self.app_path)
        at.run()
        
        radio_options = at.sidebar.radio[0].options
        self.assertNotIn("Configuration", radio_options)
        self.assertNotIn("Data Analysis", radio_options)
        self.assertNotIn("Model Analysis", radio_options)
        self.assertIn("Training", radio_options)
        self.assertIn("Inference", radio_options)

    def test_inference_inputs(self):
        """Verify inference inputs exist when a model is selected."""
        at = AppTest.from_file(self.app_path)
        at.run()
        at.sidebar.radio[0].set_value("Inference").run()
        
        if at.selectbox:
            subheaders = [sh.value for sh in at.subheader]
            self.assertTrue(len(subheaders) > 0, "Should have subheaders when model is loaded")
            
            self.assertTrue(any("Model Information" in sh.value for sh in at.subheader), 
                          "Should show Model Information section")


class TestRenderModelInformation(unittest.TestCase):
    """Tests for render_model_information function."""
    
    @patch("src.app.app.st")
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
    
    @patch("src.app.app.st")
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
    
    @patch("src.app.app.st")
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
    
    @patch("src.app.app.st")
    @patch("src.app.app.ModelRegistry")
    def test_render_inference_ui_no_models(self, mock_registry_class, mock_st):
        """Verify user is informed when no models are available."""
        mock_registry = mock_registry_class.return_value
        mock_registry.list_models.return_value = []
        
        render_inference_ui()
        
        mock_st.warning.assert_called_with("No trained models found in MLflow. Please train a new model.")
        mock_st.selectbox.assert_not_called()
    
    @patch("src.app.app.st")
    @patch("src.app.app.ModelRegistry")
    @patch("src.app.app.render_model_information")
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
    
    @patch("src.app.app.st")
    @patch("src.app.app.ModelRegistry")
    @patch("src.app.app.render_model_information")
    @patch("src.app.app.AnalyticsService")
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


class TestMain(unittest.TestCase):
    """Tests for main() function."""
    
    @patch("src.app.app.st")
    @patch("src.app.app.get_config")
    @patch("src.app.app.render_training_ui")
    def test_main_defaults_to_training(self, mock_render_training, mock_get_config, mock_st):
        """Test that main() defaults to Training page."""
        mock_get_config.return_value = {}
        mock_st.session_state = {}
        mock_st.sidebar.radio.return_value = "Training"
        
        main()
        
        mock_st.set_page_config.assert_called_once()
        mock_st.sidebar.title.assert_called_with("Navigation")
        mock_render_training.assert_called_once()
    
    @patch("src.app.app.st")
    @patch("src.app.app.get_config")
    @patch("src.app.app.render_inference_ui")
    def test_main_navigation_to_inference(self, mock_render_inference, mock_get_config, mock_st):
        """Test that main() navigates to Inference page."""
        mock_get_config.return_value = {}
        mock_st.session_state = {"nav": "Inference"}
        mock_st.sidebar.radio.return_value = "Inference"
        
        main()
        
        mock_render_inference.assert_called_once()

