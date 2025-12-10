import os
import unittest
from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest
from src.app.app import main

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

