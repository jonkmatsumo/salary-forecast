import os
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.app.app import main


class TestStreamlitApp(unittest.TestCase):
    def setUp(self):
        self.app_path = "src/app/app.py"

        if not os.path.exists(self.app_path):
            self.fail(f"App file not found at {self.app_path}")

    def test_app_smoke(self):
        at = AppTest.from_file(self.app_path)
        at.run()

        self.assertTrue(len(at.header) > 0)
        self.assertEqual(at.header[0].value, "Model Training")

        self.assertTrue(at.sidebar.title[0].value == "Navigation")

    def test_navigation_training(self):
        at = AppTest.from_file(self.app_path)
        at.run()

        at.sidebar.radio[0].set_value("Training").run()

        self.assertEqual(at.header[0].value, "Model Training")

        if len(at.checkbox) > 0:
            self.assertTrue(
                len(at.checkbox) >= 2, "Should have at least 2 checkboxes (Outliers, Tune)"
            )

    def test_navigation_inference(self):
        at = AppTest.from_file(self.app_path)
        at.run()

        at.sidebar.radio[0].set_value("Inference").run()

        self.assertEqual(at.header[0].value, "Salary Inference")

        has_warning = len(at.warning) > 0
        has_selectbox = len(at.selectbox) > 0

        self.assertTrue(
            has_warning or has_selectbox or len(at.header) > 0,
            "Should show header (and optionally warning or model selector)",
        )

        if has_selectbox:
            subheaders = [sh.value for sh in at.subheader]
            self.assertTrue(len(subheaders) > 0, "Should have subheaders when model is loaded")

    def test_navigation_configuration_removed(self):
        at = AppTest.from_file(self.app_path)
        at.run()

        radio_options = at.sidebar.radio[0].options
        self.assertNotIn("Configuration", radio_options)
        self.assertNotIn("Data Analysis", radio_options)
        self.assertNotIn("Model Analysis", radio_options)
        self.assertIn("Training", radio_options)
        self.assertIn("Inference", radio_options)

    def test_inference_inputs(self):
        at = AppTest.from_file(self.app_path)
        at.run()
        at.sidebar.radio[0].set_value("Inference").run()

        if at.selectbox:
            subheaders = [sh.value for sh in at.subheader]
            self.assertTrue(len(subheaders) > 0, "Should have subheaders when model is loaded")

            self.assertTrue(
                any("Model Information" in sh.value for sh in at.subheader),
                "Should show Model Information section",
            )


class TestMain(unittest.TestCase):
    @patch("src.app.app.st")
    @patch("src.app.app.render_training_ui")
    def test_main_defaults_to_training(self, mock_render_training, mock_st):
        mock_st.session_state = {}
        mock_st.sidebar.radio.return_value = "Training"

        main()

        mock_st.set_page_config.assert_called_once()
        mock_st.sidebar.title.assert_called_with("Navigation")
        mock_render_training.assert_called_once()
        # Verify config_override is initialized to None
        self.assertIn("config_override", mock_st.session_state)
        self.assertIsNone(mock_st.session_state["config_override"])

    @patch("src.app.app.st")
    @patch("src.app.app.render_inference_ui")
    def test_main_navigation_to_inference(self, mock_render_inference, mock_st):
        mock_st.session_state = {"nav": "Inference"}
        mock_st.sidebar.radio.return_value = "Inference"

        main()

        mock_render_inference.assert_called_once()
        # Verify config_override is initialized to None
        self.assertIn("config_override", mock_st.session_state)
        self.assertIsNone(mock_st.session_state["config_override"])

    @patch("src.app.app.st")
    @patch("src.app.app.render_training_ui")
    def test_main_initializes_empty_config_state(self, mock_render_training, mock_st):
        """Test that app initializes with empty config state."""
        mock_st.session_state = {}
        mock_st.sidebar.radio.return_value = "Training"

        main()

        # Verify config_override is initialized
        self.assertIn("config_override", mock_st.session_state)
        self.assertIsNone(mock_st.session_state["config_override"])

    @patch("src.app.app.st")
    @patch("src.app.app.render_training_ui")
    def test_main_preserves_existing_config(self, mock_render_training, mock_st):
        """Test that app preserves existing config from workflow wizard."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from conftest import create_test_config

        existing_config = create_test_config()
        mock_st.session_state = {"config_override": existing_config}
        mock_st.sidebar.radio.return_value = "Training"

        main()

        # Verify existing config is preserved
        self.assertEqual(mock_st.session_state["config_override"], existing_config)
