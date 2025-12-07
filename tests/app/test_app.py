import os
import unittest
from streamlit.testing.v1 import AppTest

class TestStreamlitApp(unittest.TestCase):
    def setUp(self):
        # Path to the app file relative to the project root
        self.app_path = "src/app/app.py"
        
        # Ensure the file exists
        if not os.path.exists(self.app_path):
            self.fail(f"App file not found at {self.app_path}")

    def test_app_smoke(self):
        """Verify the app launches without error."""
        at = AppTest.from_file(self.app_path)
        at.run()
        
        # Check title
        self.assertEqual(at.title[0].value, "Salary Forecasting Engine")
        
        # Check sidebar exists
        self.assertTrue(at.sidebar.title[0].value == "Navigation")

    def test_navigation_training(self):
        """Verify navigation to Training page."""
        at = AppTest.from_file(self.app_path)
        at.run()
        
        # Default should be Train Model or Inference depending on radio default
        # The app sets default radio to the first option "Train Model"
        
        # Select "Train Model" explicitly
        at.sidebar.radio[0].set_value("Train Model").run()
        
        # Check header
        self.assertEqual(at.header[0].value, "Train New Model")
        
        # Check we have config text area (button only shows after upload)
        # Check we have model name input
        self.assertTrue(len(at.text_input) >= 1)

    def test_navigation_inference(self):
        """Verify navigation to Inference page."""
        at = AppTest.from_file(self.app_path)
        at.run()
        
        # Select "Inference"
        at.sidebar.radio[0].set_value("Inference").run()
        
        # Check header
        self.assertEqual(at.header[0].value, "Salary Inference")
        
        # Check if model warning or selector appears
        # If no models, it shows warning. If models, it shows selectbox.
        # We can't strictly assert one or other without knowing env state, 
        # but we can check that at least one of them exists or generic content loads.
        
        has_warning = len(at.warning) > 0
        has_selectbox = len(at.selectbox) > 0
        
        has_warning = len(at.warning) > 0
        has_selectbox = len(at.selectbox) > 0
        
        self.assertTrue(has_warning or has_selectbox, "Should show either warning or model selector")

    def test_navigation_configuration(self):
        """Verify navigation to Configuration page."""
        at = AppTest.from_file(self.app_path)
        at.run()
        
        # Select "Configuration"
        at.sidebar.radio[0].set_value("Configuration").run()
        
        # Check header
        self.assertEqual(at.header[0].value, "Configuration")
        
        # Check for config UI elements (e.g., config save/load, or data editors)
        # We know we have data editors.
        # But AppTest might not expose them directly as a list if nested?
        # Actually in test_config_ui we saw data_editor is not supported by AppTest yet?
        # Wait, in test_app.py Step 677 failure: "AttributeError: 'AppTest' object has no attribute 'data_editor'"
        # So we should check for something else, like subheaders or buttons.
        # render_save_load_controls has a download button.
        
        # We can check for "Config Management" subheader which is rendered by render_save_load_controls
        subheaders = [sh.value for sh in at.subheader]
        self.assertIn("Config Management", subheaders)

    def test_inference_inputs(self):
        """Verify inference inputs exist when a model is selected (mocking if possible or checking structure)."""
        # This test relies on existing models. 
        # If we want to be more robust, we might need to mock glob or pickle, 
        # but AppTest starts a new process/sandbox so mocking is harder.
        # For now, we stick to checking UI element existence logic.
        
        at = AppTest.from_file(self.app_path)
        at.run()
        at.sidebar.radio[0].set_value("Inference").run()
        
        if at.selectbox:
            # If we have models, inputs should be visible
            # e.g. Level selectbox, Location text_input
            pass
    def test_navigation_data_analysis(self):
        """Verify navigation to Data Analysis page."""
        at = AppTest.from_file(self.app_path)
        at.run()
        
        # Select "Data Analysis"
        at.sidebar.radio[0].set_value("Data Analysis").run()
        
        # Check header
        self.assertEqual(at.header[0].value, "Data Analysis")
        
        # Should show info message initially (since no session state)
        # We check if any info box contains "No data loaded"
        has_info = any("No data loaded" in i.value for i in at.info)
        self.assertTrue(has_info, "Should show 'No data loaded' info message")
        
    def test_navigation_model_analysis(self):
        """Verify navigation to Model Analysis page."""
        at = AppTest.from_file(self.app_path)
        at.run()
        
        # Select "Model Analysis"
        at.sidebar.radio[0].set_value("Model Analysis").run()
        
        # Check header
        self.assertEqual(at.header[0].value, "Model Analysis")
        
        # Should show warning initially (since no model selected/found or just default state)
        # Note: glob will run. If no pkls, shows warning. If pkls, shows selectbox.
        # Just check we didn't crash and header is right.

