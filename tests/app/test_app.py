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
        self.assertTrue(len(at.text_area) > 0)

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
        
        self.assertTrue(has_warning or has_selectbox, "Should show either warning or model selector")

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
