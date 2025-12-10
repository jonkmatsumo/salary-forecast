import unittest
from unittest.mock import MagicMock
import pandas as pd
from src.services.analytics_service import AnalyticsService

class TestAnalyticsService(unittest.TestCase):
    def setUp(self):
        self.service = AnalyticsService()

    def test_get_data_summary(self):
        df = pd.DataFrame({
            "Location": ["A", "A", "B"],
            "Level": ["L1", "L2", "L1"]
        })
        summary = self.service.get_data_summary(df)
        
        self.assertEqual(summary["total_samples"], 3)
        self.assertEqual(summary["unique_location"], 2) # Location is object
        self.assertEqual(summary["unique_level"], 2)    # Level is object
        self.assertEqual(summary["shape"], (3, 2))

    def test_get_available_targets(self):
        mock_model = MagicMock()
        mock_model.targets = ["TargetA"]
        
        targets = self.service.get_available_targets(mock_model)
        self.assertEqual(targets, ["TargetA"])
        
        # Fallback test
        del mock_model.targets
        mock_model.models = {"TargetB_p50": None}
        targets = self.service.get_available_targets(mock_model)
        self.assertEqual(targets, ["TargetB"])

    def test_get_feature_importance(self):
        mock_model = MagicMock()
        mock_booster = MagicMock()
        # Handle the check for get_booster which MagicMock passes by default
        mock_booster.get_booster.return_value = mock_booster
        mock_booster.get_score.return_value = {"Feature1": 10.0, "Feature2": 5.0}
        
        # Mock the xgb model inside
        mock_model.models = {"TargetA_p50": mock_booster}
        
        df_imp = self.service.get_feature_importance(mock_model, "TargetA", 0.5)
        
        self.assertIsNotNone(df_imp)
        self.assertEqual(len(df_imp), 2)
        self.assertEqual(df_imp.iloc[0]["Feature"], "Feature1") # Sorted by gain desc
