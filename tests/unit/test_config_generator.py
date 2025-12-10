"""Tests for ConfigGenerator service."""

import unittest
import warnings
import pandas as pd
from src.services.config_generator import ConfigGenerator


class TestConfigGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = ConfigGenerator()
        
    def test_infer_levels_standard(self):
        data = pd.DataFrame({"Level": ["L4", "L3", "L5"]})
        levels = self.generator.infer_levels(data)
        
        # Expected: L3=0, L4=1, L5=2
        sorted_keys = sorted(levels, key=levels.get)
        self.assertEqual(sorted_keys, ["L3", "L4", "L5"])
        
    def test_infer_levels_mixed(self):
        data = pd.DataFrame({"Level": ["Senior", "Junior", "Staff"]})
        levels = self.generator.infer_levels(data)
        
        # Should be alphabetical since no numbers
        sorted_keys = sorted(levels, key=levels.get)
        self.assertEqual(sorted_keys, ["Junior", "Senior", "Staff"])
        
    def test_infer_levels_complex(self):
        data = pd.DataFrame({"Level": ["IC3", "IC4", "Manager M1", "IC5"]})
        levels = self.generator.infer_levels(data)
        
        # M1 (1) < IC3 (3) < IC4 (4) < IC5 (5) based on integer extraction
        sorted_keys = sorted(levels, key=levels.get)
        self.assertEqual(sorted_keys, ["Manager M1", "IC3", "IC4", "IC5"])

    def test_infer_levels_missing_column(self):
        data = pd.DataFrame({"Other": ["A", "B"]})
        levels = self.generator.infer_levels(data)
        self.assertEqual(levels, {})

    def test_infer_locations(self):
        data = pd.DataFrame({"Location": ["NY", "SF", "NY"]})
        locs = self.generator.infer_locations(data)
        self.assertEqual(locs, {"NY": 2, "SF": 2})

    def test_infer_locations_missing_column(self):
        data = pd.DataFrame({"Other": ["A", "B"]})
        locs = self.generator.infer_locations(data)
        self.assertEqual(locs, {})

    def test_infer_targets(self):
        data = pd.DataFrame({
            "Salary": [100000, 150000],
            "TotalComp": [120000, 180000],
            "Name": ["Alice", "Bob"],
            "Level": ["L3", "L4"]
        })
        targets = self.generator.infer_targets(data)
        
        self.assertIn("Salary", targets)
        self.assertIn("TotalComp", targets)
        self.assertNotIn("Name", targets)
        self.assertNotIn("Level", targets)

    def test_infer_features(self):
        data = pd.DataFrame({
            "Salary": [100000],
            "YearsOfExperience": [5],
            "Level": ["L3"],
            "EmployeeID": [12345]
        })
        
        features = self.generator.infer_features(data, exclude_cols=["Salary"])
        feature_names = [f["name"] for f in features]
        
        # YearsOfExperience and Level should be features
        self.assertIn("YearsOfExperience", feature_names)
        self.assertIn("Level", feature_names)
        # Salary excluded, EmployeeID should be filtered as ID column
        self.assertNotIn("Salary", feature_names)
        self.assertNotIn("EmployeeID", feature_names)
        
        # Check constraints
        yoe_feature = next(f for f in features if f["name"] == "YearsOfExperience")
        self.assertEqual(yoe_feature["monotone_constraint"], 1)  # Experience = positive

    def test_generate_config_template(self):
        data = pd.DataFrame({
            "Level": ["L3"],
            "Location": ["NY"],
            "Salary": [100000]
        })
        config = self.generator.generate_config_template(data)
        
        self.assertIn("mappings", config)
        self.assertIn("model", config)
        self.assertIn("feature_engineering", config)
        self.assertEqual(config["mappings"]["levels"]["L3"], 0)
        self.assertIn("Salary", config["model"]["targets"])

    def test_generate_config_heuristic_explicit(self):
        data = pd.DataFrame({
            "Level": ["L3"], 
            "Location": ["NY"],
            "BaseSalary": [100000]
        })
        
        config = self.generator.generate_config(data, use_llm=False)
        
        # Verify heuristic structure
        self.assertEqual(config["mappings"]["levels"]["L3"], 0)
        self.assertIn("BaseSalary", config["model"]["targets"])

    def test_generate_config_use_llm_warning(self):
        """Test that use_llm=True now warns and falls back to heuristics."""
        data = pd.DataFrame({
            "Level": ["L3"], 
            "Location": ["NY"],
            "Salary": [100000]
        })
        
        # Should warn that use_llm is deprecated
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = self.generator.generate_config(data, use_llm=True)
            
            # Check we got the config (heuristic fallback)
            self.assertIn("mappings", config)
            self.assertIn("model", config)

    def test_config_template_structure(self):
        """Test that CONFIG_TEMPLATE has all required fields."""
        template = ConfigGenerator.CONFIG_TEMPLATE
        
        self.assertIn("mappings", template)
        self.assertIn("model", template)
        self.assertIn("location_settings", template)
        
        model = template["model"]
        self.assertIn("targets", model)
        self.assertIn("features", model)
        self.assertIn("quantiles", model)
        self.assertIn("hyperparameters", model)


if __name__ == "__main__":
    unittest.main()
