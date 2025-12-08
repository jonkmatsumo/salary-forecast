import unittest
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

    def test_infer_locations(self):
        data = pd.DataFrame({"Location": ["NY", "SF", "NY"]})
        locs = self.generator.infer_locations(data)
        self.assertEqual(locs, {"NY": 2, "SF": 2})

    def test_generate_config_template(self):
        data = pd.DataFrame({
            "Level": ["L3"],
            "Location": ["NY"],
            "Salary": [100]
        })
        config = self.generator.generate_config_template(data)
        self.assertIn("mappings", config)
        self.assertIn("model", config)
        self.assertEqual(config["mappings"]["levels"]["L3"], 0)
