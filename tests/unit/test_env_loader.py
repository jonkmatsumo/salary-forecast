import unittest
from unittest.mock import patch
from src.utils.env_loader import get_env_var

class TestEnvLoader(unittest.TestCase):
    @patch("src.utils.env_loader.os.getenv")
    def test_get_env_var_existing(self, mock_getenv):
        mock_getenv.return_value = "fake_val"
        val = get_env_var("MY_KEY")
        self.assertEqual(val, "fake_val")
        mock_getenv.assert_called_with("MY_KEY", None)

    @patch("src.utils.env_loader.os.getenv")
    def test_get_env_var_default(self, mock_getenv):
        # Simulator os.getenv behavior when key missing: it returns default
        def side_effect(key, default=None):
             if key == "MISSING": return default
             return "val"
             
        mock_getenv.side_effect = side_effect
        
        val = get_env_var("MISSING", default="default_val")
        self.assertEqual(val, "default_val")
