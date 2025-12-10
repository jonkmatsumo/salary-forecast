"""Tests for JSON normalization utilities."""

import unittest
import json
from src.utils.json_utils import normalize_json_string, parse_df_json_safely


class TestNormalizeJsonString(unittest.TestCase):
    """Tests for normalize_json_string function."""
    
    def test_valid_json(self):
        """Test parsing valid JSON string."""
        json_str = '{"key": "value", "number": 42}'
        result = normalize_json_string(json_str)
        self.assertEqual(result["key"], "value")
        self.assertEqual(result["number"], 42)
    
    def test_escaped_quotes(self):
        """Test parsing JSON with escaped quotes."""
        json_str = '{\\"key\\": \\"value\\", \\"number\\": 42}'
        result = normalize_json_string(json_str)
        self.assertEqual(result["key"], "value")
        self.assertEqual(result["number"], 42)
    
    def test_quoted_json_string(self):
        """Test parsing JSON wrapped in outer quotes."""
        json_str = '"{\\"key\\": \\"value\\"}"'
        result = normalize_json_string(json_str)
        self.assertEqual(result["key"], "value")
    
    def test_double_escaped(self):
        """Test parsing double-escaped JSON."""
        original = '{"Level": {"0": "E6", "1": "E6"}}'
        escaped = json.dumps(original)
        result = normalize_json_string(escaped)
        self.assertEqual(result["Level"]["0"], "E6")
        self.assertEqual(result["Level"]["1"], "E6")
    
    def test_dataframe_like_json(self):
        """Test parsing DataFrame-like JSON structure."""
        json_str = '{"Level": {"0": "E6", "1": "E6", "2": "E3"}, "Salary": {"0": 100000, "1": 150000}}'
        result = normalize_json_string(json_str)
        self.assertEqual(result["Level"]["0"], "E6")
        self.assertEqual(result["Salary"]["0"], 100000)
    
    def test_escaped_dataframe_json(self):
        """Test parsing escaped DataFrame JSON (the actual issue)."""
        json_str = '{\\"Level\\": {\\"0\\": \\"E6\\", \\"1\\": \\"E6\\"}}'
        result = normalize_json_string(json_str)
        self.assertEqual(result["Level"]["0"], "E6")
        self.assertEqual(result["Level"]["1"], "E6")
    
    def test_nested_structures(self):
        """Test parsing nested JSON structures."""
        json_str = '{"outer": {"inner": {"value": 123}}}'
        result = normalize_json_string(json_str)
        self.assertEqual(result["outer"]["inner"]["value"], 123)
    
    def test_array_values(self):
        """Test parsing JSON with array values."""
        json_str = '{"col1": [1, 2, 3], "col2": ["a", "b"]}'
        result = normalize_json_string(json_str)
        self.assertEqual(result["col1"], [1, 2, 3])
        self.assertEqual(result["col2"], ["a", "b"])
    
    def test_empty_string(self):
        """Test handling of empty string."""
        with self.assertRaises(ValueError):
            normalize_json_string("")
    
    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        with self.assertRaises(ValueError):
            normalize_json_string("not json at all")
    
    def test_malformed_json(self):
        """Test handling of malformed JSON."""
        with self.assertRaises(ValueError):
            normalize_json_string('{"key": "value"')
    
    def test_none_input(self):
        """Test handling of None input."""
        with self.assertRaises(ValueError):
            normalize_json_string(None)
    
    def test_unicode_escape(self):
        """Test parsing JSON with unicode escape sequences."""
        json_str = '{"key": "value\\nwith\\tnewline"}'
        result = normalize_json_string(json_str)
        self.assertEqual(result["key"], "value\nwith\tnewline")


class TestParseDfJsonSafely(unittest.TestCase):
    """Tests for parse_df_json_safely function."""
    
    def test_valid_json(self):
        """Test parsing valid JSON."""
        json_str = '{"A": [1, 2], "B": [3, 4]}'
        result = parse_df_json_safely(json_str)
        self.assertEqual(result["A"], [1, 2])
        self.assertEqual(result["B"], [3, 4])
    
    def test_escaped_json(self):
        """Test parsing escaped JSON."""
        json_str = '{\\"A\\": [1, 2], \\"B\\": [3, 4]}'
        result = parse_df_json_safely(json_str)
        self.assertEqual(result["A"], [1, 2])
        self.assertEqual(result["B"], [3, 4])
    
    def test_error_formatting(self):
        """Test that errors are properly formatted."""
        with self.assertRaises(ValueError) as context:
            parse_df_json_safely("invalid json")
        
        error_str = str(context.exception)
        self.assertIn("Invalid JSON", error_str)
    
    def test_error_contains_preview(self):
        """Test that error messages contain JSON preview."""
        invalid_json = "{" * 100
        with self.assertRaises(ValueError) as context:
            parse_df_json_safely(invalid_json)
        
        error_str = str(context.exception)
        try:
            error_detail = json.loads(error_str)
            self.assertIn("df_json_preview", error_detail)
            self.assertIn("error_type", error_detail)
        except json.JSONDecodeError:
            self.assertIn("Invalid JSON", error_str)
    
    def test_real_world_scenario(self):
        """Test with real-world escaped JSON scenario from issue."""
        json_str = '{\\"Level\\": {\\"0\\": \\"E6\\", \\"1\\": \\"E6\\", \\"2\\": \\"E3\\", \\"3\\": \\"E3\\"}}'
        result = parse_df_json_safely(json_str)
        self.assertEqual(result["Level"]["0"], "E6")
        self.assertEqual(result["Level"]["1"], "E6")
        self.assertEqual(result["Level"]["2"], "E3")
    
    def test_pandas_to_json_output(self):
        """Test parsing output from pandas to_json()."""
        import pandas as pd
        df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        json_str = df.to_json(orient='columns')
        result = parse_df_json_safely(json_str)
        self.assertIn("A", result)
        self.assertIn("B", result)
        self.assertEqual(len(result["A"]), 3)


if __name__ == "__main__":
    unittest.main()

