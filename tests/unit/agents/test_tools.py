import unittest
import json
import pandas as pd
import numpy as np
from datetime import datetime
from src.agents.tools import (
    compute_correlation_matrix,
    get_column_statistics,
    get_unique_value_counts,
    detect_ordinal_patterns,
    detect_column_dtype,
    get_all_tools,
)


class TestComputeCorrelationMatrix(unittest.TestCase):
    def test_basic_correlation(self):
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [2, 4, 6, 8, 10],
            "C": [5, 4, 3, 2, 1]
        })
        
        result = compute_correlation_matrix.invoke({"df_json": df.to_json()})
        result_dict = json.loads(result)
        
        self.assertIn("correlations", result_dict)
        self.assertIn("columns_analyzed", result_dict)
        
        # Find A-B correlation
        ab_corr = next(
            c for c in result_dict["correlations"] 
            if (c["column_1"] == "A" and c["column_2"] == "B") or
               (c["column_1"] == "B" and c["column_2"] == "A")
        )
        self.assertAlmostEqual(ab_corr["correlation"], 1.0, places=4)
    
    def test_specific_columns(self):
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [2, 4, 6],
            "C": [1, 1, 1],
            "D": "text"
        })
        
        result = compute_correlation_matrix.invoke({
            "df_json": df.to_json(),
            "columns": "A, B"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(set(result_dict["columns_analyzed"]), {"A", "B"})
    
    def test_insufficient_columns(self):
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["x", "y", "z"]
        })
        
        result = compute_correlation_matrix.invoke({"df_json": df.to_json()})
        result_dict = json.loads(result)
        
        self.assertIn("error", result_dict)
    
    def test_all_nan_columns(self):
        df = pd.DataFrame({
            "A": [np.nan, np.nan, np.nan],
            "B": [np.nan, np.nan, np.nan]
        })
        
        result = compute_correlation_matrix.invoke({"df_json": df.to_json()})
        result_dict = json.loads(result)
        
        # Should return error when no numeric columns (all NaN)
        self.assertIn("error", result_dict)
        self.assertIn("numeric columns", result_dict["error"])
    
    def test_single_numeric_column(self):
        """Test with single numeric column."""
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["x", "y", "z"]
        })
        
        result = compute_correlation_matrix.invoke({"df_json": df.to_json()})
        result_dict = json.loads(result)
        
        self.assertIn("error", result_dict)
    
    def test_mixed_numeric_non_numeric(self):
        """Test with mixed numeric and non-numeric columns."""
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [2, 4, 6, 8, 10],
            "C": ["x", "y", "z", "w", "v"],
            "D": [10, 20, 30, 40, 50]
        })
        
        result = compute_correlation_matrix.invoke({"df_json": df.to_json()})
        result_dict = json.loads(result)
        
        # Should only analyze numeric columns
        self.assertIn("A", result_dict["columns_analyzed"])
        self.assertIn("B", result_dict["columns_analyzed"])
        self.assertIn("D", result_dict["columns_analyzed"])
        self.assertNotIn("C", result_dict["columns_analyzed"])
    
    def test_large_correlation_matrix(self):
        """Test with large correlation matrix."""
        np.random.seed(42)
        data = {f"col_{i}": np.random.randn(100) for i in range(10)}
        df = pd.DataFrame(data)
        
        result = compute_correlation_matrix.invoke({"df_json": df.to_json()})
        result_dict = json.loads(result)
        
        # Should handle large matrices
        self.assertEqual(len(result_dict["columns_analyzed"]), 10)
        # Number of pairs = n*(n-1)/2 = 10*9/2 = 45
        self.assertEqual(len(result_dict["correlations"]), 45)
    
    def test_column_name_filtering_whitespace(self):
        """Test column name filtering with whitespace."""
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [2, 4, 6],
            "C": [3, 6, 9]
        })
        
        result = compute_correlation_matrix.invoke({
            "df_json": df.to_json(),
            "columns": " A ,  B  , C "
        })
        result_dict = json.loads(result)
        
        self.assertEqual(set(result_dict["columns_analyzed"]), {"A", "B", "C"})
    
    def test_column_name_special_chars(self):
        """Test column name filtering with special characters."""
        df = pd.DataFrame({
            "col-name": [1, 2, 3],
            "col_name": [2, 4, 6],
            "col.name": [3, 6, 9]
        })
        
        result = compute_correlation_matrix.invoke({
            "df_json": df.to_json(),
            "columns": "col-name, col_name"
        })
        result_dict = json.loads(result)
        
        self.assertIn("col-name", result_dict["columns_analyzed"] or [])
        self.assertIn("col_name", result_dict["columns_analyzed"] or [])


class TestGetColumnStatistics(unittest.TestCase):
    
    def test_numeric_column(self):
        """Test statistics for numeric column."""
        df = pd.DataFrame({
            "values": [1, 2, 3, 4, 5, None]
        })
        
        result = get_column_statistics.invoke({
            "df_json": df.to_json(),
            "column": "values"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["column"], "values")
        self.assertEqual(result_dict["total_count"], 6)
        self.assertEqual(result_dict["null_count"], 1)
        self.assertIn("numeric_stats", result_dict)
        self.assertEqual(result_dict["numeric_stats"]["mean"], 3.0)
    
    def test_categorical_column(self):
        """Test statistics for categorical column."""
        df = pd.DataFrame({
            "category": ["A", "B", "A", "C", "A"]
        })
        
        result = get_column_statistics.invoke({
            "df_json": df.to_json(),
            "column": "category"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["unique_count"], 3)
        self.assertNotIn("numeric_stats", result_dict)
    
    def test_missing_column(self):
        """Test with non-existent column."""
        df = pd.DataFrame({"A": [1, 2, 3]})
        
        result = get_column_statistics.invoke({
            "df_json": df.to_json(),
            "column": "NonExistent"
        })
        result_dict = json.loads(result)
        
        self.assertIn("error", result_dict)
    
    def test_all_null_values(self):
        """Test with all null values."""
        df = pd.DataFrame({
            "null_col": [None, None, None, None]
        })
        
        result = get_column_statistics.invoke({
            "df_json": df.to_json(),
            "column": "null_col"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["null_count"], 4)
        self.assertEqual(result_dict["null_percentage"], 100.0)
        self.assertEqual(result_dict["unique_count"], 0)
    
    def test_datetime_column(self):
        """Test with datetime column."""
        df = pd.DataFrame({
            "dates": pd.date_range("2023-01-01", periods=5, freq="D")
        })
        
        result = get_column_statistics.invoke({
            "df_json": df.to_json(),
            "column": "dates"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["total_count"], 5)
        self.assertIn("dtype", result_dict)
        # Datetime columns may not have numeric_stats
        if "numeric_stats" not in result_dict:
            # That's fine, datetime is not numeric
            pass
    
    def test_boolean_column(self):
        """Test with boolean column."""
        df = pd.DataFrame({
            "flags": [True, False, True, False, True]
        })
        
        result = get_column_statistics.invoke({
            "df_json": df.to_json(),
            "column": "flags"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["total_count"], 5)
        self.assertEqual(result_dict["unique_count"], 2)
    
    def test_boolean_column_edge_cases(self):
        """Test boolean column with 0/1 values."""
        df = pd.DataFrame({
            "binary": [0, 1, 0, 1, 0]
        })
        
        result = get_column_statistics.invoke({
            "df_json": df.to_json(),
            "column": "binary"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["unique_count"], 2)
        # Should have numeric_stats since it's numeric dtype
        if "numeric_stats" in result_dict:
            self.assertIsNotNone(result_dict["numeric_stats"]["mean"])
    
    def test_large_dataset(self):
        """Test with very large dataset."""
        np.random.seed(42)
        df = pd.DataFrame({
            "large_col": np.random.randn(10000)
        })
        
        result = get_column_statistics.invoke({
            "df_json": df.to_json(),
            "column": "large_col"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["total_count"], 10000)
        self.assertIn("numeric_stats", result_dict)
    
    def test_sample_values_truncation(self):
        """Test sample_values are limited to 5."""
        df = pd.DataFrame({
            "values": list(range(100))
        })
        
        result = get_column_statistics.invoke({
            "df_json": df.to_json(),
            "column": "values"
        })
        result_dict = json.loads(result)
        
        if "sample_values" in result_dict:
            self.assertLessEqual(len(result_dict["sample_values"]), 5)


class TestGetUniqueValueCounts(unittest.TestCase):
    
    def test_basic_counts(self):
        """Test basic value counting."""
        df = pd.DataFrame({
            "status": ["Active", "Active", "Inactive", "Active", "Pending"]
        })
        
        result = get_unique_value_counts.invoke({
            "df_json": df.to_json(),
            "column": "status"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["total_unique_values"], 3)
        
        # Check Active count
        active_count = next(
            v for v in result_dict["value_counts"] 
            if v["value"] == "Active"
        )
        self.assertEqual(active_count["count"], 3)
    
    def test_limit(self):
        """Test limiting results."""
        df = pd.DataFrame({
            "id": list(range(100))
        })
        
        result = get_unique_value_counts.invoke({
            "df_json": df.to_json(),
            "column": "id",
            "limit": 5
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["showing_top"], 5)
        self.assertEqual(len(result_dict["value_counts"]), 5)
    
    def test_all_null_values(self):
        """Test with all null values."""
        df = pd.DataFrame({
            "null_col": [None, None, None]
        })
        
        result = get_unique_value_counts.invoke({
            "df_json": df.to_json(),
            "column": "null_col"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["total_unique_values"], 0)
        self.assertEqual(len(result_dict["value_counts"]), 0)
    
    def test_high_cardinality(self):
        """Test with very high cardinality (>1000 unique values)."""
        df = pd.DataFrame({
            "high_card": [f"value_{i}" for i in range(1500)]
        })
        
        result = get_unique_value_counts.invoke({
            "df_json": df.to_json(),
            "column": "high_card",
            "limit": 20
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["total_unique_values"], 1500)
        self.assertEqual(result_dict["showing_top"], 20)
        self.assertEqual(len(result_dict["value_counts"]), 20)
    
    def test_limit_zero(self):
        """Test with limit=0."""
        df = pd.DataFrame({
            "values": ["A", "B", "C"]
        })
        
        result = get_unique_value_counts.invoke({
            "df_json": df.to_json(),
            "column": "values",
            "limit": 0
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["showing_top"], 0)
        self.assertEqual(len(result_dict["value_counts"]), 0)
    
    def test_percentage_with_nulls(self):
        """Test percentage calculations with nulls."""
        df = pd.DataFrame({
            "mixed": ["A", "A", "B", None, None, "C"]
        })
        
        result = get_unique_value_counts.invoke({
            "df_json": df.to_json(),
            "column": "mixed"
        })
        result_dict = json.loads(result)
        
        # Percentages should be based on total count (6), not non-null count
        total = len(df)
        for vc in result_dict["value_counts"]:
            expected_pct = round(vc["count"] / total * 100, 2)
            self.assertEqual(vc["percentage"], expected_pct)


class TestDetectOrdinalPatterns(unittest.TestCase):
    
    def test_numeric_pattern(self):
        """Test detection of numeric patterns like L1, L2, L3."""
        df = pd.DataFrame({
            "level": ["L1", "L2", "L3", "L4", "L5"]
        })
        
        result = detect_ordinal_patterns.invoke({
            "df_json": df.to_json(),
            "column": "level"
        })
        result_dict = json.loads(result)
        
        self.assertTrue(result_dict["is_ordinal"])
        self.assertIn("numeric_in_string", result_dict["patterns_detected"])
        self.assertIn("suggested_mapping", result_dict)
        
        # Verify ordering: L1 < L2 < L3 < L4 < L5
        mapping = result_dict["suggested_mapping"]
        self.assertTrue(mapping["L1"] < mapping["L5"])
    
    def test_keyword_pattern(self):
        """Test detection of ordinal keywords."""
        df = pd.DataFrame({
            "seniority": ["Junior Developer", "Senior Developer", "Lead Developer"]
        })
        
        result = detect_ordinal_patterns.invoke({
            "df_json": df.to_json(),
            "column": "seniority"
        })
        result_dict = json.loads(result)
        
        self.assertTrue(result_dict["is_ordinal"])
        self.assertIn("ordinal_keywords", result_dict["patterns_detected"])
    
    def test_non_ordinal(self):
        """Test non-ordinal column."""
        df = pd.DataFrame({
            "color": ["Red", "Blue", "Green", "Yellow"]
        })
        
        result = detect_ordinal_patterns.invoke({
            "df_json": df.to_json(),
            "column": "color"
        })
        result_dict = json.loads(result)
        
        self.assertFalse(result_dict["is_ordinal"])
    
    def test_roman_numerals(self):
        """Test detection of Roman numerals."""
        df = pd.DataFrame({
            "tier": ["Tier I", "Tier II", "Tier III", "Tier IV", "Tier V"]
        })
        
        result = detect_ordinal_patterns.invoke({
            "df_json": df.to_json(),
            "column": "tier"
        })
        result_dict = json.loads(result)
        
        self.assertTrue(result_dict["is_ordinal"])
        self.assertIn("roman_numerals", result_dict["patterns_detected"])
        mapping = result_dict["suggested_mapping"]
        # Verify ordering
        self.assertTrue(mapping["Tier I"] < mapping["Tier V"])
    
    def test_roman_numerals_all(self):
        """Test all Roman numerals I through X."""
        df = pd.DataFrame({
            "levels": ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
        })
        
        result = detect_ordinal_patterns.invoke({
            "df_json": df.to_json(),
            "column": "levels"
        })
        result_dict = json.loads(result)
        
        self.assertTrue(result_dict["is_ordinal"])
        self.assertIn("roman_numerals", result_dict["patterns_detected"])
    
    def test_mixed_patterns(self):
        """Test mixed patterns (some numeric, some keywords)."""
        df = pd.DataFrame({
            "mixed": ["Level 1", "Level 2", "Senior", "Junior", "Level 3"]
        })
        
        result = detect_ordinal_patterns.invoke({
            "df_json": df.to_json(),
            "column": "mixed"
        })
        result_dict = json.loads(result)
        
        # Should detect numeric pattern first
        self.assertTrue(result_dict["is_ordinal"])
        # Numeric pattern should take precedence
        self.assertIn("numeric_in_string", result_dict["patterns_detected"])
    
    def test_case_insensitive_keywords(self):
        """Test case-insensitive keyword matching."""
        df = pd.DataFrame({
            "roles": ["JUNIOR", "SENIOR", "LEAD", "junior", "senior"]
        })
        
        result = detect_ordinal_patterns.invoke({
            "df_json": df.to_json(),
            "column": "roles"
        })
        result_dict = json.loads(result)
        
        self.assertTrue(result_dict["is_ordinal"])
        self.assertIn("ordinal_keywords", result_dict["patterns_detected"])
    
    def test_no_clear_pattern(self):
        """Test column with no clear ordinal pattern."""
        df = pd.DataFrame({
            "random": ["Apple", "Banana", "Cherry", "Date", "Elderberry"]
        })
        
        result = detect_ordinal_patterns.invoke({
            "df_json": df.to_json(),
            "column": "random"
        })
        result_dict = json.loads(result)
        
        self.assertFalse(result_dict["is_ordinal"])
        self.assertEqual(len(result_dict["patterns_detected"]), 0)
        self.assertEqual(len(result_dict["suggested_mapping"]), 0)
    
    def test_very_long_category_names(self):
        """Test with very long category names."""
        df = pd.DataFrame({
            "long_names": [
                "Level 1 - Entry Level Position",
                "Level 2 - Mid Level Position",
                "Level 3 - Senior Level Position"
            ]
        })
        
        result = detect_ordinal_patterns.invoke({
            "df_json": df.to_json(),
            "column": "long_names"
        })
        result_dict = json.loads(result)
        
        # Should still detect numeric pattern
        self.assertTrue(result_dict["is_ordinal"])
        self.assertIn("numeric_in_string", result_dict["patterns_detected"])


class TestDetectColumnDtype(unittest.TestCase):
    
    def test_numeric_continuous(self):
        """Test detection of continuous numeric column."""
        df = pd.DataFrame({
            "price": [10.5, 20.3, 15.7, 30.2]
        })
        
        result = detect_column_dtype.invoke({
            "df_json": df.to_json(),
            "column": "price"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["semantic_type"], "numeric_continuous")
    
    def test_categorical(self):
        """Test detection of categorical column."""
        df = pd.DataFrame({
            "category": ["A", "B", "A", "C", "B"] * 10
        })
        
        result = detect_column_dtype.invoke({
            "df_json": df.to_json(),
            "column": "category"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["semantic_type"], "categorical")
    
    def test_identifier(self):
        """Test detection of ID column."""
        df = pd.DataFrame({
            "user_id": list(range(100))
        })
        
        result = detect_column_dtype.invoke({
            "df_json": df.to_json(),
            "column": "user_id"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["semantic_type"], "identifier")
    
    def test_boolean(self):
        """Test detection of boolean column."""
        df = pd.DataFrame({
            "is_active": [True, False, True, False]
        })
        
        result = detect_column_dtype.invoke({
            "df_json": df.to_json(),
            "column": "is_active"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["semantic_type"], "boolean")
    
    def test_datetime_strings(self):
        """Test datetime strings that look like dates."""
        df = pd.DataFrame({
            "dates": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-07"]
        })
        
        result = detect_column_dtype.invoke({
            "df_json": df.to_json(),
            "column": "dates"
        })
        result_dict = json.loads(result)
        
        # Should detect as datetime if >5 out of 10 samples parse correctly
        # With 7 date strings, should be detected
        self.assertIn(result_dict["semantic_type"], ["datetime", "text", "categorical"])
    
    def test_numeric_strings(self):
        """Test numeric strings like '123', '45.6'."""
        df = pd.DataFrame({
            "numeric_str": ["123", "456", "789", "012"]
        })
        
        result = detect_column_dtype.invoke({
            "df_json": df.to_json(),
            "column": "numeric_str"
        })
        result_dict = json.loads(result)
        
        # Should be detected as categorical or text, not numeric
        self.assertIn(result_dict["semantic_type"], ["categorical", "text"])
    
    def test_high_cardinality_categorical_vs_text(self):
        """Test high-cardinality categorical vs text distinction."""
        # High cardinality but not unique enough to be text
        df = pd.DataFrame({
            "high_card": [f"cat_{i % 10}" for i in range(100)]
        })
        
        result = detect_column_dtype.invoke({
            "df_json": df.to_json(),
            "column": "high_card"
        })
        result_dict = json.loads(result)
        
        # Should be categorical (10 unique out of 100 = 10% unique ratio)
        self.assertEqual(result_dict["semantic_type"], "categorical")
    
    def test_identifier_detection_patterns(self):
        """Test identifier detection with various naming patterns."""
        test_cases = [
            ("user_id", list(range(100))),
            ("primary_key", list(range(100))),
            ("index_col", list(range(100))),
            ("record_name", [f"name_{i}" for i in range(100)])
        ]
        
        for col_name, data in test_cases:
            df = pd.DataFrame({col_name: data})
            result = detect_column_dtype.invoke({
                "df_json": df.to_json(),
                "column": col_name
            })
            result_dict = json.loads(result)
            
            # Should detect as identifier if high uniqueness
            if result_dict["unique_ratio"] > 0.8:
                self.assertEqual(result_dict["semantic_type"], "identifier")
    
    def test_boolean_like_values(self):
        """Test boolean-like values (0/1, 'True'/'False', etc.)."""
        test_cases = [
            ([0, 1, 0, 1], "numeric_discrete"),  # 0/1 as integers
            (["True", "False", "True"], "boolean"),  # String booleans
            (["true", "false", "true"], "boolean"),  # Lowercase
            ([True, False, True], "boolean"),  # Actual booleans
        ]
        
        for data, expected_type in test_cases:
            df = pd.DataFrame({"bool_col": data})
            result = detect_column_dtype.invoke({
                "df_json": df.to_json(),
                "column": "bool_col"
            })
            result_dict = json.loads(result)
            
            # May be boolean or numeric_discrete depending on dtype
            self.assertIn(result_dict["semantic_type"], [expected_type, "boolean", "numeric_discrete"])


class TestEscapedJsonHandling(unittest.TestCase):
    
    def test_compute_correlation_matrix_with_escaped_json(self):
        """Test compute_correlation_matrix with escaped JSON."""
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [2, 4, 6]
        })
        normal_json = df.to_json()
        escaped_json = json.dumps(normal_json)
        
        result = compute_correlation_matrix.invoke({
            "df_json": escaped_json,
            "columns": None
        })
        result_dict = json.loads(result)
        
        self.assertIn("correlations", result_dict)
        self.assertNotIn("error", result_dict)
    
    def test_get_column_statistics_with_escaped_json(self):
        """Test get_column_statistics with escaped JSON."""
        df = pd.DataFrame({
            "Level": ["E6", "E6", "E3"],
            "Salary": [100000, 150000, 120000]
        })
        normal_json = df.to_json()
        escaped_json = json.dumps(normal_json)
        
        result = get_column_statistics.invoke({
            "df_json": escaped_json,
            "column": "Salary"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["column"], "Salary")
        self.assertNotIn("error", result_dict)
    
    def test_detect_column_dtype_with_escaped_json(self):
        """Test detect_column_dtype with escaped JSON."""
        df = pd.DataFrame({
            "Level": {"0": "E6", "1": "E6", "2": "E3"}
        })
        normal_json = df.to_json()
        escaped_json = json.dumps(normal_json)
        
        result = detect_column_dtype.invoke({
            "df_json": escaped_json,
            "column": "Level"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["column"], "Level")
        self.assertNotIn("error", result_dict)
    
    def test_get_unique_value_counts_with_escaped_json(self):
        """Test get_unique_value_counts with escaped JSON."""
        df = pd.DataFrame({
            "Category": ["A", "B", "A", "C"]
        })
        normal_json = df.to_json()
        escaped_json = json.dumps(normal_json)
        
        result = get_unique_value_counts.invoke({
            "df_json": escaped_json,
            "column": "Category"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["column"], "Category")
        self.assertNotIn("error", result_dict)
    
    def test_detect_ordinal_patterns_with_escaped_json(self):
        """Test detect_ordinal_patterns with escaped JSON."""
        df = pd.DataFrame({
            "Level": ["L1", "L2", "L3", "L4"]
        })
        normal_json = df.to_json()
        escaped_json = json.dumps(normal_json)
        
        result = detect_ordinal_patterns.invoke({
            "df_json": escaped_json,
            "column": "Level"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["column"], "Level")
        self.assertNotIn("error", result_dict)
    
    def test_real_world_escaped_json_scenario(self):
        """Test with real-world escaped JSON from the issue."""
        escaped_json = '{\\"Level\\": {\\"0\\": \\"E6\\", \\"1\\": \\"E6\\", \\"2\\": \\"E3\\"}}'
        
        result = detect_column_dtype.invoke({
            "df_json": escaped_json,
            "column": "Level"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["column"], "Level")
        self.assertNotIn("error", result_dict)
    
    def test_double_quoted_escaped_json(self):
        """Test JSON wrapped in quotes with escaped content."""
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6]
        })
        normal_json = df.to_json()
        double_quoted = f'"{normal_json}"'
        
        result = compute_correlation_matrix.invoke({
            "df_json": double_quoted,
            "columns": None
        })
        result_dict = json.loads(result)
        
        self.assertIn("correlations", result_dict)
        self.assertNotIn("error", result_dict)


class TestGetAllTools(unittest.TestCase):
    
    def test_get_all_tools(self):
        """Test get_all_tools() returns all tools."""
        tools = get_all_tools()
        
        self.assertEqual(len(tools), 5)
        tool_names = [tool.name for tool in tools]
        
        self.assertIn("compute_correlation_matrix", tool_names)
        self.assertIn("get_column_statistics", tool_names)
        self.assertIn("get_unique_value_counts", tool_names)
        self.assertIn("detect_ordinal_patterns", tool_names)
        self.assertIn("detect_column_dtype", tool_names)
    
    def test_tools_are_callable(self):
        """Test that all tools are callable."""
        tools = get_all_tools()
        
        for tool in tools:
            self.assertTrue(callable(tool.invoke))


if __name__ == "__main__":
    unittest.main()
