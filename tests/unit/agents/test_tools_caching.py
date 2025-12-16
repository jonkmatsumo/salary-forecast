"""Tests for correlation matrix caching."""

import json

import pandas as pd
import pytest

from src.agents.tools import compute_correlation_matrix
from src.utils.cache_manager import get_cache_manager


class TestCorrelationMatrixCaching:
    """Test suite for correlation matrix caching."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame for testing."""
        return pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [2, 4, 6, 8, 10], "C": [1, 1, 1, 1, 1]})

    def test_cache_hit_returns_cached_result(self, sample_df: pd.DataFrame) -> None:
        """Test cache hit returns cached correlation matrix."""
        cache_manager = get_cache_manager()
        cache_manager.clear("correlation")

        df_json = sample_df.to_json()
        result1 = compute_correlation_matrix.invoke({"df_json": df_json})
        result2 = compute_correlation_matrix.invoke({"df_json": df_json})

        assert result1 == result2
        assert json.loads(result1) == json.loads(result2)

    def test_cache_miss_computes_new_result(self, sample_df: pd.DataFrame) -> None:
        """Test cache miss computes new correlation matrix."""
        cache_manager = get_cache_manager()
        cache_manager.clear("correlation")

        df1_json = sample_df.to_json()
        df2 = pd.DataFrame({"X": [1, 2, 3], "Y": [4, 5, 6]})
        df2_json = df2.to_json()

        result1 = compute_correlation_matrix.invoke({"df_json": df1_json})
        result2 = compute_correlation_matrix.invoke({"df_json": df2_json})

        assert result1 != result2
        data1 = json.loads(result1)
        data2 = json.loads(result2)
        assert data1["columns_analyzed"] != data2["columns_analyzed"]

    def test_cache_key_includes_columns_parameter(self, sample_df: pd.DataFrame) -> None:
        """Test cache key includes columns parameter."""
        cache_manager = get_cache_manager()
        cache_manager.clear("correlation")

        df_json = sample_df.to_json()
        result1 = compute_correlation_matrix.invoke({"df_json": df_json, "columns": "A, B"})
        result2 = compute_correlation_matrix.invoke({"df_json": df_json, "columns": "A, C"})

        assert result1 != result2

    def test_same_input_same_columns_uses_cache(self, sample_df: pd.DataFrame) -> None:
        """Test same input with same columns uses cache."""
        cache_manager = get_cache_manager()
        cache_manager.clear("correlation")

        df_json = sample_df.to_json()
        result1 = compute_correlation_matrix.invoke({"df_json": df_json, "columns": "A, B"})
        result2 = compute_correlation_matrix.invoke({"df_json": df_json, "columns": "A, B"})

        assert result1 == result2

    def test_cache_stores_json_string(self, sample_df: pd.DataFrame) -> None:
        """Test cache stores JSON string result."""
        cache_manager = get_cache_manager()
        cache_manager.clear("correlation")

        df_json = sample_df.to_json()
        result = compute_correlation_matrix.invoke({"df_json": df_json})

        assert isinstance(result, str)
        data = json.loads(result)
        assert "columns_analyzed" in data
        assert "correlations" in data
        assert "full_matrix" in data

    def test_error_responses_not_cached(self, sample_df: pd.DataFrame) -> None:
        """Test error responses are not cached."""
        cache_manager = get_cache_manager()
        cache_manager.clear("correlation")

        invalid_json = "not valid json"
        result1 = compute_correlation_matrix.invoke({"df_json": invalid_json})
        result2 = compute_correlation_matrix.invoke({"df_json": invalid_json})

        data1 = json.loads(result1)
        data2 = json.loads(result2)
        assert "error" in data1
        assert "error" in data2

    def test_different_dataframes_different_cache_keys(self) -> None:
        """Test different DataFrames produce different cache keys."""
        cache_manager = get_cache_manager()
        cache_manager.clear("correlation")

        df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df2 = pd.DataFrame({"X": [1, 2, 3], "Y": [4, 5, 6]})

        result1 = compute_correlation_matrix.invoke({"df_json": df1.to_json()})
        result2 = compute_correlation_matrix.invoke({"df_json": df2.to_json()})

        assert result1 != result2
