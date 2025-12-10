import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.xgboost.preprocessing import RankedCategoryEncoder, ProximityEncoder, SampleWeighter

# --- RankedCategoryEncoder Tests ---

def test_ranked_encoder_transform():
    mapping = {"E3": 0, "E4": 1, "E5": 2}
    
    # Test with direct mapping
    encoder = RankedCategoryEncoder(mapping=mapping)
    
    # Test Series input
    X = pd.Series(["E3", "E5", "E4", "Unknown"])
    result = encoder.transform(X)
    
    expected = np.array([0, 2, 1, -1])
    np.testing.assert_array_equal(result, expected)

def test_ranked_encoder_dataframe_input():
    mapping = {"E3": 0}
    encoder = RankedCategoryEncoder(mapping=mapping)
    df = pd.DataFrame({"Level": ["E3"]})
    result = encoder.transform(df)
    assert result[0] == 0

# --- ProximityEncoder Tests ---

def test_proximity_encoder_transform():
    # Mock GeoMapper
    with patch('src.xgboost.preprocessing.GeoMapper') as MockGeoMapper:
        mock_mapper = MockGeoMapper.return_value
        # Setup mock behavior: NY -> 1, SF -> 2, Unknown -> 4
        mock_mapper.get_zone.side_effect = lambda x: 1 if x == "NY" else (2 if x == "SF" else 4)
        
        encoder = ProximityEncoder()
        
        X = pd.Series(["NY", "SF", "Other", 123]) # 123 to test non-string
        result = encoder.transform(X)
        
        expected = np.array([1, 2, 4, 4])
        np.testing.assert_array_equal(result, expected)

# --- SampleWeighter Tests ---

def test_sample_weighter_transform():
    # Test with explicit k
    weighter = SampleWeighter(k=1.0, ref_date="2023-01-01")
    
    # Dates: 0 years old, 1 year old, 2 years old
    dates = pd.Series(["2023-01-01", "2022-01-01", "2021-01-01"])
    weights = weighter.transform(dates)
    
    # Expected: 1/(1+0)^1 = 1, 1/(1+1)^1 = 0.5, 1/(1+2)^1 = 0.333
    np.testing.assert_almost_equal(weights[0], 1.0)
    np.testing.assert_almost_equal(weights[1], 0.5, decimal=3)
    np.testing.assert_almost_equal(weights[2], 1/3, decimal=3)

def test_sample_weighter_future_dates():
    # Future dates should be treated as age 0
    weighter = SampleWeighter(k=1.0, ref_date="2023-01-01")
    dates = pd.Series(["2024-01-01"])
    weights = weighter.transform(dates)
    assert weights[0] == 1.0

def test_ranked_encoder_edge_cases():
    mapping = {"E3": 0}
    encoder = RankedCategoryEncoder(mapping=mapping)
    
    # Test with None, NaN, Empty string - should map to -1 (unknown)
    X = pd.Series([None, np.nan, "", "Unknown"])
    result = encoder.transform(X)
    
    expected = np.array([-1, -1, -1, -1])
    np.testing.assert_array_equal(result, expected)

def test_proximity_encoder_edge_cases():
    with patch('src.xgboost.preprocessing.GeoMapper') as MockGeoMapper:
        mock_mapper = MockGeoMapper.return_value
        # If input is not a string (e.g. NaN), get_zone might not even be called if we handle it in transform
        # or we rely on get_zone to handle it. 
        # Looking at implementation: 
        # def map_loc(loc):
        #     if not isinstance(loc, str): return 4
        #     return self.mapper.get_zone(loc)
        
        encoder = ProximityEncoder()
        
        # None, NaN -> Should return 4 (Unknown) without calling mapper.get_zone
        X = pd.Series([None, np.nan, 123])
        result = encoder.transform(X)
        
        expected = np.array([4, 4, 4])
        np.testing.assert_array_equal(result, expected)
        
        # Empty string -> Should call mapper.get_zone("") -> let's say mapper returns 4 for empty
        mock_mapper.get_zone.return_value = 4
        
        X_str = pd.Series(["", "   "])
        result_str = encoder.transform(X_str)
        
        expected_str = np.array([4, 4])
        np.testing.assert_array_equal(result_str, expected_str)

def test_sample_weighter_edge_cases():
    weighter = SampleWeighter(k=1.0)
    
    # NaT handling
    dates = pd.Series([pd.NaT, "2023-01-01"])
    # Age of NaT will be NaT/NaN. 1/(1+NaN)^k = NaN
    weights = weighter.transform(dates)
    
    assert np.isnan(weights[0])
    assert not np.isnan(weights[1])

def test_sample_weighter_k_zero():
    # If k=0, weights should always be 1.0 regardless of age
    weighter = SampleWeighter(k=0.0)
    dates = pd.Series(["2020-01-01", "2023-01-01"])
    weights = weighter.transform(dates)
    
    np.testing.assert_array_equal(weights, np.array([1.0, 1.0]))


def test_ranked_encoder_with_config_key():
    """Test RankedCategoryEncoder with config_key parameter."""
    with patch("src.xgboost.preprocessing.get_config") as mock_get_config:
        mock_config = {
            "mappings": {
                "levels": {"E3": 0, "E4": 1, "E5": 2}
            }
        }
        mock_get_config.return_value = mock_config
        
        encoder = RankedCategoryEncoder(config_key="levels")
        
        # Verify mapping was loaded from config
        assert encoder.mapping == {"E3": 0, "E4": 1, "E5": 2}
        
        # Test transform works
        X = pd.Series(["E3", "E4", "E5"])
        result = encoder.transform(X)
        expected = np.array([0, 1, 2])
        np.testing.assert_array_equal(result, expected)


def test_ranked_encoder_config_key_missing():
    """Test RankedCategoryEncoder with missing config key."""
    with patch("src.xgboost.preprocessing.get_config") as mock_get_config:
        mock_config = {"mappings": {}}
        mock_get_config.return_value = mock_config
        
        encoder = RankedCategoryEncoder(config_key="missing_key")
        
        # Should have empty mapping
        assert encoder.mapping == {}
        
        # Transform should return -1 for all values
        X = pd.Series(["E3", "E4"])
        result = encoder.transform(X)
        expected = np.array([-1, -1])
        np.testing.assert_array_equal(result, expected)


def test_ranked_encoder_empty_mapping():
    """Test RankedCategoryEncoder with empty mapping."""
    encoder = RankedCategoryEncoder(mapping={})
    
    X = pd.Series(["E3", "E4"])
    result = encoder.transform(X)
    
    # All should map to -1 (unknown)
    expected = np.array([-1, -1])
    np.testing.assert_array_equal(result, expected)


def test_sample_weighter_dataframe_with_date_col():
    """Test SampleWeighter with DataFrame input using date_col parameter."""
    weighter = SampleWeighter(k=1.0, ref_date="2023-01-01", date_col="Date")
    
    df = pd.DataFrame({
        "Date": ["2023-01-01", "2022-01-01", "2021-01-01"],
        "Other": [1, 2, 3]
    })
    
    weights = weighter.transform(df)
    
    # Should extract Date column and calculate weights
    np.testing.assert_almost_equal(weights[0], 1.0)
    np.testing.assert_almost_equal(weights[1], 0.5, decimal=3)
    np.testing.assert_almost_equal(weights[2], 1/3, decimal=3)


def test_sample_weighter_dataframe_missing_date_col():
    """Test SampleWeighter with DataFrame missing date_col falls back to first column."""
    weighter = SampleWeighter(k=1.0, ref_date="2023-01-01", date_col="MissingCol")
    
    df = pd.DataFrame({
        "Date": ["2023-01-01", "2022-01-01"],
        "Other": [1, 2]
    })
    
    # Should fall back to first column
    weights = weighter.transform(df)
    
    # Should still work (uses first column as fallback)
    assert len(weights) == 2


def test_ranked_encoder_fit():
    """Test RankedCategoryEncoder.fit() method (no-op, returns self)."""
    encoder = RankedCategoryEncoder(mapping={"E3": 0, "E4": 1})
    
    X = pd.Series(["E3", "E4"])
    y = pd.Series([100, 200])
    
    result = encoder.fit(X, y)
    
    # Should return self
    assert result is encoder
    
    # Mapping should be unchanged
    assert encoder.mapping == {"E3": 0, "E4": 1}


def test_proximity_encoder_fit():
    """Test ProximityEncoder.fit() method (no-op, returns self)."""
    with patch('src.xgboost.preprocessing.GeoMapper'):
        encoder = ProximityEncoder()
        
        X = pd.Series(["NY", "SF"])
        y = pd.Series([100, 200])
        
        result = encoder.fit(X, y)
        
        # Should return self
        assert result is encoder


def test_sample_weighter_fit():
    """Test SampleWeighter.fit() method (no-op, returns self)."""
    weighter = SampleWeighter(k=1.0, ref_date="2023-01-01")
    
    X = pd.Series(["2023-01-01", "2022-01-01"])
    y = pd.Series([100, 200])
    
    result = weighter.fit(X, y)
    
    # Should return self
    assert result is weighter
    
    # Parameters should be unchanged
    assert weighter.k == 1.0


def test_ranked_encoder_list_input():
    """Test RankedCategoryEncoder with list input."""
    encoder = RankedCategoryEncoder(mapping={"E3": 0, "E4": 1})
    
    X = ["E3", "E4", "E5"]
    result = encoder.transform(X)
    
    expected = np.array([0, 1, -1])
    np.testing.assert_array_equal(result, expected)


def test_sample_weighter_config_key():
    """Test SampleWeighter loads k from config when not provided."""
    with patch("src.xgboost.preprocessing.get_config") as mock_get_config:
        mock_config = {
            "model": {
                "sample_weight_k": 2.0
            }
        }
        mock_get_config.return_value = mock_config
        
        weighter = SampleWeighter()
        
        # Should load k from config
        assert weighter.k == 2.0
