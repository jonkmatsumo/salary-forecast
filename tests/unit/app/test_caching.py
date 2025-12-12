from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.app.caching import load_data_cached


@patch("src.app.caching.original_load_data")
def test_load_data_cached_calls_original(mock_load):
    """Verify that the cached function calls the underlying data loader."""
    # Setup
    mock_df = pd.DataFrame({"col": [1, 2]})
    mock_load.return_value = mock_df
    dummy_file = "dummy.csv"

    # Execute
    result = load_data_cached(dummy_file)

    # Assert
    pd.testing.assert_frame_equal(result, mock_df)
    mock_load.assert_called_once_with(dummy_file)


def test_caching_decorator_present():
    """Verify that the function is actually decorated with cache_data."""
    # Access the underlying streamit cache registry or attributes if possible.
    # Or just check if it has Streamlit attributes.
    # Streamlit caching decorates the function object.
    # In recent versions, it might be an instance of internal class.

    # We can check if it has expected attributes like clear() or invalidate().
    # @st.cache_data functions usually have a .clear() method.
    assert hasattr(load_data_cached, "clear")
