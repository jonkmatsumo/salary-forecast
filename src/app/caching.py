from io import BytesIO
from typing import Union

import pandas as pd
import streamlit as st

from src.utils.data_utils import load_data as original_load_data


@st.cache_data(show_spinner="Loading data...", ttl="1h")
def load_data_cached(file: Union[str, BytesIO]) -> pd.DataFrame:
    """Cached wrapper for load_data.

    Args:
        file (Union[str, BytesIO]): File path or file-like object.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return original_load_data(file)
