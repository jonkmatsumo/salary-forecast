import streamlit as st
import pandas as pd
from typing import Union
from io import BytesIO

# Import the original load_data function
from src.utils.data_utils import load_data as original_load_data

@st.cache_data(show_spinner="Loading data...", ttl="1h")
def load_data_cached(file: Union[str, BytesIO]) -> pd.DataFrame:
    """
    Cached wrapper for load_data.
    Uses cached result if file content hasn't changed (Streamlit handles hashing of file-like objects).
    """
    return original_load_data(file)
