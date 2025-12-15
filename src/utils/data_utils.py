from io import BytesIO
from typing import Union

import pandas as pd


def load_data(filepath: Union[str, BytesIO]) -> pd.DataFrame:
    """Load raw data from CSV without preprocessing. Args: filepath (Union[str, BytesIO]): CSV file path or file-like object. Returns: pd.DataFrame: Raw DataFrame."""
    return pd.read_csv(filepath)
