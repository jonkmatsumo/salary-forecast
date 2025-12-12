import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw data from CSV without preprocessing. Args: filepath (str): CSV file path. Returns: pd.DataFrame: Raw DataFrame."""
    return pd.read_csv(filepath)
