import pandas as pd
import numpy as np
import re

from typing import Union

def load_data(filepath: str) -> pd.DataFrame:
    """Load and clean salary data from CSV. Args: filepath (str): CSV file path. Returns: pd.DataFrame: Cleaned DataFrame."""
    df = pd.read_csv(filepath)
    
    def clean_years(val: Union[int, float, str]) -> float:
        """Parse year strings like '11+' or '5-10' into floats. Args: val (Union[int, float, str]): Year value. Returns: float: Parsed year."""
        if isinstance(val, (int, float)):
            return float(val)
        val = str(val).strip()
        if "+" in val:
            return float(val.replace("+", ""))
        if "-" in val:
            parts = val.split("-")
            return (float(parts[0]) + float(parts[1])) / 2
        return float(val)

    df["YearsOfExperience"] = df["YearsOfExperience"].apply(clean_years)
    df["YearsAtCompany"] = df["YearsAtCompany"].apply(clean_years)
    
    try:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce', format='mixed')
    except ValueError:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    
    targets = ["BaseSalary", "Stock", "Bonus", "TotalComp"]
    for col in targets:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    return df
