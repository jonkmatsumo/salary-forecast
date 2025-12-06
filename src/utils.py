import pandas as pd
import numpy as np
import re

def load_data(filepath):
    """
    Loads and cleans the salary data from CSV.
    """
    df = pd.read_csv(filepath)
    
    # Columns are already renamed in the CSV to match model expectations
    # Level,TotalComp,BaseSalary,Stock,Bonus,YearsOfExperience,YearsAtCompany,Date,Location
    
    # Clean numeric columns that might have strings like "11+" or "5-10"
    def clean_years(val):
        if isinstance(val, (int, float)):
            return float(val)
        val = str(val).strip()
        if "+" in val:
            return float(val.replace("+", ""))
        if "-" in val:
            # Take average of range
            parts = val.split("-")
            return (float(parts[0]) + float(parts[1])) / 2
        return float(val)

    df["YearsOfExperience"] = df["YearsOfExperience"].apply(clean_years)
    df["YearsAtCompany"] = df["YearsAtCompany"].apply(clean_years)
    
    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Ensure numeric targets
    targets = ["BaseSalary", "Stock", "Bonus", "TotalComp"]
    for col in targets:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    return df
