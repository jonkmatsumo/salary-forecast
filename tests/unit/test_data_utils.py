import pytest
import pandas as pd
import os
from src.utils.data_utils import load_data

def test_load_data(tmp_path):
    # Create a dummy CSV
    csv_content = """Level,Location,YearsOfExperience,YearsAtCompany,BaseSalary,Stock,Bonus,TotalComp,Date
E3,NY,2,1,100000,50000,10000,160000,2023-01-01
E4,SF,5-10,3+,150000,80000,20000,250000,2023-02-01
"""
    csv_file = tmp_path / "test_salaries.csv"
    csv_file.write_text(csv_content)
    
    df = load_data(str(csv_file))
    
    assert len(df) == 2
    
    # Check numeric cleaning
    # "5-10" -> 7.5
    assert df.iloc[1]["YearsOfExperience"] == 7.5
    # "3+" -> 3.0
    assert df.iloc[1]["YearsAtCompany"] == 3.0
    
    # Check date parsing
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    
    # Check numeric targets
    assert df.iloc[0]["BaseSalary"] == 100000
    assert df.iloc[0]["TotalComp"] == 160000

def test_load_data_mixed_dates(tmp_path):
    """Test loading data with various date formats."""
    csv_content = """Level,Location,YearsOfExperience,YearsAtCompany,BaseSalary,Stock,Bonus,TotalComp,Date
E3,NY,1,0,100,0,0,100,2023-01-01
E3,NY,1,0,100,0,0,100,"Jan 15, 2023"
E3,NY,1,0,100,0,0,100,02/01/2023
E3,NY,1,0,100,0,0,100,Mar 2023
"""
    csv_file = tmp_path / "mixed_dates.csv"
    csv_file.write_text(csv_content)
    
    df = load_data(str(csv_file))
    
    assert len(df) == 4
    assert not df["Date"].isnull().any(), "Some dates failed to parse"
    
    # Verify specific dates
    # Jan 15, 2023
    assert df.iloc[1]["Date"].month == 1
    assert df.iloc[1]["Date"].day == 15
    
    # 02/01/2023 (assuming M/D/Y default or smart parsing finding day 1 vs month 2)
    # With format='mixed', it usually guesses M/D/Y for US-like strings
    assert df.iloc[2]["Date"].month == 2 
    assert df.iloc[2]["Date"].day == 1
    
    # Mar 2023 -> defaults to 1st of month usually
    assert df.iloc[3]["Date"].month == 3
    assert df.iloc[3]["Date"].year == 2023
