from src.utils.data_utils import load_data


def test_load_data(tmp_path):
    """Verify load_data loads raw CSV without preprocessing."""
    csv_content = """Level,Location,YearsOfExperience,YearsAtCompany,BaseSalary,Stock,Bonus,TotalComp,Date
E3,NY,2,1,100000,50000,10000,160000,2023-01-01
E4,SF,5-10,3+,150000,80000,20000,250000,2023-02-01
"""
    csv_file = tmp_path / "test_salaries.csv"
    csv_file.write_text(csv_content)

    df = load_data(str(csv_file))

    assert len(df) == 2

    # Verify raw data is loaded (no preprocessing)
    # "5-10" should remain as string
    assert df.iloc[1]["YearsOfExperience"] == "5-10"
    # "3+" should remain as string
    assert df.iloc[1]["YearsAtCompany"] == "3+"

    # Date should be string, not parsed
    assert isinstance(df.iloc[0]["Date"], str)

    # Numeric columns should be numeric (pandas auto-converts)
    assert df.iloc[0]["BaseSalary"] == 100000
    assert df.iloc[0]["TotalComp"] == 160000


def test_load_data_mixed_dates(tmp_path):
    """Verify load_data loads raw data without date parsing."""
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
    # Dates should be strings (raw data)
    assert all(isinstance(d, str) for d in df["Date"])
