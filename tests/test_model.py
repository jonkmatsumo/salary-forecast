import pytest
import pandas as pd
import numpy as np
from src.model import SalaryForecaster

@pytest.fixture(scope="module")
def trained_model():
    # Create a small manual dataset for testing constraints
    data = []
    levels = ["E3", "E4", "E5", "E6", "E7"]
    locations = ["New York", "San Francisco", "Seattle", "Austin"]
    
    for _ in range(200): # Smaller dataset for unit tests
        for level_idx, level in enumerate(levels):
            for loc in locations:
                base = 100000 + level_idx * 50000
                if loc == "New York": base *= 1.2
                
                data.append({
                    "Level": level,
                    "Location": loc,
                    "Date": pd.Timestamp.now(),
                    "YearsOfExperience": level_idx * 3 + 2,
                    "YearsAtCompany": 2,
                    "BaseSalary": base,
                    "Stock": base * 0.5,
                    "Bonus": base * 0.1,
                    "TotalComp": base * 1.6
                })
    
    df = pd.DataFrame(data)
    model = SalaryForecaster()
    model.train(df)
    return model

def test_monotonicity_level(trained_model):
    """
    Verify that higher levels result in higher (or equal) salary, holding other factors constant.
    """
    # Create base input
    base_input = {
        "Location": "New York",
        "YearsOfExperience": 5,
        "YearsAtCompany": 2
    }
    
    levels = ["E3", "E4", "E5", "E6", "E7"]
    preds = []
    
    for level in levels:
        inp = pd.DataFrame([{**base_input, "Level": level}])
        res = trained_model.predict(inp)
        preds.append(res["BaseSalary"]["p50"][0])
        
    # Check if sorted
    print(f"\nLevel Predictions (E3-E7): {preds}")
    assert np.all(np.diff(preds) >= -1e-9), "Base Salary P50 should be monotonic with Level"

def test_monotonicity_yoe(trained_model):
    """
    Verify that more experience results in higher (or equal) salary.
    """
    # Create base input
    base_input = {
        "Level": "E5",
        "Location": "New York",
        "YearsAtCompany": 2
    }
    
    yoes = [2, 5, 8, 12, 15]
    preds = []
    
    for yoe in yoes:
        inp = pd.DataFrame([{**base_input, "YearsOfExperience": yoe}])
        res = trained_model.predict(inp)
        preds.append(res["BaseSalary"]["p50"][0])
        
    # Check if sorted
    print(f"\nYOE Predictions (2, 5, 8, 12, 15): {preds}")
    assert np.all(np.diff(preds) >= -1e-9), "Base Salary P50 should be monotonic with YOE"

def test_quantiles_ordering(trained_model):
    """
    Verify P25 <= P50 <= P75
    """
    inp = pd.DataFrame([{
        "Level": "E5",
        "Location": "New York",
        "YearsOfExperience": 8,
        "YearsAtCompany": 2
    }])
    
    res = trained_model.predict(inp)
    
    for target in res:
        p25 = res[target]["p25"][0]
        p50 = res[target]["p50"][0]
        p75 = res[target]["p75"][0]
        
        assert p25 <= p50, f"{target}: P25 ({p25}) > P50 ({p50})"
        assert p50 <= p75, f"{target}: P50 ({p50}) > P75 ({p75})"

def test_location_impact(trained_model):
    """
    Verify NY > Austin (Zone 1 > Zone 3)
    """
    inp_ny = pd.DataFrame([{
        "Level": "E5", "Location": "New York", "YearsOfExperience": 8, "YearsAtCompany": 2
    }])
    inp_austin = pd.DataFrame([{
        "Level": "E5", "Location": "Austin", "YearsOfExperience": 8, "YearsAtCompany": 2
    }])
    
    pred_ny = trained_model.predict(inp_ny)["BaseSalary"]["p50"][0]
    pred_austin = trained_model.predict(inp_austin)["BaseSalary"]["p50"][0]
    
    print(f"\nLocation Check: NY={pred_ny}, Austin={pred_austin}")
    assert pred_ny > pred_austin, "NY salary should be higher than Austin"
