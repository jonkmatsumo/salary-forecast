import pandas as pd
import pickle
import os
from src.model import SalaryForecaster
from src.utils import load_data

def main():
    csv_path = "src/salaries-list.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading data from {csv_path}...")
    df = load_data(csv_path)
    print(f"Loaded {len(df)} samples.")
    
    print("Initializing model...")
    forecaster = SalaryForecaster()
    
    print("Training model...")
    forecaster.train(df)
    
    print("Saving model...")
    with open("salary_model.pkl", "wb") as f:
        pickle.dump(forecaster, f)
    print("Model saved to salary_model.pkl")
    
    # Simple inference check
    print("\nRunning sample inference...")
    sample_input = pd.DataFrame([{
        "Level": "E5",
        "Location": "New York",
        "YearsOfExperience": 8,
        "YearsAtCompany": 2
    }])
    
    prediction = forecaster.predict(sample_input)
    print("Prediction for E5 in NY (8 YOE):")
    for target, preds in prediction.items():
        print(f"  {target}: P25={preds['p25'][0]:.0f}, P50={preds['p50'][0]:.0f}, P75={preds['p75'][0]:.0f}")

if __name__ == "__main__":
    main()
