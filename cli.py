import pickle
import os
import pandas as pd
import sys

def load_model(path="salary_model.pkl"):
    if not os.path.exists(path):
        print(f"Error: Model file '{path}' not found. Please run train.py first.")
        sys.exit(1)
    
    with open(path, "rb") as f:
        return pickle.load(f)

def get_input(prompt, type_func=str, valid_options=None):
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input:
                continue
                
            val = type_func(user_input)
            
            if valid_options and val not in valid_options:
                print(f"Invalid option. Please choose from: {', '.join(map(str, valid_options))}")
                continue
                
            return val
        except ValueError:
            print(f"Invalid input. Please enter a valid {type_func.__name__}.")

def format_currency(val):
    return f"${val:,.0f}"

def collect_user_data():
    print("\n--- Enter Candidate Details ---")
    level = get_input("Level (e.g. E3, E4, E5, E6, E7): ", str, ["E3", "E4", "E5", "E6", "E7"])
    location = get_input("Location (e.g. New York, San Francisco): ", str)
    yoe = get_input("Years of Experience: ", int)
    yac = get_input("Years at Company: ", int)
    
    return pd.DataFrame([{
        "Level": level,
        "Location": location,
        "YearsOfExperience": yoe,
        "YearsAtCompany": yac
    }])

def main():
    print("Welcome to the Salary Forecasting CLI")
    model = load_model()
    
    while True:
        try:
            input_df = collect_user_data()
            
            print("\nCalculating prediction...")
            results = model.predict(input_df)
            
            print("\n--- Prediction Results ---")
            for target, preds in results.items():
                p25 = format_currency(preds['p25'][0])
                p50 = format_currency(preds['p50'][0])
                p75 = format_currency(preds['p75'][0])
                print(f"{target}:")
                print(f"  25th Percentile: {p25}")
                print(f"  50th Percentile: {p50}")
                print(f"  75th Percentile: {p75}")
                
            cont = input("\nForecast another? (y/n): ").strip().lower()
            if cont != 'y':
                print("Goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
