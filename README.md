# Salary Forecasting Engine

A machine learning system to predict compensation distributions (Base Salary, Stock, Bonus, Total Comp) based on candidate attributes. It uses **XGBoost** with Quantile Regression to forecast the 25th, 50th, and 75th percentiles, enforcing monotonic constraints on Level and Years of Experience.

## Project Structure

- `src/`: Source code for the package.
    - `model.py`: Core `SalaryForecaster` class.
    - `preprocessing.py`: Encoders and data transformation logic.
    - `utils.py`: Data loading and cleaning utilities.
- `tests/`: Unit tests.
- `train.py`: Script to train the model and save it to `salary_model.pkl`.
- `requirements.txt`: Project dependencies.

## Installation

1.  Create a virtual environment (optional but recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training
To train the model using the data in `src/salaries-list.csv`:

```bash
python3 train.py
```

This will output training progress and save the trained model to `salary_model.pkl`.

### Testing
To run the unit tests:

```bash
python3 -m pytest tests/test_model.py
```
