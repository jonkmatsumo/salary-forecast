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

## Configuration

The system is now fully configurable via `config.json`. You can modify:
- **Mappings**: Define level mappings (e.g., E3 -> 0).
- **Location Targets**: Define major cities and their associated Cost Zones.
- **Model Parameters**: Set targets (e.g., BaseSalary), quantiles, and sample weights.
- **Features**: Define input features and monotonic constraints.

## Proximity Matching

The system uses `geopy` to automatically map input locations to the nearest target city defined in `config.json`.
- **Dynamic Matching**: "Newark" maps to "New York" (Zone 1) because it is within the configured `max_distance_km` (default 50km).
- **Caching**: Geocoding results are cached in `city_cache.json` to speed up subsequent runs and reduce API usage.
- **O(1) Lookup**: Once a city is processed, its zone is cached in memory for instant lookup.

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
