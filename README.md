# Salary Forecasting Engine

A machine learning system to predict compensation distributions (Base Salary, Stock, Bonus, Total Comp) based on candidate attributes. It uses **XGBoost** with Quantile Regression. The model is highly configurable, allowing users to define exact percentiles to forecast, enforce monotonic constraints, and customize target definitions.

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

### Web Application (Streamlit)
The easiest way to use the system is via the web interface:

```bash
streamlit run src/app/app.py
```

This launches a dashboard where you can:
- **Train Models**: Upload a CSV, adjust configurations (like quantiles), and train new models interactively.
- **Run Inference**: Select a trained model, enter candidate details, and visualize the predicted salary distribution.

### Training (CLI)
To train the model via terminal:

```bash
python3 -m src.cli.train_cli
```

You can specify the input CSV, config file, and output model path.

### Inference (CLI)
To run the interactive CLI for making predictions:

```bash
python3 -m src.cli.inference_cli
```

Follow the prompts to select a model and enter candidate details.

## Testing

To run the unit tests:

```bash
python3 -m pytest tests/
```

## Configuration

The model and data processing are configurable through a structured dictionary. This configuration can be edited **inline** directly in the web app, or provided as a `config.json` file when using the CLI.

### Configuration Structure

The configuration dictionary should follow this structure:

#### 1. Mappings (`mappings`)
Defines how categorical data is mapped to numerical values.

- **`levels`**: Maps job levels (e.g., "E3", "E4") to ordinal integers (0, 1, 2...).
  ```json
  "levels": {
      "E3": 0,
      "E4": 1,
      ...
  }
  ```
- **`location_targets`**: Maps major cities to "Cost Zones" (integers). Lower numbers typically represent higher cost of living.
  ```json
  "location_targets": {
      "New York, NY": 1,
      "San Francisco, CA": 1,
      "Austin, TX": 3,
      ...
  }
  ```

#### 2. Location Settings (`location_settings`)
Controls the proximity matching logic.

- **`max_distance_km`**: The maximum distance (in km) for a city to be considered part of a target city's zone. If a city is further than this from any target, it falls into a default "Unknown" zone.

#### 3. Model Settings (`model`)
Configures the XGBoost model and feature engineering.

- **`targets`**: List of salary components to predict (e.g., "BaseSalary", "Stock").
- **`quantiles`**: List of quantiles to predict (e.g., 0.25, 0.50, 0.75).
- **`sample_weight_k`**: Decay factor for sample weighting based on recency. Higher `k` gives more weight to recent data.
- **`features`**: List of features to use in the model.
  - **`name`**: Feature name (must match column in processed data).
  - **`monotone_constraint`**: Enforces monotonic relationships.
    - `1`: Increasing constraint (higher feature value -> higher prediction).
    - `0`: No constraint.
    - `-1`: Decreasing constraint.

### Configuration Template

Below is a complete example of the configuration structure. valid JSON format:

```json
{
    "mappings": {
        "levels": {"E3": 0, "E4": 1},
        "location_targets": {"New York, NY": 1}
    },
    "location_settings": {"max_distance_km": 50},
    "model": {
        "targets": ["BaseSalary"],
        "quantiles": [0.5],
        "sample_weight_k": 1.0,
        "features": [
            {"name": "Level_Enc", "monotone_constraint": 1}
        ]
    }
}
```

## Proximity Matching

The system uses `geopy` to automatically map input locations to the nearest target city defined in the config.
- **Dynamic Matching**: "Newark" maps to "New York" if it is within the configured `max_distance_km` (default 50km).
- **Caching**: Geocoding results are cached locally to speed up subsequent runs and reduce API usage.
- **O(1) Lookup**: Once a city is processed, its zone is cached in memory for instant lookup.
