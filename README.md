# AutoQuantile

A comprehensive framework for **Multi-Target Quantile Regression** using **XGBoost**. It automates the complex lifecycle of probabilistic modeling—from feature engineering and monotonic constraint enforcement to hyperparameter tuning and model versioning.

Key features include:
- **Automated Version Control**: Automatically tracks and versions trained models using **MLFlow**.
- **Auto-Tuning**: Integrated Hyperparameter Optimization using **Optuna** to automatically find the best model parameters.
- **LLM-Assisted Configuration**: Uses Generative AI to intelligently infer column roles and level hierarchies from data.
- **Proximity Matching**: Geo-spatial grouping of cities into cost zones.
- **Robustness**: **Outlier Detection** (IQR) to filter extreme data points and improve model generalization.

## Installation

1.  Create a virtual environment (optional but recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    ```bash
    pip install -e .
    # Or for development (includes test dependencies):
    pip install -e ".[dev]"
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
salary-forecast-train
# Or: python3 -m src.cli.train_cli
```

You can specify the input CSV, config file, and output model path.

### Inference (CLI)
You can run the CLI in two modes:

**1. Interactive Mode**:
```bash
salary-forecast-infer
```
Follow the prompts to select a model and enter candidate details.

**2. Non-Interactive (Automation) Mode**:
Pass all required arguments via flags to skip prompts. Useful for scripts.

```bash
salary-forecast-infer --model salary_model.pkl --level E5 --location "New York" --yoe 5 --yac 2
```

**JSON Output**:
Add the `--json` flag to output results as a machine-readable JSON object (suppresses charts and tables).

```bash
salary-forecast-infer ... --json
```

## Testing

To run the unit tests:

```bash
python3 -m pytest tests/
```

## Configuration

The model and data processing are configurable through a structured dictionary. This configuration can be edited **inline** directly in the web app, inferred automatically using **LLM (OpenAI/Gemini)** suggestions, or provided as a `config.json` file when using the CLI.

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
- **`hyperparameters`** (Optional): XGBoost parameters for training and cross-validation. If omitted, default values are used.
  - **`training`**: Parameters passed to `xgb.train` (e.g., `max_depth`, `eta`).
  - **`cv`**: Parameters passed to `xgb.cv` (e.g., `num_boost_round`, `nfold`).

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
        ],
        "hyperparameters": {
            "training": {
                "objective": "reg:quantileerror",
                "max_depth": 6
            },
            "cv": {
                "num_boost_round": 100
            }
        }
    }
}
```
