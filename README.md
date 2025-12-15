# AutoQuantile

A framework for **Multi-Target Quantile Regression** using **XGBoost**. Automates the lifecycle of probabilistic modeling—from feature engineering to hyperparameter tuning and model versioning.

Key features include:
- **Automated Versioning**: Automatically tracks and versions trained models using **MLflow**.
- **Auto-Tuning**: Integrated Hyperparameter Optimization using **Optuna** to automatically find the best model parameters.
- **LLM-Assisted Feature Engineering**: Multi-step agentic workflow powered by **LangGraph** that generates model configurations using Generative AI (OpenAI GPT-4 or Google Gemini) via the following specialized AI agents:
   1. **Column Classification**: Identifies targets, features, and columns to ignore
   2. **Feature Encoding**: Determines optimal encodings (ordinal, one-hot, proximity, label)
   3. **Model Configuration**: Proposes monotonic constraints, quantiles, and hyperparameters
- **Outlier Detection**: IQR-based outlier filtering to improve model generalization.
- **Proximity Matching**: Geo-spatial grouping of cities into cost zones using distance calculations.

## Installation

Requirements: Python 3.12

1. Create a virtual environment (recommended):
    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2. Upgrade pip (required for pyproject.toml based installs):
    ```bash
    pip install --upgrade pip
    ```

3. Install the package:
    ```bash
    # Development (recommended)
    pip install -e ".[dev]"

    # Or production only
    pip install -r requirements.txt
    ```

4. Set up pre-commit hooks (recommended for development):
    ```bash
    pre-commit install
    ```
    This will automatically run code quality checks (ruff, black, isort, mypy) before each commit.

5. Set up environment variables (for LLM features):
    ```bash
    export OPENAI_API_KEY=your_openai_key_here
    export GEMINI_API_KEY=your_gemini_key_here  # Optional
    ```

## Performance Monitoring

The project includes comprehensive performance monitoring and profiling capabilities:

- **LLM API Tracking**: Automatically tracks token usage, latency, and estimated costs for all LLM API calls
- **Training Pipeline Profiling**: Tracks timing for preprocessing, training per quantile, and hyperparameter tuning
- **Preprocessing Metrics**: Monitors time spent on data cleaning, outlier removal, and feature encoding
- **MLflow Integration**: All performance metrics are automatically logged to MLflow runs

### Accessing Performance Metrics

**Via MLflow UI:**
- Start MLflow UI: `mlflow ui`
- Navigate to runs and view metrics like `llm_total_tokens`, `llm_total_cost`, `training_total_time`, etc.

**Via Code:**
```python
from src.utils.performance import (
    generate_performance_summary,
    get_llm_metrics_summary,
    print_performance_report,
)

# Get comprehensive summary
summary = generate_performance_summary()

# Get LLM metrics only
llm_metrics = get_llm_metrics_summary()

# Print formatted report
print_performance_report()
```

**Export Metrics:**
```python
from src.utils.performance import export_metrics_to_json, export_metrics_to_csv

export_metrics_to_json("performance_report.json")
export_metrics_to_csv("performance_metrics.csv")
```

### UI Performance Tracking

UI performance tracking is optional and disabled by default. To enable:
```bash
export ENABLE_UI_PERFORMANCE_TRACKING=true
```

## Development Commands

This project includes a `Makefile` with common development tasks. Run `make help` to see all available commands.

### Quick Start
```bash
make setup          # Install dev dependencies and set up pre-commit hooks
make test           # Run tests
make check          # Run all checks (format, lint, type-check)
```

### Common Tasks

**Testing:**
```bash
make test           # Run tests
make test-cov       # Run tests with coverage report
```

**Code Quality:**
```bash
make format         # Format code with black and isort
make format-check   # Check formatting without changes
make lint           # Run linting checks
make lint-fix       # Run linting and auto-fix issues
make type-check     # Run type checking
make check          # Run all checks (format, lint, type-check)
make pre-commit     # Run all pre-commit hooks
```

**Security:**
```bash
make security       # Run security scanning (bandit and pip-audit)
```

**Cleanup:**
```bash
make clean          # Remove generated files and caches
```

**Running Applications:**
```bash
make run-api        # Run the FastAPI server
make run-streamlit  # Run the Streamlit application
```

### Manual Commands

If you prefer not to use Make, you can run commands directly:

```bash
# Testing
python3 -m pytest tests/

# Type checking
mypy src/

# Code formatting
black src tests
isort src tests

# Linting
ruff check src tests

# Pre-commit checks
pre-commit run --all-files
```

## Usage

### Web Application (Streamlit)
The easiest way to use the system is via the web interface:

```bash
streamlit run src/app/app.py
```

This launches a dashboard with training and inference pages:
- **Training**: Upload CSV files, use AI-powered configuration wizard, train models with hyperparameter tuning
- **Inference**: Select models, make predictions, view visualizations and feature importance

### REST API

Comprehensive REST API built with **FastAPI** for programmatic access. Features: model management, inference (single/batch), training jobs, configuration workflow, and analytics.

**Start server:**
```bash
uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

**Documentation**: `http://localhost:8000/docs` (Swagger UI) or `/redoc` (ReDoc)

**Authentication** (optional):
```bash
export API_KEY=your_api_key_here
```

**Example:**
```python
import requests
response = requests.get(
    "http://localhost:8000/api/v1/models",
    headers={"X-API-Key": "your_api_key"}
)
```

All endpoints prefixed with `/api/v1`.

### MCP Server (Model Context Protocol)

Native **MCP** server implementation for agent-native interactions via JSON-RPC 2.0. Includes 11 tools for model operations, inference, training, configuration workflow, and analytics.

**Endpoint**: `POST /mcp/rpc` (mounted as FastAPI sub-application)

**Protocol**: JSON-RPC 2.0

**Example request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "predict_salary",
    "arguments": {"run_id": "abc123", "features": {"Level": "L5"}}
  },
  "id": 1
}
```

**Tool discovery**: Use `{"method": "tools/list", "id": 1}` to list all available tools. Each tool includes semantic descriptions, JSON Schema definitions, and examples for LLM integration.
