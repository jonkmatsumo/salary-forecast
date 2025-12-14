# AutoQuantile

A comprehensive framework for **Multi-Target Quantile Regression** using **XGBoost**. It automates the complex lifecycle of probabilistic modeling—from feature engineering and monotonic constraint enforcement to hyperparameter tuning and model versioning.

Key features include:
- **Automated Versioning**: Automatically tracks and versions trained models using **MLflow**.
- **Auto-Tuning**: Integrated Hyperparameter Optimization using **Optuna** to automatically find the best model parameters.
- **LLM-Assisted Feature Engineering**: Uses Generative AI (OpenAI GPT-4 or Google Gemini) to intelligently infer feature and target variables, along with encodings and monotonic constraints through a multi-step agentic workflow.
- **Outlier Detection**: IQR-based outlier filtering to improve model generalization.
- **Proximity Matching**: Geo-spatial grouping of cities into cost zones using distance calculations.
- **Prompt Injection Detection**: Security validation to prevent malicious inputs in the AI workflow.
- **Type Safety**: Comprehensive type annotations with mypy static type checking.

## Installation

1. Create a virtual environment (recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2. Upgrade pip (required for pyproject.toml based installs):
    ```bash
    pip install --upgrade pip
    ```

3. Install the package:

   **Option A: Using pyproject.toml (recommended for development):**
    ```bash
    pip install -e .
    # Or for development (includes test dependencies and type checking):
    pip install -e ".[dev]"
    ```

   **Option B: Using requirements files (recommended for production/reproducibility):**
    ```bash
    # Production dependencies only
    pip install -r requirements.txt
    
    # Or with dev dependencies
    pip install -r requirements-dev.txt
    ```

   **Note:** `requirements.txt` and `requirements-dev.txt` contain pinned versions for reproducibility. `pyproject.toml` uses flexible version constraints (`>=`) for development flexibility.

4. Set up environment variables (for LLM features):
    Create a `.env` file in the project root:
    ```bash
    OPENAI_API_KEY=your_openai_key_here
    GEMINI_API_KEY=your_gemini_key_here  # Optional
    ```

   Or export them in your shell:
    ```bash
    export OPENAI_API_KEY=your_openai_key_here
    export GEMINI_API_KEY=your_gemini_key_here  # Optional
    ```

## Usage

### Web Application (Streamlit)
The easiest way to use the system is via the web interface:

```bash
streamlit run src/app/app.py
```

This launches a dashboard with two main sections:

#### Training Page
- **Upload Data**: Upload CSV files for training
- **Data Analysis**: View dataset statistics and visualizations
- **AI-Powered Configuration Wizard**: Use the multi-step agentic workflow to generate configurations
- **Manual Configuration**: Edit configurations directly
- **Train Models**: Train models with hyperparameter tuning and outlier removal options
- **Model Management**: View training progress and access trained models via MLflow

#### Inference Page
- **Model Selection**: Browse and select trained models from MLflow
- **Interactive Prediction**: Enter candidate details and get quantile predictions
- **Visualizations**: View salary distributions across quantiles
- **Model Analysis**: Explore feature importance and model metrics

## Testing

To run the unit tests:

```bash
python3 -m pytest tests/
```

## Type Checking

AutoQuantile uses **mypy** for static type checking to catch type errors before runtime and improve code quality.

### Setup

Type checking is included in the dev dependencies:

```bash
pip install -e ".[dev]"
```

### Running Type Checks

Check all source files:

```bash
mypy src/
```

Check a specific file or directory:

```bash
mypy src/agents/workflow.py
mypy src/services/
```

### Configuration

Type checking configuration is defined in `mypy.ini`. The configuration:
- Warns on `Any` return types
- Checks untyped definitions
- Ignores missing imports for third-party libraries
- Excludes test files from strict checking

Type checking is also integrated into the CI/CD pipeline (see `.github/workflows/ci.yml`) and runs automatically on pushes and pull requests.

## AI-Powered Configuration Workflow

AutoQuantile features an advanced **multi-step agentic workflow** powered by **LangGraph** that intelligently generates model configurations through a collaborative process between specialized AI agents and human oversight.

### Workflow Overview

The configuration generation process follows a structured 3-phase workflow, with each phase handled by a specialized AI agent:

#### Phase 1: Column Classification
The **Column Classification Agent** analyzes your dataset to identify:
- **Targets**: Columns to predict (e.g., salary components)
- **Features**: Columns to use as input features
- **Ignored**: Columns to exclude (e.g., IDs, metadata)

The agent uses data analysis tools to:
- Compute correlation matrices between columns
- Analyze column statistics (dtypes, null counts, unique values)
- Detect semantic types (numeric, categorical, datetime, boolean)
- Provide reasoning for each classification decision

**Human Review**: You can review and modify the agent's classifications before proceeding.

#### Phase 2: Feature Encoding
The **Feature Encoding Agent** determines optimal encoding strategies for categorical features:
- **Ordinal Encoding**: For features with inherent ordering (e.g., job levels: E3 < E4 < E5)
- **One-Hot Encoding**: For nominal categories with few unique values
- **Proximity Encoding**: For geographic features (cities grouped by distance)
- **Label Encoding**: For high-cardinality categorical features

The agent uses tools to:
- Detect ordinal patterns in categorical data
- Analyze unique value counts and distributions
- Examine correlation with target variables
- Generate encoding mappings where applicable

**Human Review**: You can adjust encoding types and mappings before finalizing.

#### Phase 3: Model Configuration
The **Model Configuration Agent** proposes:
- **Monotonic Constraints**: Enforces relationships between features and predictions
  - `1`: Increasing (higher feature → higher prediction)
  - `0`: No constraint
  - `-1`: Decreasing (higher feature → lower prediction)
- **Quantiles**: Optimal quantile levels for prediction (e.g., [0.1, 0.25, 0.5, 0.75, 0.9])
- **Hyperparameters**: XGBoost training parameters (max_depth, learning rate, etc.)

The agent considers:
- Feature correlations with targets
- Data characteristics (sample size, feature distributions)
- Best practices for quantile regression

**Human Review**: You can fine-tune hyperparameters, quantiles, and constraints.

### Using the Workflow

#### Via Web Interface
1. Launch the Streamlit app: `streamlit run src/app/app.py`
2. Upload your CSV dataset
3. Click **"Start AI-Powered Configuration Wizard"**
4. Review and confirm each phase:
   - Modify classifications if needed
   - Adjust encoding strategies
   - Refine model parameters
5. The workflow generates a complete configuration ready for training

#### Supported LLM Providers
- **OpenAI** (GPT-4, GPT-3.5): Requires `OPENAI_API_KEY` environment variable
- **Google Gemini**: Requires `GEMINI_API_KEY` environment variable

The system automatically detects available providers based on installed packages and API keys.
