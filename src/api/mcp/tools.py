"""MCP tool definitions with semantic enrichment."""

from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MCPTool:
    """Represents an MCP tool definition with metadata."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        examples: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize MCP tool.

        Args:
            name (str): Tool name.
            description (str): Tool description.
            input_schema (Dict[str, Any]): JSON schema for inputs.
            examples (Optional[List[Dict[str, Any]]]): Example inputs/outputs.
        """
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.examples = examples or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary.

        Returns:
            Dict[str, Any]: Tool definition.
        """
        result: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
        if self.examples:
            result["examples"] = self.examples
        return result


def get_mcp_tools() -> List[MCPTool]:
    """Get list of all MCP tools.

    Returns:
        List[MCPTool]: List of tool definitions.
    """
    return [
        _tool_list_models(),
        _tool_get_model_details(),
        _tool_get_model_schema(),
        _tool_predict_salary(),
        _tool_start_training(),
        _tool_get_training_status(),
        _tool_start_configuration_workflow(),
        _tool_confirm_classification(),
        _tool_confirm_encoding(),
        _tool_finalize_configuration(),
        _tool_get_feature_importance(),
    ]


def _tool_list_models() -> MCPTool:
    """Create list_models tool.

    Returns:
        MCPTool: Tool definition.
    """
    return MCPTool(
        name="list_models",
        description=(
            "List all available trained models. Returns metadata including run ID, "
            "training date, model type, CV scores, and dataset names. Use this to "
            "discover which models are available before making predictions or analyzing models."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of models to return",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 100,
                },
                "offset": {
                    "type": "integer",
                    "description": "Number of models to skip",
                    "default": 0,
                    "minimum": 0,
                },
                "experiment_name": {
                    "type": "string",
                    "description": "Filter models by experiment name (optional)",
                },
            },
        },
        examples=[
            {
                "input": {"limit": 10},
                "output": {
                    "models": [
                        {
                            "run_id": "abc123",
                            "start_time": "2024-01-15T10:30:00",
                            "model_type": "XGBoost",
                            "cv_mean_score": 0.85,
                            "dataset_name": "salary_data_2024",
                        }
                    ]
                },
            }
        ],
    )


def _tool_get_model_details() -> MCPTool:
    """Create get_model_details tool.

    Returns:
        MCPTool: Tool definition.
    """
    return MCPTool(
        name="get_model_details",
        description=(
            "Get detailed information about a specific model including metadata, "
            "schema (features, targets, quantiles), and configuration. Use this to "
            "understand what features a model expects before making predictions."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "MLflow run ID of the model",
                }
            },
            "required": ["run_id"],
        },
        examples=[
            {
                "input": {"run_id": "abc123"},
                "output": {
                    "run_id": "abc123",
                    "metadata": {
                        "model_type": "XGBoost",
                        "cv_mean_score": 0.85,
                    },
                    "targets": ["BaseSalary", "TotalComp"],
                    "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
                },
            }
        ],
    )


def _tool_get_model_schema() -> MCPTool:
    """Create get_model_schema tool.

    Returns:
        MCPTool: Tool definition.
    """
    return MCPTool(
        name="get_model_schema",
        description=(
            "Get the schema of a model including ranked features, proximity features, "
            "and numerical features. Use this to understand what input features are required "
            "and what values are valid for categorical features."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "MLflow run ID of the model",
                }
            },
            "required": ["run_id"],
        },
    )


def _tool_predict_salary() -> MCPTool:
    """Create predict_salary tool.

    Returns:
        MCPTool: Tool definition.
    """
    return MCPTool(
        name="predict_salary",
        description=(
            "Predict salary quantiles (e.g., 10th, 50th, 90th percentile) for given features. "
            "Use this when you need to estimate compensation ranges. The model predicts multiple "
            "targets (e.g., BaseSalary, TotalComp) at various quantiles. Returns predictions "
            "for all configured targets and quantiles."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "MLflow run ID of the model to use",
                },
                "features": {
                    "type": "object",
                    "description": "Feature name to value mapping",
                    "additionalProperties": True,
                },
            },
            "required": ["run_id", "features"],
        },
        examples=[
            {
                "input": {
                    "run_id": "abc123",
                    "features": {
                        "Level": "L5",
                        "Location": "San Francisco",
                        "YearsOfExperience": 5,
                    },
                },
                "output": {
                    "predictions": {
                        "BaseSalary": {"p10": 150000, "p50": 180000, "p90": 220000},
                        "TotalComp": {"p10": 180000, "p50": 220000, "p90": 280000},
                    }
                },
            }
        ],
    )


def _tool_start_training() -> MCPTool:
    """Create start_training tool.

    Returns:
        MCPTool: Tool definition.
    """
    return MCPTool(
        name="start_training",
        description=(
            "Start an asynchronous training job for a salary forecasting model. "
            "Requires a dataset_id (from upload_data) and a configuration object. "
            "Returns a job_id that can be used to check training status. Training runs "
            "asynchronously and can take several minutes to hours depending on dataset size."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "dataset_id": {
                    "type": "string",
                    "description": "Dataset ID from upload_data operation",
                },
                "config": {
                    "type": "object",
                    "description": "Model configuration dictionary",
                },
                "remove_outliers": {
                    "type": "boolean",
                    "description": "Whether to remove outliers using IQR method",
                    "default": True,
                },
                "do_tune": {
                    "type": "boolean",
                    "description": "Whether to run hyperparameter tuning",
                    "default": False,
                },
                "n_trials": {
                    "type": "integer",
                    "description": "Number of tuning trials (if do_tune is true)",
                },
                "additional_tag": {
                    "type": "string",
                    "description": "Optional tag to identify this training run",
                },
            },
            "required": ["dataset_id", "config"],
        },
    )


def _tool_get_training_status() -> MCPTool:
    """Create get_training_status tool.

    Returns:
        MCPTool: Tool definition.
    """
    return MCPTool(
        name="get_training_status",
        description=(
            "Get the status of a training job. Returns current status (QUEUED, RUNNING, "
            "COMPLETED, FAILED), logs, and if completed, the run_id where the model was saved. "
            "Use this to poll for training completion."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "Training job ID from start_training",
                }
            },
            "required": ["job_id"],
        },
    )


def _tool_start_configuration_workflow() -> MCPTool:
    """Create start_configuration_workflow tool.

    Returns:
        MCPTool: Tool definition.
    """
    return MCPTool(
        name="start_configuration_workflow",
        description=(
            "Start an AI-powered configuration workflow that analyzes your dataset and "
            "generates optimal model configuration through a multi-step process. The workflow "
            "includes column classification, feature encoding, and model configuration. Returns "
            "a workflow_id for tracking progress through confirmation steps."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "JSON string of DataFrame (records orient)",
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of column names",
                },
                "dtypes": {
                    "type": "object",
                    "description": "Dictionary mapping column names to data types",
                },
                "dataset_size": {
                    "type": "integer",
                    "description": "Total number of rows in the dataset",
                },
                "provider": {
                    "type": "string",
                    "description": "LLM provider (e.g., 'openai', 'gemini')",
                    "default": "openai",
                },
                "preset": {
                    "type": "string",
                    "description": "Optional preset prompt name (e.g., 'salary')",
                },
            },
            "required": ["data", "columns", "dtypes", "dataset_size"],
        },
    )


def _tool_confirm_classification() -> MCPTool:
    """Create confirm_classification tool.

    Returns:
        MCPTool: Tool definition.
    """
    return MCPTool(
        name="confirm_classification",
        description=(
            "Confirm or modify the column classification from the configuration workflow. "
            "Use this after start_configuration_workflow returns classification results. "
            "You can modify which columns are targets, features, or ignored before proceeding."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "Workflow ID from start_configuration_workflow",
                },
                "modifications": {
                    "type": "object",
                    "properties": {
                        "targets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of target column names",
                        },
                        "features": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of feature column names",
                        },
                        "ignore": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of columns to ignore",
                        },
                    },
                },
            },
            "required": ["workflow_id", "modifications"],
        },
    )


def _tool_confirm_encoding() -> MCPTool:
    """Create confirm_encoding tool.

    Returns:
        MCPTool: Tool definition.
    """
    return MCPTool(
        name="confirm_encoding",
        description=(
            "Confirm or modify feature encodings from the configuration workflow. "
            "Use this after confirm_classification. You can modify encoding types and "
            "optional encodings (e.g., cost of living for locations)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "Workflow ID",
                },
                "modifications": {
                    "type": "object",
                    "properties": {
                        "encodings": {
                            "type": "object",
                            "description": "Dictionary of column name to encoding configuration",
                        },
                        "optional_encodings": {
                            "type": "object",
                            "description": "Dictionary of optional encoding configurations",
                        },
                    },
                },
            },
            "required": ["workflow_id", "modifications"],
        },
    )


def _tool_finalize_configuration() -> MCPTool:
    """Create finalize_configuration tool.

    Returns:
        MCPTool: Tool definition.
    """
    return MCPTool(
        name="finalize_configuration",
        description=(
            "Finalize the configuration workflow and generate the complete configuration. "
            "Use this after confirm_encoding. Returns the final configuration that can be "
            "used for model training."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "Workflow ID",
                },
                "config_updates": {
                    "type": "object",
                    "description": "Configuration updates including features, quantiles, hyperparameters",
                },
            },
            "required": ["workflow_id", "config_updates"],
        },
    )


def _tool_get_feature_importance() -> MCPTool:
    """Create get_feature_importance tool.

    Returns:
        MCPTool: Tool definition.
    """
    return MCPTool(
        name="get_feature_importance",
        description=(
            "Get feature importance scores for a specific model, target, and quantile. "
            "Returns features ranked by their importance (Gain score) in the model. "
            "Use this to understand which features are most influential for predictions."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "MLflow run ID of the model",
                },
                "target": {
                    "type": "string",
                    "description": "Target column name (e.g., 'BaseSalary')",
                },
                "quantile": {
                    "type": "number",
                    "description": "Quantile value (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": ["run_id", "target", "quantile"],
        },
        examples=[
            {
                "input": {"run_id": "abc123", "target": "BaseSalary", "quantile": 0.5},
                "output": {
                    "features": [
                        {"name": "Level", "gain": 0.45},
                        {"name": "Location", "gain": 0.30},
                        {"name": "YearsOfExperience", "gain": 0.25},
                    ]
                },
            }
        ],
    )
