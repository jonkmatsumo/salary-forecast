"""
Agents module for LangGraph-based agentic workflow.

This module provides a multi-step AI-powered configuration generation workflow:

1. Column Classification Agent: Identifies targets, features, and columns to ignore
2. Feature Encoding Agent: Determines encoding strategies for categorical features
3. Model Configurator Agent: Proposes hyperparameters and constraints

The workflow supports human-in-the-loop confirmation at each phase.

Usage:
    from src.agents.workflow import ConfigWorkflow
    from src.llm.client import get_langchain_llm
    
    llm = get_langchain_llm("openai")
    workflow = ConfigWorkflow(llm)
    
    # Start workflow
    state = workflow.start(df_json, columns, dtypes, dataset_size)
    
    # Confirm phases
    state = workflow.confirm_classification()
    state = workflow.confirm_encoding()
    
    # Get final config
    config = workflow.get_final_config()
"""

from src.agents.tools import (
    compute_correlation_matrix,
    get_column_statistics,
    get_unique_value_counts,
    detect_ordinal_patterns,
    detect_column_dtype,
    get_all_tools,
)

from src.agents.column_classifier import (
    run_column_classifier_sync,
    get_column_classifier_tools,
)

from src.agents.feature_encoder import (
    run_feature_encoder_sync,
    get_feature_encoder_tools,
)

from src.agents.model_configurator import (
    run_model_configurator_sync,
    get_default_hyperparameters,
)

from src.agents.workflow import (
    ConfigWorkflow,
    WorkflowState,
    create_workflow_graph,
    compile_workflow,
)

__all__ = [
    "compute_correlation_matrix",
    "get_column_statistics",
    "get_unique_value_counts",
    "detect_ordinal_patterns",
    "detect_column_dtype",
    "get_all_tools",
    "run_column_classifier_sync",
    "get_column_classifier_tools",
    "run_feature_encoder_sync",
    "get_feature_encoder_tools",
    "run_model_configurator_sync",
    "get_default_hyperparameters",
    "ConfigWorkflow",
    "WorkflowState",
    "create_workflow_graph",
    "compile_workflow",
]
