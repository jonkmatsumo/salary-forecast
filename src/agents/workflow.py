"""
LangGraph Workflow for Configuration Generation.

This module defines the state graph that connects the three agents:
1. Column Classifier -> User Confirmation
2. Feature Encoder -> User Confirmation  
3. Model Configurator -> Final Config

Human-in-the-loop checkpoints allow users to review and modify
agent outputs before proceeding to the next phase.
"""

from typing import Any, Dict, List, Optional, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models import BaseChatModel

from src.agents.column_classifier import run_column_classifier_sync
from src.agents.feature_encoder import run_feature_encoder_sync
from src.agents.model_configurator import run_model_configurator_sync
from src.agents.tools import compute_correlation_matrix
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WorkflowState(TypedDict, total=False):
    """State schema for the configuration workflow."""
    # Input data
    df_json: str
    columns: List[str]
    dtypes: Dict[str, str]
    dataset_size: int
    
    # Phase 1: Column Classification
    column_classification: Dict[str, Any]
    classification_confirmed: bool
    
    # Phase 2: Feature Encoding
    feature_encodings: Dict[str, Any]
    encodings_confirmed: bool
    
    # Phase 3: Model Configuration
    model_config: Dict[str, Any]
    
    # Computed data for later phases
    correlation_data: Optional[str]
    
    # Final output
    final_config: Dict[str, Any]
    
    # Workflow metadata
    current_phase: str
    error: Optional[str]


def classify_columns_node(state: WorkflowState, llm: BaseChatModel) -> Dict[str, Any]:
    """
    Node that runs the column classification agent.
    
    Args:
        state: Current workflow state.
        llm: Language model for the agent.
        
    Returns:
        State updates with classification results.
    """
    logger.info("Running column classification agent...")
    
    try:
        result = run_column_classifier_sync(
            llm=llm,
            df_json=state["df_json"],
            columns=state["columns"],
            dtypes=state["dtypes"]
        )
        
        # Compute correlations for later use
        correlation_data = None
        try:
            correlation_data = compute_correlation_matrix.invoke({
                "df_json": state["df_json"],
                "columns": None  # All numeric columns
            })
        except Exception as e:
            logger.warning(f"Could not compute correlations: {e}")
        
        return {
            "column_classification": result,
            "classification_confirmed": False,
            "correlation_data": correlation_data,
            "current_phase": "classification",
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Column classification failed: {e}")
        return {
            "column_classification": {},
            "classification_confirmed": False,
            "current_phase": "classification",
            "error": str(e)
        }


def encode_features_node(state: WorkflowState, llm: BaseChatModel) -> Dict[str, Any]:
    """
    Node that runs the feature encoding agent.
    
    Args:
        state: Current workflow state.
        llm: Language model for the agent.
        
    Returns:
        State updates with encoding recommendations.
    """
    logger.info("Running feature encoding agent...")
    
    try:
        classification = state.get("column_classification", {})
        features = classification.get("features", [])
        
        result = run_feature_encoder_sync(
            llm=llm,
            df_json=state["df_json"],
            features=features,
            dtypes=state["dtypes"]
        )
        
        return {
            "feature_encodings": result,
            "encodings_confirmed": False,
            "current_phase": "encoding",
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Feature encoding failed: {e}")
        return {
            "feature_encodings": {},
            "encodings_confirmed": False,
            "current_phase": "encoding",
            "error": str(e)
        }


def configure_model_node(state: WorkflowState, llm: BaseChatModel) -> Dict[str, Any]:
    """
    Node that runs the model configuration agent.
    
    Args:
        state: Current workflow state.
        llm: Language model for the agent.
        
    Returns:
        State updates with model configuration.
    """
    logger.info("Running model configuration agent...")
    
    try:
        classification = state.get("column_classification", {})
        targets = classification.get("targets", [])
        encodings = state.get("feature_encodings", {})
        
        result = run_model_configurator_sync(
            llm=llm,
            targets=targets,
            encodings=encodings,
            correlation_data=state.get("correlation_data"),
            column_stats=None,
            dataset_size=state.get("dataset_size", 0)
        )
        
        return {
            "model_config": result,
            "current_phase": "configuration",
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Model configuration failed: {e}")
        return {
            "model_config": {},
            "current_phase": "configuration",
            "error": str(e)
        }


def build_final_config_node(state: WorkflowState) -> Dict[str, Any]:
    """
    Node that assembles the final configuration from all agent outputs.
    
    Args:
        state: Current workflow state.
        
    Returns:
        State updates with final configuration.
    """
    logger.info("Building final configuration...")
    
    classification = state.get("column_classification", {})
    encodings = state.get("feature_encodings", {})
    model_config = state.get("model_config", {})
    
    # Build mappings from encodings
    mappings = {
        "levels": {},
        "location_targets": {}
    }
    
    feature_engineering = {
        "ranked_cols": {},
        "proximity_cols": []
    }
    
    for col, enc_config in encodings.get("encodings", {}).items():
        enc_type = enc_config.get("type", "")
        
        if enc_type == "ordinal" and "mapping" in enc_config:
            # Create a mapping key for this column
            mapping_key = f"{col.lower()}_mapping"
            mappings[mapping_key] = enc_config["mapping"]
            feature_engineering["ranked_cols"][col] = mapping_key
            
        elif enc_type == "proximity":
            feature_engineering["proximity_cols"].append(col)
    
    # Build final config structure
    final_config = {
        "mappings": mappings,
        "feature_engineering": feature_engineering,
        "location_settings": {
            "max_distance_km": 50.0
        },
        "model": {
            "targets": classification.get("targets", []),
            "features": model_config.get("features", []),
            "quantiles": model_config.get("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9]),
            "sample_weight_k": 1.0,
            "hyperparameters": model_config.get("hyperparameters", {})
        },
        "_metadata": {
            "classification_reasoning": classification.get("reasoning", ""),
            "encoding_summary": encodings.get("summary", ""),
            "configuration_reasoning": model_config.get("reasoning", "")
        }
    }
    
    return {
        "final_config": final_config,
        "current_phase": "complete",
        "error": None
    }


def should_continue_after_classification(state: WorkflowState) -> Literal["encode_features", "await_classification"]:
    """Determine if we should proceed after classification or wait for user."""
    if state.get("classification_confirmed", False):
        return "encode_features"
    return "await_classification"


def should_continue_after_encoding(state: WorkflowState) -> Literal["configure_model", "await_encoding"]:
    """Determine if we should proceed after encoding or wait for user."""
    if state.get("encodings_confirmed", False):
        return "configure_model"
    return "await_encoding"


def create_workflow_graph(llm: BaseChatModel) -> StateGraph:
    """
    Create the LangGraph workflow for configuration generation.
    
    The workflow has three main phases with human-in-the-loop checkpoints:
    1. classify_columns -> (await confirmation) -> 
    2. encode_features -> (await confirmation) ->
    3. configure_model -> build_final_config -> END
    
    Args:
        llm: Language model to use for agents.
        
    Returns:
        Compiled StateGraph ready for execution.
    """
    # Create the graph
    workflow = StateGraph(WorkflowState)
    
    # Define nodes with llm bound
    workflow.add_node("classify_columns", lambda state: classify_columns_node(state, llm))
    workflow.add_node("await_classification", lambda state: state)  # No-op, just a checkpoint
    workflow.add_node("encode_features", lambda state: encode_features_node(state, llm))
    workflow.add_node("await_encoding", lambda state: state)  # No-op, just a checkpoint
    workflow.add_node("configure_model", lambda state: configure_model_node(state, llm))
    workflow.add_node("build_final_config", build_final_config_node)
    
    # Set entry point
    workflow.set_entry_point("classify_columns")
    
    # Add edges
    workflow.add_conditional_edges(
        "classify_columns",
        should_continue_after_classification,
        {
            "encode_features": "encode_features",
            "await_classification": "await_classification"
        }
    )
    
    workflow.add_conditional_edges(
        "await_classification",
        should_continue_after_classification,
        {
            "encode_features": "encode_features",
            "await_classification": "await_classification"  # Stay here until confirmed
        }
    )
    
    workflow.add_conditional_edges(
        "encode_features",
        should_continue_after_encoding,
        {
            "configure_model": "configure_model",
            "await_encoding": "await_encoding"
        }
    )
    
    workflow.add_conditional_edges(
        "await_encoding",
        should_continue_after_encoding,
        {
            "configure_model": "configure_model",
            "await_encoding": "await_encoding"  # Stay here until confirmed
        }
    )
    
    workflow.add_edge("configure_model", "build_final_config")
    workflow.add_edge("build_final_config", END)
    
    return workflow


def compile_workflow(llm: BaseChatModel, checkpointer: Optional[MemorySaver] = None):
    """
    Compile the workflow graph with optional checkpointing.
    
    Args:
        llm: Language model to use for agents.
        checkpointer: Optional memory saver for state persistence.
        
    Returns:
        Compiled workflow ready for execution.
    """
    workflow = create_workflow_graph(llm)
    
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["await_classification", "await_encoding"]
    )


class ConfigWorkflow:
    """
    High-level wrapper for the configuration workflow.
    
    Provides methods to run the workflow step-by-step with
    user confirmation between phases.
    """
    
    def __init__(self, llm: BaseChatModel):
        """
        Initialize the workflow.
        
        Args:
            llm: Language model to use for agents.
        """
        self.llm = llm
        self.checkpointer = MemorySaver()
        self.compiled = compile_workflow(llm, self.checkpointer)
        self.thread_id = None
        self.current_state = None
    
    def start(self, df_json: str, columns: List[str], dtypes: Dict[str, str], dataset_size: int) -> Dict[str, Any]:
        """
        Start the workflow with initial data.
        
        Runs until the first checkpoint (after column classification).
        
        Args:
            df_json: JSON representation of DataFrame sample.
            columns: List of column names.
            dtypes: Dict mapping column names to dtypes.
            dataset_size: Number of rows in the dataset.
            
        Returns:
            Current workflow state after classification.
        """
        import uuid
        self.thread_id = str(uuid.uuid4())
        
        initial_state = {
            "df_json": df_json,
            "columns": columns,
            "dtypes": dtypes,
            "dataset_size": dataset_size,
            "classification_confirmed": False,
            "encodings_confirmed": False,
            "current_phase": "starting"
        }
        
        config = {"configurable": {"thread_id": self.thread_id}}
        
        # Run until first interrupt
        for event in self.compiled.stream(initial_state, config):
            self.current_state = event
        
        # Get latest state
        state_snapshot = self.compiled.get_state(config)
        self.current_state = dict(state_snapshot.values) if state_snapshot.values else {}
        
        return self.current_state
    
    def confirm_classification(self, modifications: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Confirm the column classification and proceed to encoding.
        
        Args:
            modifications: Optional modifications to the classification.
            
        Returns:
            Current workflow state after encoding.
        """
        if not self.thread_id:
            raise RuntimeError("Workflow not started")
        
        config = {"configurable": {"thread_id": self.thread_id}}
        
        # Apply modifications if provided
        update_state = {"classification_confirmed": True}
        if modifications:
            current_classification = self.current_state.get("column_classification", {})
            current_classification.update(modifications)
            update_state["column_classification"] = current_classification
        
        # Update state and resume
        self.compiled.update_state(config, update_state)
        
        # Run until next interrupt
        for event in self.compiled.stream(None, config):
            self.current_state = event
        
        state_snapshot = self.compiled.get_state(config)
        self.current_state = dict(state_snapshot.values) if state_snapshot.values else {}
        
        return self.current_state
    
    def confirm_encoding(self, modifications: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Confirm the feature encoding and proceed to model configuration.
        
        Args:
            modifications: Optional modifications to the encodings.
            
        Returns:
            Current workflow state with final config.
        """
        if not self.thread_id:
            raise RuntimeError("Workflow not started")
        
        config = {"configurable": {"thread_id": self.thread_id}}
        
        # Apply modifications if provided
        update_state = {"encodings_confirmed": True}
        if modifications:
            current_encodings = self.current_state.get("feature_encodings", {})
            if "encodings" in current_encodings:
                current_encodings["encodings"].update(modifications.get("encodings", {}))
            update_state["feature_encodings"] = current_encodings
        
        # Update state and resume
        self.compiled.update_state(config, update_state)
        
        # Run to completion
        for event in self.compiled.stream(None, config):
            self.current_state = event
        
        state_snapshot = self.compiled.get_state(config)
        self.current_state = dict(state_snapshot.values) if state_snapshot.values else {}
        
        return self.current_state
    
    def get_current_phase(self) -> str:
        """Get the current workflow phase."""
        return self.current_state.get("current_phase", "unknown") if self.current_state else "not_started"
    
    def get_final_config(self) -> Optional[Dict[str, Any]]:
        """Get the final configuration if workflow is complete."""
        if self.current_state and self.current_state.get("current_phase") == "complete":
            return self.current_state.get("final_config")
        return None

