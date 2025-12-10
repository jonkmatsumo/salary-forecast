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
from src.agents.prompt_injection_detector import detect_prompt_injection
from src.utils.logger import get_logger
from src.utils.observability import log_workflow_state_transition

logger = get_logger(__name__)


class PromptInjectionError(Exception):
    """Exception raised when prompt injection is detected in user input."""
    
    def __init__(self, message: str, confidence: float, reasoning: str, suspicious_content: str):
        """
        Initialize the error.
        
        Args:
            message: Error message.
            confidence: Detection confidence level.
            reasoning: Explanation of the detection.
            suspicious_content: Content that triggered the detection.
        """
        super().__init__(message)
        self.confidence = confidence
        self.reasoning = reasoning
        self.suspicious_content = suspicious_content


class WorkflowState(TypedDict, total=False):
    """State schema for the configuration workflow."""
    # Input data
    df_json: str
    columns: List[str]
    dtypes: Dict[str, str]
    dataset_size: int
    preset: Optional[str]  # Optional preset prompt name (e.g., "salary")
    
    # Phase 1: Column Classification
    column_classification: Dict[str, Any]
    classification_confirmed: bool
    location_columns: List[str]  # Extracted location columns from column_types
    column_types: Dict[str, str]  # Semantic types (e.g., {"Location": "location", "Date": "datetime"})
    optional_encodings: Dict[str, Dict[str, Any]]  # Optional encoding selections per column
    
    # Phase 2: Feature Encoding
    feature_encodings: Dict[str, Any]
    encodings_confirmed: bool
    
    # Phase 3: Model Configuration
    model_config: Dict[str, Any]
    location_settings: Dict[str, Any]  # max_distance_km setting
    
    # Computed data for later phases
    correlation_data: Optional[str]
    
    # Final output
    final_config: Dict[str, Any]
    
    # Workflow metadata
    current_phase: str
    current_node: Optional[str]  # Current LangGraph node being executed
    error: Optional[str]


def validate_input_node(state: WorkflowState, llm: BaseChatModel) -> Dict[str, Any]:
    """
    Node that validates user input for prompt injection attacks.
    
    Args:
        state: Current workflow state.
        llm: Language model for detection.
        
    Returns:
        Empty dict if validation passes (no state change).
        
    Raises:
        PromptInjectionError: If prompt injection is detected.
    """
    logger.info("Validating input for prompt injection...")
    log_workflow_state_transition("validate_input_before", state)
    
    df_json = state.get("df_json", "")
    columns = state.get("columns", [])
    
    if not df_json:
        logger.warning("Empty df_json in validation node")
        return {"current_node": None}
    
    try:
        detection_result = detect_prompt_injection(llm, df_json, columns)
        
        if detection_result.get("is_suspicious", False):
            confidence = detection_result.get("confidence", 0.0)
            reasoning = detection_result.get("reasoning", "No reasoning provided")
            suspicious_content = detection_result.get("suspicious_content", "")
            
            logger.warning(
                f"Prompt injection detected: confidence={confidence}, "
                f"reasoning={reasoning}, suspicious_content={suspicious_content[:200]}"
            )
            
            error_message = (
                f"Potential prompt injection detected in uploaded data. "
                f"Confidence: {confidence:.2f}. {reasoning}"
            )
            
            raise PromptInjectionError(
                error_message,
                confidence,
                reasoning,
                suspicious_content
            )
        
        logger.info("Input validation passed")
        log_workflow_state_transition("validate_input_after", state)
        return {"current_node": None}
        
    except PromptInjectionError:
        raise
    except Exception as e:
        logger.error(f"Error during input validation: {e}", exc_info=True)
        return {}


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
    log_workflow_state_transition("classify_columns_before", state)
    
    logger.debug(f"State keys: {list(state.keys())}")
    logger.debug(f"df_json type: {type(state.get('df_json'))}, length: {len(state.get('df_json', '')) if state.get('df_json') else 0}")
    logger.debug(f"Columns: {state.get('columns', [])}")
    logger.debug(f"Dtypes: {state.get('dtypes', {})}")
    
    try:
        logger.info("Calling run_column_classifier_sync...")
        result = run_column_classifier_sync(
            llm=llm,
            df_json=state["df_json"],
            columns=state["columns"],
            dtypes=state["dtypes"],
            preset=state.get("preset")
        )
        logger.info("run_column_classifier_sync completed successfully")
        logger.debug(f"Classification result keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
        
        # Extract location columns from column_types
        column_types = result.get("column_types", {})
        location_columns = [col for col, col_type in column_types.items() if col_type == "location"]
        logger.debug(f"Detected location columns: {location_columns}")
        
        # Compute correlations for later use
        correlation_data = None
        try:
            logger.debug("Computing correlation matrix...")
            correlation_data = compute_correlation_matrix.invoke({
                "df_json": state["df_json"],
                "columns": None  # All numeric columns
            })
            logger.debug(f"Correlation data computed (length: {len(correlation_data) if correlation_data else 0})")
        except Exception as e:
            logger.warning(f"Could not compute correlations: {e}", exc_info=True)
        
        new_state = {
            "column_classification": result,
            "classification_confirmed": False,
            "location_columns": location_columns,
            "column_types": column_types,
            "optional_encodings": state.get("optional_encodings", {}),
            "correlation_data": correlation_data,
            "current_phase": "classification",
            "current_node": "classifying_columns",
            "error": None
        }
        updated_state = {**state, **new_state}
        log_workflow_state_transition("classify_columns_after", updated_state)
        return new_state
        
    except Exception as e:
        logger.error(f"Column classification failed: {e}", exc_info=True)
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            "column_classification": {},
            "classification_confirmed": False,
            "current_phase": "classification",
            "current_node": None,
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
    log_workflow_state_transition("encode_features_before", state)
    
    try:
        classification = state.get("column_classification", {})
        features = classification.get("features", [])
        location_columns = state.get("location_columns", [])
        
        result = run_feature_encoder_sync(
            llm=llm,
            df_json=state["df_json"],
            features=features,
            dtypes=state["dtypes"],
            preset=state.get("preset")
        )
        
        # Automatically set proximity encoding for location columns
        if location_columns and "encodings" in result:
            for loc_col in location_columns:
                if loc_col in result["encodings"]:
                    # Override with proximity encoding
                    result["encodings"][loc_col] = {
                        "type": "proximity",
                        "reasoning": "Location column detected in classification phase"
                    }
                else:
                    # Add if not already present
                    result["encodings"][loc_col] = {
                        "type": "proximity",
                        "reasoning": "Location column detected in classification phase"
                    }
            logger.debug(f"Set proximity encoding for location columns: {location_columns}")
        
        new_state = {
            "feature_encodings": result,
            "encodings_confirmed": False,
            "optional_encodings": state.get("optional_encodings", {}),
            "current_phase": "encoding",
            "current_node": "evaluating_features",
            "error": None
        }
        updated_state = {**state, **new_state}
        log_workflow_state_transition("encode_features_after", updated_state)
        return new_state
        
    except Exception as e:
        logger.error(f"Feature encoding failed: {e}")
        return {
            "feature_encodings": {},
            "encodings_confirmed": False,
            "current_phase": "encoding",
            "current_node": None,
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
    log_workflow_state_transition("configure_model_before", state)
    
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
            dataset_size=state.get("dataset_size", 0),
            preset=state.get("preset")
        )
        
        new_state = {
            "model_config": result,
            "optional_encodings": state.get("optional_encodings", {}),
            "current_phase": "configuration",
            "current_node": "configuring_model",
            "error": None
        }
        updated_state = {**state, **new_state}
        log_workflow_state_transition("configure_model_after", updated_state)
        return new_state
        
    except Exception as e:
        logger.error(f"Model configuration failed: {e}")
        return {
            "model_config": {},
            "current_phase": "configuration",
            "current_node": None,
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
    location_columns = state.get("location_columns", [])
    location_settings = state.get("location_settings", {"max_distance_km": 50})
    
    # Build mappings from encodings
    mappings = {
        "levels": {},
        "location_targets": {}  # Empty initially, user can populate later
    }
    
    feature_engineering = {
        "ranked_cols": {},
        "proximity_cols": []
    }
    
    # Process encodings
    for col, enc_config in encodings.get("encodings", {}).items():
        enc_type = enc_config.get("type", "")
        
        if enc_type == "ordinal" and "mapping" in enc_config:
            # Create a mapping key for this column
            mapping_key = f"{col.lower()}_mapping"
            mappings[mapping_key] = enc_config["mapping"]
            feature_engineering["ranked_cols"][col] = mapping_key
            
        elif enc_type == "proximity":
            feature_engineering["proximity_cols"].append(col)
    
    # Also add location columns from classification (in case they weren't in encodings)
    for loc_col in location_columns:
        if loc_col not in feature_engineering["proximity_cols"]:
            feature_engineering["proximity_cols"].append(loc_col)
    
    # Get optional encodings from state
    optional_encodings = state.get("optional_encodings", {})
    
    # Build final config structure
    final_config = {
        "mappings": mappings,
        "feature_engineering": feature_engineering,
        "location_settings": {
            "max_distance_km": int(location_settings.get("max_distance_km", 50))
        },
        "model": {
            "targets": classification.get("targets", []),
            "features": model_config.get("features", []),
            "quantiles": model_config.get("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9]),
            "sample_weight_k": 1.0,
            "hyperparameters": model_config.get("hyperparameters", {})
        },
        "optional_encodings": optional_encodings,
        "_metadata": {
            "classification_reasoning": classification.get("reasoning", ""),
            "encoding_summary": encodings.get("summary", ""),
            "configuration_reasoning": model_config.get("reasoning", "")
        }
    }
    
    return {
        "final_config": final_config,
        "current_phase": "complete",
        "current_node": "building_config",
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
    
    The workflow has validation and three main phases with human-in-the-loop checkpoints:
    1. validate_input -> 
    2. classify_columns -> (await confirmation) -> 
    3. encode_features -> (await confirmation) ->
    4. configure_model -> build_final_config -> END
    
    Args:
        llm: Language model to use for agents.
        
    Returns:
        Compiled StateGraph ready for execution.
    """
    # Create the graph
    workflow = StateGraph(WorkflowState)
    
    # Define nodes with llm bound
    workflow.add_node("validate_input", lambda state: validate_input_node(state, llm))
    workflow.add_node("classify_columns", lambda state: classify_columns_node(state, llm))
    workflow.add_node("await_classification", lambda state: state)  # No-op, just a checkpoint
    workflow.add_node("encode_features", lambda state: encode_features_node(state, llm))
    workflow.add_node("await_encoding", lambda state: state)  # No-op, just a checkpoint
    workflow.add_node("configure_model", lambda state: configure_model_node(state, llm))
    workflow.add_node("build_final_config", build_final_config_node)
    
    # Set entry point
    workflow.set_entry_point("validate_input")
    
    # Add edge from validation to classification
    workflow.add_edge("validate_input", "classify_columns")
    
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
    
    def start(self, df_json: str, columns: List[str], dtypes: Dict[str, str], dataset_size: int, preset: Optional[str] = None) -> Dict[str, Any]:
        """
        Start the workflow with initial data.
        
        Runs until the first checkpoint (after column classification).
        
        Args:
            df_json: JSON representation of DataFrame sample.
            columns: List of column names.
            dtypes: Dict mapping column names to dtypes.
            dataset_size: Number of rows in the dataset.
            preset: Optional preset prompt name (e.g., "salary").
            
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
            "preset": preset,
            "optional_encodings": {},
            "classification_confirmed": False,
            "encodings_confirmed": False,
            "current_phase": "starting",
            "current_node": None
        }
        
        config = {"configurable": {"thread_id": self.thread_id}}
        
        # Run until first interrupt
        for event in self.compiled.stream(initial_state, config):
            self.current_state = event
        
        # Get latest state
        state_snapshot = self.compiled.get_state(config)
        self.current_state = dict(state_snapshot.values) if state_snapshot.values else {}
        
        log_workflow_state_transition("ConfigWorkflow.start", self.current_state)
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
            # Extract optional_encodings separately if present (copy to avoid mutating original)
            modifications_copy = modifications.copy()
            optional_encodings = modifications_copy.pop("optional_encodings", None)
            if optional_encodings is not None:
                update_state["optional_encodings"] = optional_encodings
            
            current_classification = self.current_state.get("column_classification", {})
            current_classification.update(modifications_copy)
            update_state["column_classification"] = current_classification
            # Extract location columns from column_types if present
            column_types = current_classification.get("column_types", {})
            if column_types:
                location_columns = [col for col, col_type in column_types.items() if col_type == "location"]
                update_state["location_columns"] = location_columns
                update_state["column_types"] = column_types
            # Backward compatibility: if locations key exists, migrate to column_types
            elif "locations" in modifications_copy:
                location_columns = modifications_copy["locations"]
                update_state["location_columns"] = location_columns
                # Create column_types from locations
                column_types = {col: "location" for col in location_columns}
                update_state["column_types"] = column_types
                current_classification["column_types"] = column_types
                update_state["column_classification"] = current_classification
        
        # Update state and resume
        self.compiled.update_state(config, update_state)
        
        # Run until next interrupt
        for event in self.compiled.stream(None, config):
            self.current_state = event
        
        state_snapshot = self.compiled.get_state(config)
        self.current_state = dict(state_snapshot.values) if state_snapshot.values else {}
        
        log_workflow_state_transition("ConfigWorkflow.confirm_classification", self.current_state)
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
            # Update optional encodings if provided
            if "optional_encodings" in modifications:
                update_state["optional_encodings"] = modifications["optional_encodings"]
        
        # Update state and resume
        self.compiled.update_state(config, update_state)
        
        # Run to completion
        for event in self.compiled.stream(None, config):
            self.current_state = event
        
        state_snapshot = self.compiled.get_state(config)
        self.current_state = dict(state_snapshot.values) if state_snapshot.values else {}
        
        log_workflow_state_transition("ConfigWorkflow.confirm_encoding", self.current_state)
        return self.current_state
    
    def get_current_phase(self) -> str:
        """Get the current workflow phase."""
        return self.current_state.get("current_phase", "unknown") if self.current_state else "not_started"
    
    def get_final_config(self) -> Optional[Dict[str, Any]]:
        """Get the final configuration if workflow is complete."""
        if self.current_state and self.current_state.get("current_phase") == "complete":
            return self.current_state.get("final_config")
        return None

