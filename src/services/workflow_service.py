"""
Workflow Service for orchestrating the agentic configuration generation.

This service wraps the LangGraph workflow and provides a simple interface
for the Streamlit UI to manage the multi-step configuration process.
"""

import json
from typing import Any, Dict, List, Optional
import pandas as pd

from src.llm.client import get_langchain_llm, get_available_providers
from src.agents.workflow import ConfigWorkflow, PromptInjectionError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WorkflowService:
    """
    Service to manage the agentic configuration workflow.
    
    This service provides methods to:
    - Start a new workflow with a dataset
    - Get current workflow state
    - Confirm and modify phase outputs
    - Retrieve the final configuration
    """
    
    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        """
        Initialize the workflow service.
        
        Args:
            provider: LLM provider name ("openai" or "gemini").
            model: Optional model name override.
        """
        self.provider = provider
        self.model = model
        self.workflow: Optional[ConfigWorkflow] = None
        self.current_state: Dict[str, Any] = {}
        
        # Initialize LLM
        try:
            self.llm = get_langchain_llm(provider=provider, model=model)
            logger.info(f"WorkflowService initialized with provider: {provider}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def start_workflow(self, df: pd.DataFrame, sample_size: int = 50) -> Dict[str, Any]:
        """Start a new configuration workflow. Args: df (pd.DataFrame): Input DataFrame. sample_size (int): Rows to sample. Returns: Dict[str, Any]: Classification results."""
        logger.info(f"Starting workflow with {len(df)} rows, sampling {sample_size}")
        logger.debug(f"DataFrame columns: {df.columns.tolist()}")
        logger.debug(f"DataFrame dtypes: {df.dtypes.to_dict()}")
        
        sample_df = df.head(sample_size)
        logger.debug(f"Sampled DataFrame shape: {sample_df.shape}")
        
        try:
            df_json = sample_df.to_json(orient='columns', date_format='iso')
            logger.debug(f"Generated df_json length: {len(df_json)} characters")
            logger.debug(f"df_json preview (first 200 chars): {df_json[:200]}")
            
            import json
            from src.utils.json_utils import parse_df_json_safely
            
            try:
                parsed = parse_df_json_safely(df_json)
                logger.debug(f"df_json is valid JSON, parsed type: {type(parsed)}")
                if isinstance(parsed, dict):
                    logger.debug(f"JSON has {len(parsed)} top-level keys: {list(parsed.keys())[:10]}")
                    logger.debug(f"df_json format validated successfully - can be parsed by tools")
            except ValueError as json_err:
                logger.error(f"df_json validation failed: {json_err}")
                logger.error(f"df_json content (first 500 chars): {df_json[:500]}")
                logger.error(f"df_json content (last 200 chars): {df_json[-200:]}")
                logger.warning("Attempting alternative JSON serialization...")
                try:
                    df_json = sample_df.to_json(orient='records', date_format='iso')
                    parse_df_json_safely(df_json)  # Validate with normalization utility
                    logger.info("Alternative serialization (orient='records') succeeded and validated")
                except Exception as alt_err:
                    logger.error(f"Alternative serialization also failed: {alt_err}")
                    raise ValueError(f"Failed to serialize and validate DataFrame to JSON: {json_err}") from json_err
            
        except Exception as e:
            logger.error(f"Failed to prepare DataFrame JSON: {e}", exc_info=True)
            raise
        
        columns = df.columns.tolist()
        dtypes = {col: str(df[col].dtype) for col in columns}
        dataset_size = len(df)
        
        logger.info(f"Prepared workflow input: {len(columns)} columns, {dataset_size} total rows")
        logger.debug(f"Column dtypes: {dtypes}")
        
        # Create and start workflow
        try:
            self.workflow = ConfigWorkflow(self.llm)
            logger.debug("ConfigWorkflow instance created")
        except Exception as e:
            logger.error(f"Failed to create ConfigWorkflow: {e}", exc_info=True)
            raise
        
        try:
            logger.info("Calling workflow.start()...")
            self.current_state = self.workflow.start(
                df_json=df_json,
                columns=columns,
                dtypes=dtypes,
                dataset_size=dataset_size
            )
            logger.info("workflow.start() completed successfully")
            logger.debug(f"Current state keys: {list(self.current_state.keys())}")
            
            error_value = self.current_state.get("error")
            if error_value:
                logger.error(f"Workflow state contains error: {error_value}")
            else:
                logger.debug("Workflow state has no errors")
            
            return self._format_phase_result("classification")
            
        except PromptInjectionError as e:
            logger.warning(
                f"Prompt injection detected: confidence={e.confidence}, "
                f"reasoning={e.reasoning}, suspicious_content={e.suspicious_content[:200]}"
            )
            logger.info("[OBSERVABILITY] prompt_injection detection=failed confidence={:.2f}".format(e.confidence))
            
            user_message = (
                "Your uploaded data contains content that appears to be an attempt to "
                "manipulate the system. For security reasons, the workflow cannot proceed. "
                "Please review your data and remove any instructions or commands that are not "
                "part of the actual data values."
            )
            
            return {
                "phase": "validation",
                "status": "error",
                "error": user_message,
                "error_type": "prompt_injection",
                "confidence": e.confidence,
                "reasoning": e.reasoning
            }
            
        except Exception as e:
            logger.error(f"Workflow start failed: {e}", exc_info=True)
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "phase": "classification",
                "status": "error",
                "error": str(e)
            }
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current workflow state. Returns: Dict[str, Any]: Current state dictionary."""
        if not self.workflow:
            return {"phase": "not_started", "status": "pending"}
        
        phase = self.workflow.get_current_phase()
        return self._format_phase_result(phase)
    
    def confirm_classification(
        self, 
        modifications: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Confirm column classification and proceed to encoding. Args: modifications (Optional[Dict[str, Any]]): Modified classification. Returns: Dict[str, Any]: Encoding results."""
        if not self.workflow:
            raise RuntimeError("Workflow not started")
        
        logger.info("Confirming classification phase")
        
        try:
            self.current_state = self.workflow.confirm_classification(modifications)
            return self._format_phase_result("encoding")
            
        except Exception as e:
            logger.error(f"Classification confirmation failed: {e}")
            return {
                "phase": "encoding",
                "status": "error",
                "error": str(e)
            }
    
    def confirm_encoding(
        self,
        modifications: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Confirm feature encoding and proceed to model configuration. Args: modifications (Optional[Dict[str, Any]]): Modified encodings. Returns: Dict[str, Any]: Configuration results."""
        if not self.workflow:
            raise RuntimeError("Workflow not started")
        
        logger.info("Confirming encoding phase")
        
        try:
            self.current_state = self.workflow.confirm_encoding(modifications)
            return self._format_phase_result("configuration")
            
        except Exception as e:
            logger.error(f"Encoding confirmation failed: {e}")
            return {
                "phase": "configuration",
                "status": "error",
                "error": str(e)
            }
    
    def get_final_config(self) -> Optional[Dict[str, Any]]:
        """Get the final configuration if workflow is complete. Returns: Optional[Dict[str, Any]]: Final configuration or None."""
        if not self.workflow:
            return None
        
        return self.workflow.get_final_config()
    
    def is_complete(self) -> bool:
        """Check if the workflow has completed successfully. Returns: bool: True if complete."""
        if not self.workflow:
            return False
        return self.workflow.get_current_phase() == "complete"
    
    def _format_phase_result(self, phase: str) -> Dict[str, Any]:
        """Format the current state for UI consumption. Args: phase (str): Current phase name. Returns: Dict[str, Any]: Formatted result dictionary."""
        result = {
            "phase": phase,
            "status": "success" if not self.current_state.get("error") else "error"
        }
        
        if self.current_state.get("error"):
            result["error"] = self.current_state["error"]
        
        if phase == "classification" or phase == "starting":
            classification = self.current_state.get("column_classification", {})
            result["data"] = {
                "targets": classification.get("targets", []),
                "features": classification.get("features", []),
                "locations": classification.get("locations", []),
                "ignore": classification.get("ignore", []),
                "reasoning": classification.get("reasoning", "")
            }
            result["confirmed"] = self.current_state.get("classification_confirmed", False)
            
        elif phase == "encoding":
            encodings = self.current_state.get("feature_encodings", {})
            result["data"] = {
                "encodings": encodings.get("encodings", {}),
                "summary": encodings.get("summary", "")
            }
            result["confirmed"] = self.current_state.get("encodings_confirmed", False)
            
        elif phase == "configuration" or phase == "complete":
            model_config = self.current_state.get("model_config", {})
            final_config = self.current_state.get("final_config")
            
            result["data"] = {
                "features": model_config.get("features", []),
                "quantiles": model_config.get("quantiles", []),
                "hyperparameters": model_config.get("hyperparameters", {}),
                "reasoning": model_config.get("reasoning", "")
            }
            
            if final_config:
                result["final_config"] = final_config
                result["status"] = "complete"
        
        return result


def create_workflow_service(provider: str = "openai", model: Optional[str] = None) -> WorkflowService:
    """Factory function to create a WorkflowService instance. Args: provider (str): LLM provider name. model (Optional[str]): Model name. Returns: WorkflowService: WorkflowService instance."""
    return WorkflowService(provider=provider, model=model)


def get_workflow_providers() -> List[str]:
    """Get list of available LLM providers for the workflow. Returns: List[str]: Provider names."""
    return get_available_providers()

