"""Workflow service that wraps the LangGraph workflow and provides a simple interface for the Streamlit UI to manage the multi-step configuration process."""

from typing import Any, Dict, List, Optional

import pandas as pd

from src.agents.workflow import ConfigWorkflow, PromptInjectionError
from src.llm.client import get_available_providers, get_langchain_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WorkflowService:
    """Service to manage the agentic configuration workflow with methods to start workflows, get state, confirm phases, and retrieve final configuration."""

    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        """Initializes the workflow service. Args: provider (str): LLM provider name. model (Optional[str]): Optional model name override."""
        self.provider = provider
        self.model = model
        self.workflow: Optional[ConfigWorkflow] = None
        self.current_state: Dict[str, Any] = {}

        try:
            self.llm = get_langchain_llm(provider=provider, model=model)
            logger.info(f"WorkflowService initialized with provider: {provider}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate DataFrame before processing. Args: df (pd.DataFrame): DataFrame to validate. Returns: None. Raises: ValueError: If DataFrame is invalid."""
        if df is None:
            raise ValueError("DataFrame cannot be None")
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        if len(df.columns) == 0:
            raise ValueError("DataFrame must have at least one column")

    def _prepare_workflow_input(
        self, df: pd.DataFrame, sample_size: int
    ) -> tuple[str, list[str], Dict[str, str], int]:
        """Prepare and validate workflow input data. Args: df (pd.DataFrame): Input DataFrame. sample_size (int): Rows to sample. Returns: tuple[str, list[str], Dict[str, str], int]: (df_json, columns, dtypes, dataset_size). Raises: ValueError: If data preparation fails."""
        self._validate_dataframe(df)

        sample_df = df.head(sample_size)
        logger.debug(f"Sampled DataFrame shape: {sample_df.shape}")

        df_json = self._serialize_dataframe(sample_df)
        columns = df.columns.tolist()
        dtypes = {col: str(df[col].dtype) for col in columns}
        dataset_size = len(df)

        logger.info(f"Prepared workflow input: {len(columns)} columns, {dataset_size} total rows")
        logger.debug(f"Column dtypes: {dtypes}")

        return df_json, columns, dtypes, dataset_size

    def _serialize_dataframe(self, df: pd.DataFrame) -> str:
        """Serialize DataFrame to JSON with validation and fallback. Args: df (pd.DataFrame): DataFrame to serialize. Returns: str: JSON string. Raises: ValueError: If serialization fails."""
        from src.utils.json_utils import parse_df_json_safely

        # Try 'records' orient first as it's more compact and less likely to cause parsing issues
        # when passed through LLM tool calls
        try:
            df_json = df.to_json(orient="records", date_format="iso")
            logger.debug(f"Generated df_json (records) length: {len(df_json)} characters")

            # Validate JSON can be parsed
            parse_df_json_safely(df_json)
            logger.debug("df_json validated successfully")
            from typing import cast

            return cast(str, df_json)
        except ValueError as records_err:
            logger.warning(f"Records serialization failed, trying columns: {records_err}")
            try:
                df_json = df.to_json(orient="columns", date_format="iso")
                logger.debug(f"Generated df_json (columns) length: {len(df_json)} characters")
                parse_df_json_safely(df_json)
                logger.debug("df_json (columns) validated successfully")
                from typing import cast

                return cast(str, df_json)
            except Exception as columns_err:
                error_msg = (
                    f"Failed to serialize DataFrame to JSON. "
                    f"Records error: {records_err}. "
                    f"Columns error: {columns_err}. "
                    f"Please check your data for unsupported types or encoding issues."
                )
                logger.error(error_msg)
                raise ValueError(error_msg) from records_err

    def start_workflow(
        self, df: pd.DataFrame, sample_size: int = 50, preset: Optional[str] = None
    ) -> Dict[str, Any]:
        """Start a new configuration workflow. Args: df (pd.DataFrame): Input DataFrame. sample_size (int): Rows to sample. preset (Optional[str]): Optional preset prompt name. Returns: Dict[str, Any]: Classification results. Raises: ValueError: If data validation fails. RuntimeError: If workflow initialization fails."""
        logger.info(f"Starting workflow with {len(df)} rows, sampling {sample_size}")
        logger.debug(f"DataFrame columns: {df.columns.tolist()}")
        logger.debug(f"DataFrame dtypes: {df.dtypes.to_dict()}")

        try:
            df_json, columns, dtypes, dataset_size = self._prepare_workflow_input(df, sample_size)
        except ValueError as e:
            error_msg = f"Data preparation failed: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

        try:
            self.workflow = ConfigWorkflow(self.llm)
            logger.debug("ConfigWorkflow instance created")
        except Exception as e:
            error_msg = f"Failed to initialize workflow: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

        try:
            logger.info("Calling workflow.start()...")
            workflow_state = self.workflow.start(
                df_json=df_json,
                columns=columns,
                dtypes=dtypes,
                dataset_size=dataset_size,
                preset=preset,
            )
            self.current_state = workflow_state
            logger.info("Workflow started successfully")
            logger.debug(f"Current state keys: {list(self.current_state.keys())}")
            logger.debug(
                f"Column classification keys: {list(self.current_state.get('column_classification', {}).keys())}"
            )

            error_value = self.current_state.get("error")
            if error_value:
                logger.error(f"Workflow state contains error: {error_value}")
            else:
                logger.debug("Workflow state has no errors")
                classification = self.current_state.get("column_classification", {})
                targets = classification.get("targets", [])
                features = classification.get("features", [])
                ignore = classification.get("ignore", [])
                logger.info(
                    f"Classification result: {len(targets)} targets, {len(features)} features, {len(ignore)} ignore"
                )
                logger.debug(f"Classification targets: {targets}")
                logger.debug(f"Classification features: {features}")
                logger.debug(f"Classification ignore: {ignore}")

                if not targets and not features and not ignore:
                    logger.warning(
                        "WARNING: Classification result is empty! All columns will show as Unclassified. "
                        "This may indicate a parsing issue with the LLM response."
                    )
                    logger.debug(f"Full classification dict: {classification}")
                    logger.debug(f"Reasoning present: {bool(classification.get('reasoning'))}")

            return self._format_phase_result("classification")

        except PromptInjectionError as e:
            logger.warning(
                f"Prompt injection detected: confidence={e.confidence}, "
                f"reasoning={e.reasoning}, suspicious_content={e.suspicious_content[:200]}"
            )
            logger.info(
                "[OBSERVABILITY] prompt_injection detection=failed confidence={:.2f}".format(
                    e.confidence
                )
            )

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
                "reasoning": e.reasoning,
            }

        except Exception as e:
            logger.error(f"Workflow start failed: {e}", exc_info=True)
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"phase": "classification", "status": "error", "error": str(e)}

    def get_current_state(self) -> Dict[str, Any]:
        """Get the current workflow state. Returns: Dict[str, Any]: Current state dictionary."""
        if not self.workflow:
            return {"phase": "not_started", "status": "pending"}

        phase = self.workflow.get_current_phase()
        return self._format_phase_result(phase)

    def confirm_classification(
        self, modifications: Optional[Dict[str, Any]] = None
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
            return {"phase": "encoding", "status": "error", "error": str(e)}

    def confirm_encoding(self, modifications: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Confirm feature encoding and proceed to model configuration. Args: modifications (Optional[Dict[str, Any]]): Modified encodings. Returns: Dict[str, Any]: Configuration results."""
        if not self.workflow:
            raise RuntimeError("Workflow not started")

        logger.info("Confirming encoding phase")

        try:
            self.current_state = self.workflow.confirm_encoding(modifications)
            return self._format_phase_result("configuration")

        except Exception as e:
            logger.error(f"Encoding confirmation failed: {e}")
            return {"phase": "configuration", "status": "error", "error": str(e)}

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
        state: Dict[str, Any] = {}
        if self.workflow and self.workflow.current_state:
            state = self.workflow.current_state
        else:
            state = self.current_state

        result: Dict[str, Any] = {
            "phase": phase,
            "status": "success" if not state.get("error") else "error",
        }

        if state.get("error"):
            result["error"] = state["error"]

        if phase == "classification" or phase == "starting":
            classification = state.get("column_classification", {})
            result["data"] = {
                "targets": classification.get("targets", []),
                "features": classification.get("features", []),
                "ignore": classification.get("ignore", []),
                "reasoning": classification.get("reasoning", ""),
            }
            result["confirmed"] = state.get("classification_confirmed", False)

        elif phase == "encoding":
            encodings = state.get("feature_encodings", {})
            result["data"] = {
                "encodings": encodings.get("encodings", {}),
                "summary": encodings.get("summary", ""),
            }
            result["confirmed"] = state.get("encodings_confirmed", False)

        elif phase == "configuration" or phase == "complete":
            model_config = state.get("model_config", {})
            final_config = state.get("final_config")

            result["data"] = {
                "features": model_config.get("features", []),
                "quantiles": model_config.get("quantiles", []),
                "hyperparameters": model_config.get("hyperparameters", {}),
                "reasoning": model_config.get("reasoning", ""),
            }

            if final_config:
                result["final_config"] = final_config
                result["status"] = "complete"

        return result


def create_workflow_service(
    provider: str = "openai", model: Optional[str] = None
) -> WorkflowService:
    """Factory function to create a WorkflowService instance. Args: provider (str): LLM provider name. model (Optional[str]): Model name. Returns: WorkflowService: WorkflowService instance."""
    return WorkflowService(provider=provider, model=model)


def get_workflow_providers() -> List[str]:
    """Get list of available LLM providers for the workflow. Returns: List[str]: Provider names."""
    return get_available_providers()
