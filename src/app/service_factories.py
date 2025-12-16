"""Service factory functions for Streamlit app to match API dependency injection pattern."""

from typing import Optional

import streamlit as st

from src.services.analytics_service import AnalyticsService
from src.services.inference_service import InferenceService
from src.services.training_service import TrainingService
from src.services.workflow_service import WorkflowService


@st.cache_resource
def get_training_service() -> TrainingService:
    """Get training service instance. Returns: TrainingService: Training service."""
    return TrainingService()


@st.cache_resource
def get_inference_service() -> InferenceService:
    """Get inference service instance. Returns: InferenceService: Inference service."""
    return InferenceService()


@st.cache_resource
def get_analytics_service() -> AnalyticsService:
    """Get analytics service instance. Returns: AnalyticsService: Analytics service."""
    return AnalyticsService()


def get_workflow_service(provider: str = "openai", model: Optional[str] = None) -> WorkflowService:
    """Get workflow service instance. Note: Not cached as it's stateful and should be stored in session state.

    Args:
        provider (str): LLM provider name.
        model (Optional[str]): Optional model name override.

    Returns:
        WorkflowService: Workflow service.
    """
    return WorkflowService(provider=provider, model=model)
