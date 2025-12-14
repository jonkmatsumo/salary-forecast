"""
Services module for the salary forecasting application.

Provides:
- ConfigGenerator: Heuristic-based configuration generation
- WorkflowService: AI-powered multi-step configuration workflow
- AnalyticsService: Data and model analytics
- ModelRegistry: MLflow model management
- TrainingService: Model training orchestration
"""

from src.services.analytics_service import AnalyticsService
from src.services.config_generator import ConfigGenerator
from src.services.inference_service import InferenceService
from src.services.workflow_service import (
    WorkflowService,
    create_workflow_service,
    get_workflow_providers,
)

__all__ = [
    "ConfigGenerator",
    "WorkflowService",
    "create_workflow_service",
    "get_workflow_providers",
    "AnalyticsService",
    "InferenceService",
]
