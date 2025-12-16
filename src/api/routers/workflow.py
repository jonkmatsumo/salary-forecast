"""Configuration workflow API endpoints."""

import uuid
from typing import Dict, Optional

from fastapi import APIRouter, Depends

from src.api.dependencies import get_current_user
from src.api.dto.workflow import (
    ClassificationConfirmationRequest,
    ConfigurationFinalizationRequest,
    EncodingConfirmationRequest,
    WorkflowCompleteResponse,
    WorkflowProgressResponse,
    WorkflowStartRequest,
    WorkflowStartResponse,
    WorkflowStateResponse,
)
from src.api.exceptions import WorkflowNotFoundError
from src.services.workflow_service import WorkflowService
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/workflow", tags=["workflow"])

_workflow_storage: Dict[str, WorkflowService] = {}


def get_workflow_service(workflow_id: str) -> Optional[WorkflowService]:
    """Get workflow service by ID.

    Args:
        workflow_id (str): Workflow identifier.

    Returns:
        Optional[WorkflowService]: Workflow service or None.
    """
    return _workflow_storage.get(workflow_id)


@router.post("/start", response_model=WorkflowStartResponse)
async def start_workflow(
    request: WorkflowStartRequest,
    user: str = Depends(get_current_user),
):
    """Start a configuration workflow.

    Args:
        request (WorkflowStartRequest): Workflow start request.
        user (str): Current user.

    Returns:
        WorkflowStartResponse: Workflow start response.

    Raises:
        InvalidInputError: If workflow start fails.
    """
    from io import StringIO

    import pandas as pd

    workflow_id = str(uuid.uuid4())
    service = WorkflowService(provider=request.provider)

    df_json = request.data
    df = pd.read_json(StringIO(df_json), orient="records")

    result = service.start_workflow(df, sample_size=50, preset=request.preset)

    if result.get("status") == "error":
        from src.api.exceptions import InvalidInputError

        error_msg = result.get("error", "Workflow start failed")
        raise InvalidInputError(error_msg)

    _workflow_storage[workflow_id] = service

    from src.api.dto.workflow import WorkflowState

    workflow_state = WorkflowState(
        phase=result.get("phase", "classification"),
        status=result.get("status", "success"),
        current_result=result.get("data"),
    )

    return WorkflowStartResponse(
        workflow_id=workflow_id,
        phase="classification",
        state=workflow_state,
    )


@router.get("/{workflow_id}", response_model=WorkflowStateResponse)
async def get_workflow_state(
    workflow_id: str,
    user: str = Depends(get_current_user),
):
    """Get workflow state.

    Args:
        workflow_id (str): Workflow identifier.
        user (str): Current user.

    Returns:
        WorkflowStateResponse: Workflow state.

    Raises:
        WorkflowNotFoundError: If workflow not found.
    """
    service = get_workflow_service(workflow_id)

    if not service:
        raise WorkflowNotFoundError(workflow_id)

    state_result = service.get_current_state()

    return WorkflowStateResponse(
        workflow_id=workflow_id,
        phase=state_result.get("phase", "unknown"),
        state=service.current_state if service.workflow else {},
        current_result=state_result.get("data", {}),
    )


@router.post("/{workflow_id}/confirm/classification", response_model=WorkflowProgressResponse)
async def confirm_classification(
    workflow_id: str,
    request: ClassificationConfirmationRequest,
    user: str = Depends(get_current_user),
):
    """Confirm classification phase.

    Args:
        workflow_id (str): Workflow identifier.
        request (ClassificationConfirmationRequest): Classification confirmation.
        user (str): Current user.

    Returns:
        WorkflowProgressResponse: Progress response.

    Raises:
        WorkflowNotFoundError: If workflow not found.
        InvalidInputError: If confirmation fails.
    """
    service = get_workflow_service(workflow_id)

    if not service:
        raise WorkflowNotFoundError(workflow_id)

    modifications = {
        "targets": request.modifications.targets,
        "features": request.modifications.features,
        "ignore": request.modifications.ignore,
    }

    result = service.confirm_classification(modifications)

    if result.get("status") == "error":
        from src.api.exceptions import InvalidInputError

        error_msg = result.get("error", "Classification confirmation failed")
        raise InvalidInputError(error_msg)

    return WorkflowProgressResponse(
        workflow_id=workflow_id,
        phase="encoding",
        result=result.get("data", {}),
    )


@router.post("/{workflow_id}/confirm/encoding", response_model=WorkflowProgressResponse)
async def confirm_encoding(
    workflow_id: str,
    request: EncodingConfirmationRequest,
    user: str = Depends(get_current_user),
):
    """Confirm encoding phase.

    Args:
        workflow_id (str): Workflow identifier.
        request (EncodingConfirmationRequest): Encoding confirmation.
        user (str): Current user.

    Returns:
        WorkflowProgressResponse: Progress response.

    Raises:
        WorkflowNotFoundError: If workflow not found.
        InvalidInputError: If confirmation fails.
    """
    service = get_workflow_service(workflow_id)

    if not service:
        raise WorkflowNotFoundError(workflow_id)

    encodings = {}
    for col, enc_config in request.modifications.encodings.items():
        encodings[col] = {
            "type": enc_config.type,
            "mapping": enc_config.mapping or {},
            "reasoning": enc_config.reasoning or "",
        }

    optional_encodings = {}
    for col, opt_enc_config in request.modifications.optional_encodings.items():
        optional_encodings[col] = {
            "type": opt_enc_config.type,
            "params": opt_enc_config.params,
        }

    modifications = {
        "encodings": encodings,
        "optional_encodings": optional_encodings,
    }

    result = service.confirm_encoding(modifications)

    if result.get("status") == "error":
        from src.api.exceptions import InvalidInputError

        error_msg = result.get("error", "Encoding confirmation failed")
        raise InvalidInputError(error_msg)

    return WorkflowProgressResponse(
        workflow_id=workflow_id,
        phase="configuration",
        result=result.get("data", {}),
    )


@router.post("/{workflow_id}/finalize", response_model=WorkflowCompleteResponse)
async def finalize_configuration(
    workflow_id: str,
    request: ConfigurationFinalizationRequest,
    user: str = Depends(get_current_user),
):
    """Finalize configuration.

    Args:
        workflow_id (str): Workflow identifier.
        request (ConfigurationFinalizationRequest): Finalization request.
        user (str): Current user.

    Returns:
        WorkflowCompleteResponse: Complete response.

    Raises:
        WorkflowNotFoundError: If workflow not found.
        InvalidInputError: If finalization fails.
    """
    service = get_workflow_service(workflow_id)

    if not service:
        raise WorkflowNotFoundError(workflow_id)

    final_config = service.get_final_config()

    if not final_config:
        from src.api.exceptions import InvalidInputError

        raise InvalidInputError(
            "No final configuration available. Ensure workflow is in configuration phase."
        )

    final_config["model"]["features"] = [
        {"name": f.name, "monotone_constraint": f.monotone_constraint} for f in request.features
    ]
    final_config["model"]["quantiles"] = request.quantiles
    final_config["model"]["hyperparameters"] = {
        "training": request.hyperparameters.training,
        "cv": request.hyperparameters.cv,
    }

    if request.location_settings:
        final_config["location_settings"] = request.location_settings

    return WorkflowCompleteResponse(
        workflow_id=workflow_id,
        phase="complete",
        final_config=final_config,
    )
