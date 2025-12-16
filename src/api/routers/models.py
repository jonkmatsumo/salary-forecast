"""Model management API endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_current_user
from src.api.dto.common import BaseResponse, PaginationResponse
from src.api.dto.models import (
    ModelDetailsResponse,
    ModelMetadata,
    ModelSchema,
    ModelSchemaResponse,
    ProximityFeatureSchema,
    RankedFeatureSchema,
)
from src.api.exceptions import ModelNotFoundError as APIModelNotFoundError
from src.services.inference_service import InferenceService, ModelNotFoundError
from src.services.model_registry import ModelRegistry
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/models", tags=["models"])


def get_model_registry() -> ModelRegistry:
    """Get model registry instance.

    Returns:
        ModelRegistry: Model registry.
    """
    return ModelRegistry()


def get_inference_service() -> InferenceService:
    """Get inference service instance.

    Returns:
        InferenceService: Inference service.
    """
    return InferenceService()


@router.get("", response_model=BaseResponse)
async def list_models(
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of items to return"),
    offset: int = Query(default=0, ge=0, description="Number of items to skip"),
    experiment_name: Optional[str] = Query(default=None, description="Filter by experiment name"),
    user: str = Depends(get_current_user),
    registry: ModelRegistry = Depends(get_model_registry),
):
    """List all available models.

    Args:
        limit (int): Maximum items to return.
        offset (int): Items to skip.
        experiment_name (Optional[str]): Filter by experiment.
        user (str): Current user.
        registry (ModelRegistry): Model registry.

    Returns:
        BaseResponse: List of models with pagination.
    """
    runs = registry.list_models()

    if experiment_name:
        runs = [r for r in runs if r.get("tags.experiment_name") == experiment_name]

    total = len(runs)
    paginated_runs = runs[offset : offset + limit]

    models = []
    for run in paginated_runs:
        models.append(
            ModelMetadata(
                run_id=run["run_id"],
                start_time=run["start_time"],
                model_type=run.get("tags.model_type", "XGBoost"),
                cv_mean_score=run.get("metrics.cv_mean_score"),
                dataset_name=run.get("tags.dataset_name", "Unknown"),
                additional_tag=run.get("tags.additional_tag"),
            )
        )

    has_more = offset + limit < total

    return BaseResponse(
        status="success",
        data={
            "models": [m.model_dump() for m in models],
            "pagination": PaginationResponse(
                total=total, limit=limit, offset=offset, has_more=has_more
            ).model_dump(),
        },
    )


@router.get("/{run_id}", response_model=ModelDetailsResponse)
async def get_model_details(
    run_id: str,
    user: str = Depends(get_current_user),
    inference_service: InferenceService = Depends(get_inference_service),
):
    """Get detailed information about a model.

    Args:
        run_id (str): MLflow run ID.
        user (str): Current user.
        inference_service (InferenceService): Inference service.

    Returns:
        ModelDetailsResponse: Model details.

    Raises:
        APIModelNotFoundError: If model not found.
    """
    try:
        model = inference_service.load_model(run_id)
        schema = inference_service.get_model_schema(model)

        ranked_features = [
            RankedFeatureSchema(
                name=name,
                levels=list(model.ranked_encoders[name].mapping.keys()),
                encoding_type="ranked",
            )
            for name in schema.ranked_features
        ]

        proximity_features = [
            ProximityFeatureSchema(name=name, encoding_type="proximity")
            for name in schema.proximity_features
        ]

        model_schema = ModelSchema(
            ranked_features=ranked_features,
            proximity_features=proximity_features,
            numerical_features=schema.numerical_features,
        )

        registry = ModelRegistry()
        runs = registry.list_models()
        run_data = next((r for r in runs if r["run_id"] == run_id), None)

        if not run_data:
            raise APIModelNotFoundError(run_id)

        metadata = ModelMetadata(
            run_id=run_id,
            start_time=run_data["start_time"],
            model_type=run_data.get("tags.model_type", "XGBoost"),
            cv_mean_score=run_data.get("metrics.cv_mean_score"),
            dataset_name=run_data.get("tags.dataset_name", "Unknown"),
            additional_tag=run_data.get("tags.additional_tag"),
        )

        return ModelDetailsResponse(
            run_id=run_id,
            metadata=metadata,
            model_schema=model_schema,
            feature_names=schema.all_feature_names,
            targets=schema.targets,
            quantiles=schema.quantiles,
        )
    except ModelNotFoundError as e:
        raise APIModelNotFoundError(run_id) from e


@router.get("/{run_id}/schema", response_model=ModelSchemaResponse)
async def get_model_schema(
    run_id: str,
    user: str = Depends(get_current_user),
    inference_service: InferenceService = Depends(get_inference_service),
):
    """Get model schema. Args: run_id (str): MLflow run ID. user (str): Current user. inference_service (InferenceService): Inference service. Returns: ModelSchemaResponse: Model schema. Raises: APIModelNotFoundError: If model not found."""
    try:
        model = inference_service.load_model(run_id)
        schema = inference_service.get_model_schema(model)

        ranked_features = [
            RankedFeatureSchema(
                name=name,
                levels=list(model.ranked_encoders[name].mapping.keys()),
                encoding_type="ranked",
            )
            for name in schema.ranked_features
        ]

        proximity_features = [
            ProximityFeatureSchema(name=name, encoding_type="proximity")
            for name in schema.proximity_features
        ]

        model_schema = ModelSchema(
            ranked_features=ranked_features,
            proximity_features=proximity_features,
            numerical_features=schema.numerical_features,
        )

        return ModelSchemaResponse(run_id=run_id, model_schema=model_schema)
    except ModelNotFoundError as e:
        raise APIModelNotFoundError(run_id) from e
