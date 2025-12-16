"""Inference/prediction API endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends

from src.api.dependencies import get_current_user
from src.api.dto.inference import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.api.exceptions import InvalidInputError
from src.api.exceptions import ModelNotFoundError as APIModelNotFoundError
from src.services.inference_service import InferenceService
from src.services.inference_service import InvalidInputError as ServiceInvalidInputError
from src.services.inference_service import ModelNotFoundError
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/models", tags=["inference"])


def get_inference_service() -> InferenceService:
    """Get inference service instance.

    Returns:
        InferenceService: Inference service.
    """
    return InferenceService()


@router.post("/{run_id}/predict", response_model=PredictionResponse)
async def predict(
    run_id: str,
    request: PredictionRequest,
    user: str = Depends(get_current_user),
    inference_service: InferenceService = Depends(get_inference_service),
):
    """Predict salary quantiles for given features.

    Args:
        run_id (str): MLflow run ID.
        request (PredictionRequest): Prediction request.
        user (str): Current user.
        inference_service (InferenceService): Inference service.

    Returns:
        PredictionResponse: Prediction results.

    Raises:
        APIModelNotFoundError: If model not found.
        InvalidInputError: If input validation fails.
    """
    try:
        model = inference_service.load_model(run_id)
        result = inference_service.predict(model, request.features)

        from src.api.dto.inference import PredictionMetadata

        return PredictionResponse(
            predictions=result.predictions,
            metadata=PredictionMetadata(
                model_run_id=run_id,
                prediction_timestamp=datetime.now(),
                location_zone=result.metadata.get("location_zone"),
            ),
        )
    except ModelNotFoundError as e:
        raise APIModelNotFoundError(run_id) from e
    except ServiceInvalidInputError as e:
        raise InvalidInputError(str(e)) from e


@router.post("/{run_id}/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    run_id: str,
    request: BatchPredictionRequest,
    user: str = Depends(get_current_user),
    inference_service: InferenceService = Depends(get_inference_service),
):
    """Batch predict salary quantiles for multiple feature sets.

    Args:
        run_id (str): MLflow run ID.
        request (BatchPredictionRequest): Batch prediction request.
        user (str): Current user.
        inference_service (InferenceService): Inference service.

    Returns:
        BatchPredictionResponse: Batch prediction results.

    Raises:
        APIModelNotFoundError: If model not found.
        InvalidInputError: If input validation fails.
    """
    try:
        model = inference_service.load_model(run_id)
        predictions = []

        for features in request.features:
            result = inference_service.predict(model, features)
            from src.api.dto.inference import PredictionMetadata

            predictions.append(
                PredictionResponse(
                    predictions=result.predictions,
                    metadata=PredictionMetadata(
                        model_run_id=run_id,
                        prediction_timestamp=datetime.now(),
                        location_zone=result.metadata.get("location_zone"),
                    ),
                )
            )

        return BatchPredictionResponse(predictions=predictions, total=len(predictions))
    except ModelNotFoundError as e:
        raise APIModelNotFoundError(run_id) from e
    except ServiceInvalidInputError as e:
        raise InvalidInputError(str(e)) from e
