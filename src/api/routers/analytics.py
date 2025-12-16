"""Analytics API endpoints."""

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_current_user
from src.api.dto.analytics import (
    DataSummaryRequest,
    DataSummaryResponse,
    FeatureImportance,
    FeatureImportanceResponse,
)
from src.api.exceptions import InvalidInputError
from src.api.exceptions import ModelNotFoundError as APIModelNotFoundError
from src.services.analytics_service import AnalyticsService
from src.services.inference_service import InferenceService, ModelNotFoundError
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analytics"])


def get_analytics_service() -> AnalyticsService:
    """Get analytics service instance.

    Returns:
        AnalyticsService: Analytics service.
    """
    return AnalyticsService()


def get_inference_service() -> InferenceService:
    """Get inference service instance.

    Returns:
        InferenceService: Inference service.
    """
    return InferenceService()


@router.post("/analytics/data-summary", response_model=DataSummaryResponse)
async def get_data_summary(
    request: DataSummaryRequest,
    user: str = Depends(get_current_user),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
):
    """Get data summary statistics.

    Args:
        request (DataSummaryRequest): Data summary request.
        user (str): Current user.
        analytics_service (AnalyticsService): Analytics service.

    Returns:
        DataSummaryResponse: Data summary.

    Raises:
        InvalidInputError: If data parsing fails.
    """
    from io import StringIO

    import pandas as pd

    try:
        df = pd.read_json(StringIO(request.data), orient="records")
        summary = analytics_service.get_data_summary(df)

        unique_counts = {
            k.replace("unique_", ""): v
            for k, v in summary.items()
            if k.startswith("unique_") and k != "unique_counts"
        }

        return DataSummaryResponse(
            total_samples=summary.get("total_samples", len(df)),
            shape=summary.get("shape", df.shape),
            unique_counts=unique_counts,
        )
    except Exception as e:
        raise InvalidInputError(f"Failed to parse data: {str(e)}") from e


@router.get(
    "/models/{run_id}/analytics/feature-importance", response_model=FeatureImportanceResponse
)
async def get_feature_importance(
    run_id: str,
    target: str = Query(..., description="Target column name"),
    quantile: float = Query(..., ge=0.0, le=1.0, description="Quantile value (0.0-1.0)"),
    user: str = Depends(get_current_user),
    inference_service: InferenceService = Depends(get_inference_service),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
):
    """Get feature importance for a model.

    Args:
        run_id (str): MLflow run ID.
        target (str): Target column.
        quantile (float): Quantile value.
        user (str): Current user.
        inference_service (InferenceService): Inference service.
        analytics_service (AnalyticsService): Analytics service.

    Returns:
        FeatureImportanceResponse: Feature importance.

    Raises:
        APIModelNotFoundError: If model not found.
        InvalidInputError: If feature importance not found.
    """
    try:
        model = inference_service.load_model(run_id)
        df_imp = analytics_service.get_feature_importance(model, target, quantile)

        if df_imp is None:
            raise InvalidInputError(
                f"No feature importance found for target '{target}' at quantile {quantile}"
            )

        features = [
            FeatureImportance(name=row["Feature"], gain=row["Gain"]) for _, row in df_imp.iterrows()
        ]

        return FeatureImportanceResponse(features=features)
    except ModelNotFoundError as e:
        raise APIModelNotFoundError(run_id) from e
