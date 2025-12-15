"""Training API endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from src.api.dependencies import get_current_user
from src.api.dto.common import BaseResponse, PaginationResponse
from src.api.dto.training import (
    DataUploadResponse,
    TrainingJobRequest,
    TrainingJobResponse,
    TrainingJobStatusResponse,
    TrainingJobSummary,
)
from src.api.exceptions import InvalidInputError, TrainingJobNotFoundError
from src.api.storage import get_dataset_storage
from src.services.analytics_service import AnalyticsService
from src.services.training_service import TrainingService
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/training", tags=["training"])


def get_training_service() -> TrainingService:
    """Get training service instance. Returns: TrainingService: Training service."""
    return TrainingService()


def get_analytics_service() -> AnalyticsService:
    """Get analytics service instance. Returns: AnalyticsService: Analytics service."""
    return AnalyticsService()


@router.post("/data/upload", response_model=DataUploadResponse)
async def upload_training_data(
    file: UploadFile = File(..., description="CSV file"),
    dataset_name: Optional[str] = Form(default=None, description="Optional dataset name"),
    user: str = Depends(get_current_user),
    training_service: TrainingService = Depends(get_training_service),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
):
    """Upload training data CSV file. Args: file (UploadFile): CSV file. dataset_name (Optional[str]): Dataset name. user (str): Current user. training_service (TrainingService): Training service. analytics_service (AnalyticsService): Analytics service. Returns: DataUploadResponse: Upload response with dataset ID and summary. Raises: InvalidInputError: If file validation fails."""
    import uuid

    file_content = await file.read()
    is_valid, error_msg, df = training_service.validate_csv_file(file_content, file.filename)

    if not is_valid:
        raise InvalidInputError(error_msg or "Invalid CSV file")

    if df is None:
        raise InvalidInputError("Failed to parse CSV file")

    dataset_id = str(uuid.uuid4())

    storage = get_dataset_storage()
    storage.store(dataset_id, df)

    summary = analytics_service.get_data_summary(df)

    from src.api.dto.analytics import DataSummary

    data_summary = DataSummary(
        total_samples=summary.get("total_samples", len(df)),
        shape=summary.get("shape", df.shape),
        unique_counts={
            k.replace("unique_", ""): v
            for k, v in summary.items()
            if k.startswith("unique_") and k != "unique_counts"
        },
    )

    return DataUploadResponse(
        dataset_id=dataset_id,
        row_count=len(df),
        column_count=len(df.columns),
        summary=data_summary,
    )


@router.post("/jobs", response_model=TrainingJobResponse)
async def start_training(
    request: TrainingJobRequest,
    user: str = Depends(get_current_user),
    training_service: TrainingService = Depends(get_training_service),
):
    """Start a training job. Args: request (TrainingJobRequest): Training job request. user (str): Current user. training_service (TrainingService): Training service. Returns: TrainingJobResponse: Job response. Raises: InvalidInputError: If request validation fails."""
    dataset_id = request.dataset_id

    storage = get_dataset_storage()
    df = storage.get(dataset_id)

    if df is None:
        raise InvalidInputError(f"Dataset {dataset_id} not found")

    try:
        n_trials_value = (
            request.n_trials if (request.do_tune and request.n_trials is not None) else 20
        )
        job_id = training_service.start_training_async(
            data=df,
            config=request.config,
            remove_outliers=request.remove_outliers,
            do_tune=request.do_tune,
            n_trials=n_trials_value,
            additional_tag=request.additional_tag,
            dataset_name=request.dataset_name or "Unknown",
        )

        return TrainingJobResponse(job_id=job_id, status="QUEUED")
    except ValueError as e:
        raise InvalidInputError(str(e)) from e


@router.get("/jobs/{job_id}", response_model=TrainingJobStatusResponse)
async def get_training_job_status(
    job_id: str,
    user: str = Depends(get_current_user),
    training_service: TrainingService = Depends(get_training_service),
):
    """Get training job status. Args: job_id (str): Job identifier. user (str): Current user. training_service (TrainingService): Training service. Returns: TrainingJobStatusResponse: Job status. Raises: TrainingJobNotFoundError: If job not found."""
    job_status = training_service.get_job_status(job_id)

    if job_status is None:
        raise TrainingJobNotFoundError(job_id)

    from src.api.dto.training import TrainingResult

    result = None
    if job_status.get("status") == "COMPLETED" and job_status.get("run_id"):
        scores = job_status.get("scores")
        cv_mean_score = scores[0] if scores and len(scores) > 0 else None
        result = TrainingResult(
            run_id=job_status["run_id"],
            model_type="XGBoost",
            cv_mean_score=cv_mean_score,
        )

    return TrainingJobStatusResponse(
        job_id=job_id,
        status=job_status["status"],
        progress=(
            0.5
            if job_status["status"] == "RUNNING"
            else (1.0 if job_status["status"] == "COMPLETED" else 0.0)
        ),
        logs=job_status.get("logs", []),
        submitted_at=job_status.get("submitted_at"),
        completed_at=job_status.get("completed_at"),
        result=result,
        error=job_status.get("error"),
        run_id=job_status.get("run_id"),
    )


@router.get("/jobs", response_model=BaseResponse)
async def list_training_jobs(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(default=None, description="Filter by status"),
    user: str = Depends(get_current_user),
    training_service: TrainingService = Depends(get_training_service),
):
    """List training jobs. Args: limit (int): Maximum items. offset (int): Items to skip. status (Optional[str]): Status filter. user (str): Current user. training_service (TrainingService): Training service. Returns: BaseResponse: List of jobs with pagination."""
    all_jobs = training_service._jobs
    jobs_list = [
        TrainingJobSummary(
            job_id=job_id,
            status=job_data["status"],
            submitted_at=job_data.get("submitted_at"),
            completed_at=job_data.get("completed_at"),
            run_id=job_data.get("run_id"),
        )
        for job_id, job_data in all_jobs.items()
        if not status or job_data["status"] == status
    ]

    total = len(jobs_list)
    paginated_jobs = jobs_list[offset : offset + limit]
    has_more = offset + limit < total

    return BaseResponse(
        status="success",
        data={
            "jobs": [j.model_dump() for j in paginated_jobs],
            "pagination": PaginationResponse(
                total=total, limit=limit, offset=offset, has_more=has_more
            ).model_dump(),
        },
    )
