"""Unit tests for training router endpoints."""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.api.dto.training import TrainingJobRequest
from src.api.exceptions import InvalidInputError, TrainingJobNotFoundError
from src.api.routers.training import (
    get_training_job_status,
    list_training_jobs,
    start_training,
    upload_training_data,
)
from src.api.storage import get_dataset_storage
from src.services.analytics_service import AnalyticsService
from src.services.training_service import TrainingService


@pytest.fixture(autouse=True)
def clear_storage():
    """Clear dataset storage before each test."""
    storage = get_dataset_storage()
    if hasattr(storage, "_datasets"):
        storage._datasets.clear()
    yield
    if hasattr(storage, "_datasets"):
        storage._datasets.clear()


class TestUploadTrainingData:
    """Tests for upload_training_data endpoint."""

    def test_upload_training_data_df_is_none(self):
        """Test df is None after validation raises InvalidInputError."""
        mock_file = MagicMock()
        mock_file.filename = "test.csv"
        
        async def mock_read():
            return b"col1,col2\n1,2"
        mock_file.read = mock_read
        
        training_service = MagicMock(spec=TrainingService)
        training_service.validate_csv_file.return_value = (True, None, None)
        
        analytics_service = MagicMock(spec=AnalyticsService)
        
        with pytest.raises(InvalidInputError) as exc_info:
            asyncio.run(upload_training_data(
                file=mock_file,
                dataset_name=None,
                user="test_user",
                training_service=training_service,
                analytics_service=analytics_service,
            ))
        
        assert "Failed to parse CSV file" in str(exc_info.value.message)

    def test_upload_training_data_invalid_file(self):
        """Test invalid CSV file raises InvalidInputError."""
        mock_file = MagicMock()
        mock_file.filename = "test.csv"
        
        async def mock_read():
            return b"invalid"
        mock_file.read = mock_read
        
        training_service = MagicMock(spec=TrainingService)
        training_service.validate_csv_file.return_value = (False, "Invalid format", None)
        
        analytics_service = MagicMock(spec=AnalyticsService)
        
        with pytest.raises(InvalidInputError) as exc_info:
            asyncio.run(upload_training_data(
                file=mock_file,
                dataset_name=None,
                user="test_user",
                training_service=training_service,
                analytics_service=analytics_service,
            ))
        
        assert "Invalid format" in str(exc_info.value.message)


class TestStartTraining:
    """Tests for start_training endpoint."""

    def test_start_training_dataset_not_found(self):
        """Test dataset not found raises InvalidInputError."""
        training_service = MagicMock(spec=TrainingService)
        
        request = TrainingJobRequest(
            dataset_id="nonexistent",
            config={"model": {"targets": ["target1"]}},
        )
        
        with pytest.raises(InvalidInputError) as exc_info:
            asyncio.run(start_training(request, user="test_user", training_service=training_service))
        
        assert "nonexistent" in str(exc_info.value.message)
        assert "not found" in str(exc_info.value.message)

    def test_start_training_value_error_from_service(self):
        """Test ValueError from service raises InvalidInputError."""
        df = pd.DataFrame({"col1": [1, 2]})
        storage = get_dataset_storage()
        storage.store("test_dataset", df)
        
        training_service = MagicMock(spec=TrainingService)
        training_service.start_training_async.side_effect = ValueError("Invalid config")
        
        request = TrainingJobRequest(
            dataset_id="test_dataset",
            config={"model": {"targets": ["target1"]}},
        )
        
        with pytest.raises(InvalidInputError) as exc_info:
            asyncio.run(start_training(request, user="test_user", training_service=training_service))
        
        assert "Invalid config" in str(exc_info.value.message)
        assert exc_info.value.__cause__ is not None


class TestGetTrainingJobStatus:
    """Tests for get_training_job_status endpoint."""

    def test_get_training_job_status_not_found(self):
        """Test job not found raises TrainingJobNotFoundError."""
        training_service = MagicMock(spec=TrainingService)
        training_service.get_job_status.return_value = None
        
        with pytest.raises(TrainingJobNotFoundError) as exc_info:
            asyncio.run(get_training_job_status("nonexistent", user="test_user", training_service=training_service))
        
        assert "nonexistent" in str(exc_info.value.message)

    def test_get_training_job_status_completed_with_run_id_and_scores(self):
        """Test completed job with run_id and scores."""
        training_service = MagicMock(spec=TrainingService)
        training_service.get_job_status.return_value = {
            "status": "COMPLETED",
            "run_id": "run123",
            "scores": [0.95, 0.94, 0.96],
            "logs": ["log1", "log2"],
            "submitted_at": datetime(2023, 1, 1),
            "completed_at": datetime(2023, 1, 2),
        }
        
        response = asyncio.run(get_training_job_status("job123", user="test_user", training_service=training_service))
        
        assert response.status == "COMPLETED"
        assert response.progress == 1.0
        assert response.run_id == "run123"
        assert response.result is not None
        assert response.result.run_id == "run123"
        assert response.result.cv_mean_score == 0.95

    def test_get_training_job_status_running(self):
        """Test running job has progress=0.5."""
        training_service = MagicMock(spec=TrainingService)
        training_service.get_job_status.return_value = {
            "status": "RUNNING",
            "logs": ["log1"],
            "submitted_at": datetime(2023, 1, 1),
        }
        
        response = asyncio.run(get_training_job_status("job123", user="test_user", training_service=training_service))
        
        assert response.status == "RUNNING"
        assert response.progress == 0.5

    def test_get_training_job_status_completed_progress(self):
        """Test completed job has progress=1.0."""
        training_service = MagicMock(spec=TrainingService)
        training_service.get_job_status.return_value = {
            "status": "COMPLETED",
            "run_id": "run123",
            "submitted_at": datetime(2023, 1, 1),
            "completed_at": datetime(2023, 1, 2),
        }
        
        response = asyncio.run(get_training_job_status("job123", user="test_user", training_service=training_service))
        
        assert response.status == "COMPLETED"
        assert response.progress == 1.0

    def test_get_training_job_status_queued_progress(self):
        """Test queued job has progress=0.0."""
        training_service = MagicMock(spec=TrainingService)
        training_service.get_job_status.return_value = {
            "status": "QUEUED",
            "submitted_at": datetime(2023, 1, 1),
        }
        
        response = asyncio.run(get_training_job_status("job123", user="test_user", training_service=training_service))
        
        assert response.status == "QUEUED"
        assert response.progress == 0.0

    def test_get_training_job_status_failed_progress(self):
        """Test failed job has progress=0.0."""
        training_service = MagicMock(spec=TrainingService)
        training_service.get_job_status.return_value = {
            "status": "FAILED",
            "error": "Training failed",
            "submitted_at": datetime(2023, 1, 1),
        }
        
        response = asyncio.run(get_training_job_status("job123", user="test_user", training_service=training_service))
        
        assert response.status == "FAILED"
        assert response.progress == 0.0
        assert response.error == "Training failed"

    def test_get_training_job_status_completed_without_scores(self):
        """Test completed job without scores."""
        training_service = MagicMock(spec=TrainingService)
        training_service.get_job_status.return_value = {
            "status": "COMPLETED",
            "run_id": "run123",
            "scores": None,
            "submitted_at": datetime(2023, 1, 1),
            "completed_at": datetime(2023, 1, 2),
        }
        
        response = asyncio.run(get_training_job_status("job123", user="test_user", training_service=training_service))
        
        assert response.result is not None
        assert response.result.cv_mean_score is None


class TestListTrainingJobs:
    """Tests for list_training_jobs endpoint."""

    def test_list_training_jobs_filtering_by_status(self):
        """Test filtering by status."""
        training_service = MagicMock(spec=TrainingService)
        training_service._jobs = {
            "job1": {
                "status": "COMPLETED",
                "submitted_at": datetime(2023, 1, 1),
                "completed_at": datetime(2023, 1, 2),
                "run_id": "run1",
            },
            "job2": {
                "status": "RUNNING",
                "submitted_at": datetime(2023, 1, 2),
            },
            "job3": {
                "status": "COMPLETED",
                "submitted_at": datetime(2023, 1, 3),
                "completed_at": datetime(2023, 1, 4),
                "run_id": "run3",
            },
        }
        
        response = asyncio.run(list_training_jobs(
            limit=50,
            offset=0,
            status="COMPLETED",
            user="test_user",
            training_service=training_service,
        ))
        
        assert response.status == "success"
        assert len(response.data["jobs"]) == 2
        assert all(job["status"] == "COMPLETED" for job in response.data["jobs"])

    def test_list_training_jobs_pagination(self):
        """Test pagination."""
        training_service = MagicMock(spec=TrainingService)
        training_service._jobs = {
            f"job{i}": {
                "status": "COMPLETED",
                "submitted_at": datetime(2023, 1, i),
                "completed_at": datetime(2023, 1, i + 1),
            }
            for i in range(1, 11)
        }
        
        response = asyncio.run(list_training_jobs(
            limit=3,
            offset=2,
            status=None,
            user="test_user",
            training_service=training_service,
        ))
        
        assert response.status == "success"
        assert len(response.data["jobs"]) == 3
        assert response.data["pagination"]["total"] == 10
        assert response.data["pagination"]["limit"] == 3
        assert response.data["pagination"]["offset"] == 2
        assert response.data["pagination"]["has_more"] is True

    def test_list_training_jobs_pagination_no_more(self):
        """Test pagination when has_more is False."""
        training_service = MagicMock(spec=TrainingService)
        training_service._jobs = {
            f"job{i}": {
                "status": "COMPLETED",
                "submitted_at": datetime(2023, 1, i),
            }
            for i in range(1, 6)
        }
        
        response = asyncio.run(list_training_jobs(
            limit=3,
            offset=3,
            status=None,
            user="test_user",
            training_service=training_service,
        ))
        
        assert response.status == "success"
        assert len(response.data["jobs"]) == 2
        assert response.data["pagination"]["has_more"] is False

    def test_list_training_jobs_empty_list(self):
        """Test pagination with empty list."""
        training_service = MagicMock(spec=TrainingService)
        training_service._jobs = {}
        
        response = asyncio.run(list_training_jobs(
            limit=50,
            offset=0,
            status=None,
            user="test_user",
            training_service=training_service,
        ))
        
        assert response.status == "success"
        assert len(response.data["jobs"]) == 0
        assert response.data["pagination"]["total"] == 0
        assert response.data["pagination"]["has_more"] is False

