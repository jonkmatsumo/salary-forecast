"""Unit tests for API client."""

from io import StringIO
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
from requests.exceptions import HTTPError, RequestException

from src.app.api_client import APIClient, APIError, get_api_client


class TestAPIError:
    """Tests for APIError class."""

    def test_api_error_with_message_only(self):
        """Test APIError initialization with message only."""
        error = APIError("Test error message")
        assert error.message == "Test error message"
        assert error.status_code is None
        assert error.details == {}
        assert str(error) == "Test error message"

    def test_api_error_with_status_code(self):
        """Test APIError initialization with status code."""
        error = APIError("Not found", status_code=404)
        assert error.message == "Not found"
        assert error.status_code == 404
        assert error.details == {}

    def test_api_error_with_details(self):
        """Test APIError initialization with details dict."""
        details = {"field": "value", "code": "ERROR_CODE"}
        error = APIError("Error occurred", status_code=400, details=details)
        assert error.message == "Error occurred"
        assert error.status_code == 400
        assert error.details == details


class TestAPIClientInit:
    """Tests for APIClient.__init__."""

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_init_with_provided_params(self, mock_get_logger, mock_get_env_var):
        """Test initialization with provided base_url and api_key."""
        mock_get_logger.return_value = MagicMock()

        client = APIClient(base_url="https://api.example.com", api_key="test_key")

        assert client.base_url == "https://api.example.com"
        assert client.api_key == "test_key"
        assert client.session.headers.get("X-API-Key") == "test_key"
        mock_get_env_var.assert_not_called()

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_init_without_params_uses_env_vars(self, mock_get_logger, mock_get_env_var):
        """Test initialization without parameters uses environment variables."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: {
            "API_BASE_URL": "https://default.com",
            "API_KEY": "env_key",
        }.get(key, default)

        client = APIClient()

        assert client.base_url == "https://default.com"
        assert client.api_key == "env_key"
        assert client.session.headers.get("X-API-Key") == "env_key"

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_init_without_api_key(self, mock_get_logger, mock_get_env_var):
        """Test initialization without API key."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: {
            "API_BASE_URL": "https://default.com"
        }.get(key, default)

        client = APIClient()

        assert client.base_url == "https://default.com"
        assert client.api_key is None
        assert "X-API-Key" not in client.session.headers

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_retry_strategy_configuration(self, mock_get_logger, mock_get_env_var):
        """Test retry strategy is configured correctly."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()

        assert "http://" in client.session.adapters
        assert "https://" in client.session.adapters
        http_adapter = client.session.adapters["http://"]
        assert isinstance(http_adapter, requests.adapters.HTTPAdapter)
        assert http_adapter.max_retries.total == 3


class TestAPIClientRequest:
    """Tests for APIClient._request method."""

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_request_success(self, mock_get_logger, mock_get_env_var):
        """Test successful GET request."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient(base_url="https://api.test.com")
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "success", "data": {"key": "value"}}
        mock_response.raise_for_status.return_value = None

        client.session.request = MagicMock(return_value=mock_response)

        result = client._request("GET", "/api/test")

        assert result == {"status": "success", "data": {"key": "value"}}
        client.session.request.assert_called_once()
        call_args = client.session.request.call_args
        assert call_args[0] == ("GET", "https://api.test.com/api/test")

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_request_httperror_with_json_response(self, mock_get_logger, mock_get_env_var):
        """Test HTTPError with JSON error response."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            "error": {"message": "Model not found", "details": {"run_id": "test123"}}
        }

        http_error = HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error
        client.session.request = MagicMock(return_value=mock_response)

        with pytest.raises(APIError) as exc_info:
            client._request("GET", "/api/test")

        assert exc_info.value.message == "Model not found"
        assert exc_info.value.status_code == 404
        assert exc_info.value.details == {"run_id": "test123"}
        assert exc_info.value.__cause__ == http_error

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_request_httperror_with_non_json_response(self, mock_get_logger, mock_get_env_var):
        """Test HTTPError with non-JSON error response."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.side_effect = ValueError("Not JSON")

        http_error = HTTPError("Server error")
        http_error.response = mock_response
        mock_response.raise_for_status.side_effect = http_error
        client.session.request = MagicMock(return_value=mock_response)

        with pytest.raises(APIError) as exc_info:
            client._request("GET", "/api/test")

        assert "Server error" in exc_info.value.message or str(http_error) in exc_info.value.message
        assert exc_info.value.status_code == 500

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_request_requestexception(self, mock_get_logger, mock_get_env_var):
        """Test RequestException handling."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        request_exception = RequestException("Connection failed")
        client.session.request = MagicMock(side_effect=request_exception)

        with pytest.raises(APIError) as exc_info:
            client._request("GET", "/api/test")

        assert "Request failed" in exc_info.value.message
        assert "Connection failed" in exc_info.value.message
        assert exc_info.value.__cause__ == request_exception

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_request_url_normalization(self, mock_get_logger, mock_get_env_var):
        """Test URL normalization in _request."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient(base_url="https://api.test.com/")
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None

        client.session.request = MagicMock(return_value=mock_response)

        client._request("GET", "/api/test")

        call_args = client.session.request.call_args
        assert call_args[0][1] == "https://api.test.com/api/test"


class TestAPIClientListModels:
    """Tests for APIClient.list_models."""

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_list_models_success(self, mock_get_logger, mock_get_env_var):
        """Test successful list_models call."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        mock_response_data = {
            "data": {
                "models": [
                    {
                        "run_id": "run1",
                        "start_time": "2023-01-01T00:00:00",
                        "model_type": "XGBoost",
                        "cv_mean_score": 0.95,
                        "dataset_name": "test_dataset",
                    }
                ]
            }
        }
        client._request = MagicMock(return_value=mock_response_data)

        models = client.list_models()

        assert len(models) == 1
        assert models[0].run_id == "run1"
        client._request.assert_called_once_with(
            "GET", "/api/v1/models", params={"limit": 50, "offset": 0}
        )

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_list_models_with_experiment_filter(self, mock_get_logger, mock_get_env_var):
        """Test list_models with experiment_name filter."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        mock_response_data = {"data": {"models": []}}
        client._request = MagicMock(return_value=mock_response_data)

        client.list_models(limit=10, offset=5, experiment_name="exp1")

        client._request.assert_called_once_with(
            "GET", "/api/v1/models", params={"limit": 10, "offset": 5, "experiment_name": "exp1"}
        )


class TestAPIClientGetModelDetails:
    """Tests for APIClient.get_model_details."""

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_get_model_details_success(self, mock_get_logger, mock_get_env_var):
        """Test successful get_model_details call."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        mock_response_data = {
            "run_id": "run1",
            "metadata": {
                "run_id": "run1",
                "start_time": "2023-01-01T00:00:00",
                "model_type": "XGBoost",
                "dataset_name": "test",
            },
            "schema": {
                "ranked_features": [],
                "proximity_features": [],
                "numerical_features": ["feat1"],
            },
            "feature_names": ["feat1"],
            "targets": ["target1"],
            "quantiles": [0.5],
        }
        client._request = MagicMock(return_value=mock_response_data)

        result = client.get_model_details("run1")

        assert result.run_id == "run1"
        client._request.assert_called_once_with("GET", "/api/v1/models/run1")


class TestAPIClientGetModelSchema:
    """Tests for APIClient.get_model_schema."""

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_get_model_schema_success(self, mock_get_logger, mock_get_env_var):
        """Test successful get_model_schema call."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        mock_response_data = {
            "run_id": "run1",
            "schema": {
                "ranked_features": [],
                "proximity_features": [],
                "numerical_features": ["feat1"],
            },
        }
        client._request = MagicMock(return_value=mock_response_data)

        result = client.get_model_schema("run1")

        assert result.run_id == "run1"
        client._request.assert_called_once_with("GET", "/api/v1/models/run1/schema")


class TestAPIClientPredict:
    """Tests for APIClient.predict."""

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_predict_success(self, mock_get_logger, mock_get_env_var):
        """Test successful predict call."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        features = {"feature1": "value1", "feature2": 123}
        mock_response_data = {
            "predictions": {"target1": {"p50": 100000.0}},
            "metadata": {
                "model_run_id": "run1",
                "prediction_timestamp": "2023-01-01T00:00:00",
            },
        }
        client._request = MagicMock(return_value=mock_response_data)

        result = client.predict("run1", features)

        assert result.predictions == {"target1": {"p50": 100000.0}}
        client._request.assert_called_once_with(
            "POST", "/api/v1/models/run1/predict", json={"features": features}
        )


class TestAPIClientPredictBatch:
    """Tests for APIClient.predict_batch."""

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_predict_batch_success(self, mock_get_logger, mock_get_env_var):
        """Test successful predict_batch call."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        features_list = [
            {"feature1": "value1"},
            {"feature1": "value2"},
        ]
        mock_response_data = {
            "predictions": [
                {
                    "predictions": {"target1": {"p50": 100000.0}},
                    "metadata": {
                        "model_run_id": "run1",
                        "prediction_timestamp": "2023-01-01T00:00:00",
                    },
                },
                {
                    "predictions": {"target1": {"p50": 110000.0}},
                    "metadata": {
                        "model_run_id": "run1",
                        "prediction_timestamp": "2023-01-01T00:00:00",
                    },
                },
            ],
            "total": 2,
        }
        client._request = MagicMock(return_value=mock_response_data)

        result = client.predict_batch("run1", features_list)

        assert len(result.predictions) == 2
        assert result.total == 2
        client._request.assert_called_once_with(
            "POST", "/api/v1/models/run1/predict/batch", json={"features": features_list}
        )


class TestAPIClientUploadTrainingData:
    """Tests for APIClient.upload_training_data."""

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_upload_training_data_with_dataset_name(self, mock_get_logger, mock_get_env_var):
        """Test upload_training_data with dataset_name."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        file_content = b"col1,col2\n1,2\n3,4"
        mock_response_data = {
            "dataset_id": "dataset1",
            "row_count": 2,
            "column_count": 2,
            "summary": {
                "total_samples": 2,
                "shape": [2, 2],
                "unique_counts": {},
            },
        }
        client._request = MagicMock(return_value=mock_response_data)

        result = client.upload_training_data(file_content, "test.csv", "my_dataset")

        assert result.dataset_id == "dataset1"
        call_kwargs = client._request.call_args[1]
        assert "files" in call_kwargs
        assert "data" in call_kwargs
        assert call_kwargs["data"]["dataset_name"] == "my_dataset"

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_upload_training_data_without_dataset_name(self, mock_get_logger, mock_get_env_var):
        """Test upload_training_data without dataset_name."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        file_content = b"col1,col2\n1,2"
        mock_response_data = {
            "dataset_id": "dataset1",
            "row_count": 1,
            "column_count": 2,
            "summary": {"total_samples": 1, "shape": [1, 2], "unique_counts": {}},
        }
        client._request = MagicMock(return_value=mock_response_data)

        result = client.upload_training_data(file_content, "test.csv")

        assert result.dataset_id == "dataset1"
        call_kwargs = client._request.call_args[1]
        assert "dataset_name" not in call_kwargs["data"]


class TestAPIClientStartTraining:
    """Tests for APIClient.start_training."""

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_start_training_with_all_params(self, mock_get_logger, mock_get_env_var):
        """Test start_training with all optional parameters."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        config = {"model": {"targets": ["target1"]}}
        mock_response_data = {"job_id": "job1", "status": "QUEUED"}
        client._request = MagicMock(return_value=mock_response_data)

        result = client.start_training(
            dataset_id="dataset1",
            config=config,
            remove_outliers=False,
            do_tune=True,
            n_trials=50,
            additional_tag="test_tag",
            dataset_name="test_dataset",
        )

        assert result.job_id == "job1"
        call_kwargs = client._request.call_args[1]
        request_data = call_kwargs["json"]
        assert request_data["dataset_id"] == "dataset1"
        assert request_data["n_trials"] == 50
        assert request_data["additional_tag"] == "test_tag"
        assert request_data["dataset_name"] == "test_dataset"

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_start_training_minimal_params(self, mock_get_logger, mock_get_env_var):
        """Test start_training with minimal parameters."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        config = {"model": {"targets": ["target1"]}}
        mock_response_data = {"job_id": "job1", "status": "QUEUED"}
        client._request = MagicMock(return_value=mock_response_data)

        result = client.start_training(dataset_id="dataset1", config=config)

        assert result.job_id == "job1"
        call_kwargs = client._request.call_args[1]
        request_data = call_kwargs["json"]
        assert "n_trials" not in request_data
        assert "additional_tag" not in request_data
        assert "dataset_name" not in request_data


class TestAPIClientGetTrainingJobStatus:
    """Tests for APIClient.get_training_job_status."""

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_get_training_job_status_success(self, mock_get_logger, mock_get_env_var):
        """Test successful get_training_job_status call."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        mock_response_data = {
            "job_id": "job1",
            "status": "COMPLETED",
            "progress": 1.0,
            "run_id": "run1",
        }
        client._request = MagicMock(return_value=mock_response_data)

        result = client.get_training_job_status("job1")

        assert result.job_id == "job1"
        assert result.status == "COMPLETED"
        client._request.assert_called_once_with("GET", "/api/v1/training/jobs/job1")


class TestAPIClientListTrainingJobs:
    """Tests for APIClient.list_training_jobs."""

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_list_training_jobs_with_status_filter(self, mock_get_logger, mock_get_env_var):
        """Test list_training_jobs with status filter."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        mock_response_data = {
            "data": {
                "jobs": [
                    {
                        "job_id": "job1",
                        "status": "COMPLETED",
                        "submitted_at": "2023-01-01T00:00:00",
                    },
                ]
            }
        }
        client._request = MagicMock(return_value=mock_response_data)

        jobs = client.list_training_jobs(limit=10, offset=5, status="COMPLETED")

        assert len(jobs) == 1
        client._request.assert_called_once_with(
            "GET", "/api/v1/training/jobs", params={"limit": 10, "offset": 5, "status": "COMPLETED"}
        )

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_list_training_jobs_with_pagination(self, mock_get_logger, mock_get_env_var):
        """Test list_training_jobs with pagination."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        mock_response_data = {"data": {"jobs": []}}
        client._request = MagicMock(return_value=mock_response_data)

        jobs = client.list_training_jobs(limit=20, offset=10)

        assert len(jobs) == 0
        client._request.assert_called_once_with(
            "GET", "/api/v1/training/jobs", params={"limit": 20, "offset": 10}
        )


class TestAPIClientStartWorkflow:
    """Tests for APIClient.start_workflow."""

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_start_workflow_with_preset(self, mock_get_logger, mock_get_env_var):
        """Test start_workflow with preset."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        mock_response_data = {
            "workflow_id": "workflow1",
            "phase": "classification",
            "state": {"phase": "classification", "status": "success"},
        }
        client._request = MagicMock(return_value=mock_response_data)

        result = client.start_workflow(df, provider="gemini", preset="salary")

        assert result.workflow_id == "workflow1"
        call_kwargs = client._request.call_args[1]
        request_data = call_kwargs["json"]
        assert request_data["preset"] == "salary"
        assert request_data["provider"] == "gemini"
        assert "data" in request_data

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_start_workflow_without_preset(self, mock_get_logger, mock_get_env_var):
        """Test start_workflow without preset."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        df = pd.DataFrame({"col1": [1, 2]})
        mock_response_data = {
            "workflow_id": "workflow1",
            "phase": "classification",
            "state": {"phase": "classification", "status": "success"},
        }
        client._request = MagicMock(return_value=mock_response_data)

        result = client.start_workflow(df)

        assert result.workflow_id == "workflow1"
        call_kwargs = client._request.call_args[1]
        request_data = call_kwargs["json"]
        assert "preset" not in request_data

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_start_workflow_dataframe_serialization(self, mock_get_logger, mock_get_env_var):
        """Test DataFrame serialization in start_workflow."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        df = pd.DataFrame({"col1": [1, 2, 3, 4, 5] * 20})  # 100 rows
        mock_response_data = {
            "workflow_id": "workflow1",
            "phase": "classification",
            "state": {"phase": "classification", "status": "success"},
        }
        client._request = MagicMock(return_value=mock_response_data)

        client.start_workflow(df)

        call_kwargs = client._request.call_args[1]
        request_data = call_kwargs["json"]
        assert request_data["dataset_size"] == 100
        assert len(pd.read_json(StringIO(request_data["data"]), orient="records")) == 50


class TestAPIClientGetWorkflowState:
    """Tests for APIClient.get_workflow_state."""

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_get_workflow_state_success(self, mock_get_logger, mock_get_env_var):
        """Test successful get_workflow_state call."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        mock_response_data = {
            "workflow_id": "workflow1",
            "phase": "encoding",
            "state": {},
            "current_result": {},
        }
        client._request = MagicMock(return_value=mock_response_data)

        result = client.get_workflow_state("workflow1")

        assert result.workflow_id == "workflow1"
        client._request.assert_called_once_with("GET", "/api/v1/workflow/workflow1")


class TestAPIClientConfirmClassification:
    """Tests for APIClient.confirm_classification."""

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_confirm_classification_modifications(self, mock_get_logger, mock_get_env_var):
        """Test confirm_classification modifications dict construction."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        modifications = {
            "targets": ["target1"],
            "features": ["feature1"],
            "ignore": ["ignore1"],
        }
        mock_response_data = {
            "workflow_id": "workflow1",
            "phase": "encoding",
            "result": {},
        }
        client._request = MagicMock(return_value=mock_response_data)

        result = client.confirm_classification("workflow1", modifications)

        assert result.phase == "encoding"
        call_kwargs = client._request.call_args[1]
        request_data = call_kwargs["json"]
        assert request_data["modifications"]["targets"] == ["target1"]
        assert request_data["modifications"]["features"] == ["feature1"]
        assert request_data["modifications"]["ignore"] == ["ignore1"]


class TestAPIClientConfirmEncoding:
    """Tests for APIClient.confirm_encoding."""

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_confirm_encoding_complex_modifications(self, mock_get_logger, mock_get_env_var):
        """Test confirm_encoding with complex encoding modifications."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        modifications = {
            "encodings": {
                "col1": {
                    "type": "ordinal",
                    "mapping": {"low": 0, "high": 1},
                    "reasoning": "Test reasoning",
                }
            },
            "optional_encodings": {
                "col2": {
                    "type": "cost_of_living",
                    "params": {},
                }
            },
        }
        mock_response_data = {
            "workflow_id": "workflow1",
            "phase": "configuration",
            "result": {},
        }
        client._request = MagicMock(return_value=mock_response_data)

        result = client.confirm_encoding("workflow1", modifications)

        assert result.phase == "configuration"
        call_kwargs = client._request.call_args[1]
        request_data = call_kwargs["json"]
        assert "encodings" in request_data["modifications"]
        assert "optional_encodings" in request_data["modifications"]


class TestAPIClientFinalizeConfiguration:
    """Tests for APIClient.finalize_configuration."""

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_finalize_configuration_with_location_settings(self, mock_get_logger, mock_get_env_var):
        """Test finalize_configuration with location_settings."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        config_updates = {
            "features": [{"name": "feat1", "monotone_constraint": 1}],
            "quantiles": [0.1, 0.5, 0.9],
            "hyperparameters": {
                "training": {"max_depth": 6},
                "cv": {"nfold": 5},
            },
            "location_settings": {"max_distance_km": 50},
        }
        mock_response_data = {
            "workflow_id": "workflow1",
            "phase": "complete",
            "final_config": {"model": {"targets": ["target1"]}},
        }
        client._request = MagicMock(return_value=mock_response_data)

        result = client.finalize_configuration("workflow1", config_updates)

        assert result.phase == "complete"
        call_kwargs = client._request.call_args[1]
        request_data = call_kwargs["json"]
        assert len(request_data["features"]) == 1
        assert request_data["location_settings"] == {"max_distance_km": 50}

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_finalize_configuration_without_location_settings(
        self, mock_get_logger, mock_get_env_var
    ):
        """Test finalize_configuration without location_settings."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        config_updates = {
            "features": [{"name": "feat1", "monotone_constraint": 1}],
            "quantiles": [0.5],
            "hyperparameters": {"training": {}, "cv": {}},
        }
        mock_response_data = {
            "workflow_id": "workflow1",
            "phase": "complete",
            "final_config": {"model": {"targets": ["target1"]}},
        }
        client._request = MagicMock(return_value=mock_response_data)

        result = client.finalize_configuration("workflow1", config_updates)

        assert result.phase == "complete"
        call_kwargs = client._request.call_args[1]
        request_data = call_kwargs["json"]
        assert request_data["location_settings"] is None


class TestAPIClientGetDataSummary:
    """Tests for APIClient.get_data_summary."""

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_get_data_summary_dataframe_serialization(self, mock_get_logger, mock_get_env_var):
        """Test DataFrame serialization in get_data_summary."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        mock_response_data = {
            "total_samples": 3,
            "shape": [3, 2],
            "unique_counts": {},
        }
        client._request = MagicMock(return_value=mock_response_data)

        result = client.get_data_summary(df)

        assert result.total_samples == 3
        call_kwargs = client._request.call_args[1]
        assert "data" in call_kwargs["json"]
        json_data = call_kwargs["json"]["data"]
        parsed_df = pd.read_json(StringIO(json_data), orient="records")
        assert len(parsed_df) == 3


class TestAPIClientGetFeatureImportance:
    """Tests for APIClient.get_feature_importance."""

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.get_logger")
    def test_get_feature_importance_query_params(self, mock_get_logger, mock_get_env_var):
        """Test query parameter construction in get_feature_importance."""
        mock_get_logger.return_value = MagicMock()
        mock_get_env_var.side_effect = lambda key, default=None: default or None

        client = APIClient()
        mock_response_data = {
            "features": [
                {"name": "feat1", "gain": 0.5},
                {"name": "feat2", "gain": 0.3},
            ]
        }
        client._request = MagicMock(return_value=mock_response_data)

        result = client.get_feature_importance("run1", "target1", 0.5)

        assert len(result.features) == 2
        client._request.assert_called_once_with(
            "GET",
            "/api/v1/models/run1/analytics/feature-importance",
            params={"target": "target1", "quantile": 0.5},
        )


class TestGetAPIClient:
    """Tests for get_api_client function."""

    @patch("src.app.api_client.get_env_var")
    def test_get_api_client_returns_none_when_disabled(self, mock_get_env_var):
        """Test get_api_client returns None when USE_API is false."""
        mock_get_env_var.side_effect = lambda key, default=None: {"USE_API": "false"}.get(
            key, default
        )

        result = get_api_client()

        assert result is None

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.APIClient")
    def test_get_api_client_returns_client_when_enabled(self, mock_api_client, mock_get_env_var):
        """Test get_api_client returns APIClient when USE_API is true."""
        mock_get_env_var.side_effect = lambda key, default=None: {"USE_API": "true"}.get(
            key, default
        )
        mock_instance = MagicMock()
        mock_api_client.return_value = mock_instance

        result = get_api_client()

        assert result == mock_instance
        mock_api_client.assert_called_once()

    @patch("src.app.api_client.get_env_var")
    @patch("src.app.api_client.APIClient")
    def test_get_api_client_various_true_values(self, mock_api_client, mock_get_env_var):
        """Test get_api_client recognizes various true values."""
        for use_api_value in ["1", "yes", "TRUE"]:
            mock_get_env_var.reset_mock()
            mock_api_client.reset_mock()
            mock_get_env_var.side_effect = lambda key, default=None: {"USE_API": use_api_value}.get(
                key, default
            )
            mock_instance = MagicMock()
            mock_api_client.return_value = mock_instance

            result = get_api_client()

            assert result == mock_instance
