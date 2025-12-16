"""API client for Streamlit UI to interact with REST API."""

from typing import Any, Dict, List, Optional, cast

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.api.dto.analytics import DataSummaryResponse, FeatureImportanceResponse
from src.api.dto.inference import BatchPredictionResponse, PredictionResponse
from src.api.dto.models import ModelDetailsResponse, ModelMetadata, ModelSchemaResponse
from src.api.dto.training import (
    DataUploadResponse,
    TrainingJobResponse,
    TrainingJobStatusResponse,
    TrainingJobSummary,
)
from src.api.dto.workflow import (
    WorkflowCompleteResponse,
    WorkflowProgressResponse,
    WorkflowStartResponse,
    WorkflowStateResponse,
)
from src.utils.env_loader import get_env_var
from src.utils.logger import get_logger

logger = get_logger(__name__)


class APIError(Exception):
    """Base exception for API client errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize API error.

        Args:
            message (str): Error message.
            status_code (Optional[int]): HTTP status code.
            details (Optional[Dict[str, Any]]): Error details.
        """
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class APIClient:
    """HTTP client for interacting with the REST API."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize API client.

        Args:
            base_url (Optional[str]): API base URL.
            api_key (Optional[str]): API key.
        """
        self.base_url: str = (
            base_url
            or get_env_var("API_BASE_URL", "http://localhost:8000")
            or "http://localhost:8000"
        )
        self.api_key = api_key or get_env_var("API_KEY")
        self.logger = get_logger(__name__)

        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        if self.api_key:
            self.session.headers.update({"X-API-Key": self.api_key})

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request.

        Args:
            method (str): HTTP method.
            endpoint (str): API endpoint.
            **kwargs: Request kwargs.

        Returns:
            Dict[str, Any]: Response JSON.

        Raises:
            APIError: If request fails.
        """
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            json_response = response.json()
            if not isinstance(json_response, dict):
                raise APIError(f"Expected dict response, got {type(json_response)}")
            return json_response
        except requests.exceptions.HTTPError as e:
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {})
                raise APIError(
                    message=error_detail.get("message", str(e)),
                    status_code=e.response.status_code,
                    details=error_detail.get("details", {}),
                ) from e
            except (ValueError, KeyError):
                raise APIError(
                    message=str(e),
                    status_code=e.response.status_code,
                ) from e
        except requests.exceptions.RequestException as e:
            raise APIError(f"Request failed: {str(e)}") from e

    def list_models(
        self, limit: int = 50, offset: int = 0, experiment_name: Optional[str] = None
    ) -> List[ModelMetadata]:
        """List available models.

        Args:
            limit (int): Maximum items.
            offset (int): Items to skip.
            experiment_name (Optional[str]): Filter by experiment.

        Returns:
            List[ModelMetadata]: List of models.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if experiment_name:
            params["experiment_name"] = experiment_name

        response = self._request("GET", "/api/v1/models", params=params)
        models_data = response.get("data", {}).get("models", [])
        return [cast(ModelMetadata, ModelMetadata.model_validate(m)) for m in models_data]

    def get_model_details(self, run_id: str) -> ModelDetailsResponse:
        """Get model details.

        Args:
            run_id (str): MLflow run ID.

        Returns:
            ModelDetailsResponse: Model details.
        """
        response = self._request("GET", f"/api/v1/models/{run_id}")
        return cast(ModelDetailsResponse, ModelDetailsResponse.model_validate(response))

    def get_model_schema(self, run_id: str) -> ModelSchemaResponse:
        """Get model schema.

        Args:
            run_id (str): MLflow run ID.

        Returns:
            ModelSchemaResponse: Model schema.
        """
        response = self._request("GET", f"/api/v1/models/{run_id}/schema")
        return cast(ModelSchemaResponse, ModelSchemaResponse.model_validate(response))

    def predict(self, run_id: str, features: Dict[str, Any]) -> PredictionResponse:
        """Predict salary quantiles.

        Args:
            run_id (str): MLflow run ID.
            features (Dict[str, Any]): Input features.

        Returns:
            PredictionResponse: Prediction results.
        """
        response = self._request(
            "POST",
            f"/api/v1/models/{run_id}/predict",
            json={"features": features},
        )
        return cast(PredictionResponse, PredictionResponse.model_validate(response))

    def predict_batch(
        self, run_id: str, features_list: List[Dict[str, Any]]
    ) -> BatchPredictionResponse:
        """Batch predict salary quantiles.

        Args:
            run_id (str): MLflow run ID.
            features_list (List[Dict[str, Any]]): List of feature dictionaries.

        Returns:
            BatchPredictionResponse: Batch prediction results.
        """
        response = self._request(
            "POST",
            f"/api/v1/models/{run_id}/predict/batch",
            json={"features": features_list},
        )
        return cast(BatchPredictionResponse, BatchPredictionResponse.model_validate(response))

    def upload_training_data(
        self, file_content: bytes, filename: str, dataset_name: Optional[str] = None
    ) -> DataUploadResponse:
        """Upload training data CSV.

        Args:
            file_content (bytes): File content.
            filename (str): Filename.
            dataset_name (Optional[str]): Dataset name.

        Returns:
            DataUploadResponse: Upload response.
        """
        files = {"file": (filename, file_content, "text/csv")}
        data = {}
        if dataset_name:
            data["dataset_name"] = dataset_name

        response = self._request("POST", "/api/v1/training/data/upload", files=files, data=data)
        return cast(DataUploadResponse, DataUploadResponse.model_validate(response))

    def start_training(
        self,
        dataset_id: str,
        config: Dict[str, Any],
        remove_outliers: bool = True,
        do_tune: bool = False,
        n_trials: Optional[int] = None,
        additional_tag: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> TrainingJobResponse:
        """Start training job.

        Args:
            dataset_id (str): Dataset ID.
            config (Dict[str, Any]): Model config.
            remove_outliers (bool): Remove outliers.
            do_tune (bool): Run tuning.
            n_trials (Optional[int]): Tuning trials.
            additional_tag (Optional[str]): Additional tag.
            dataset_name (Optional[str]): Dataset name.

        Returns:
            TrainingJobResponse: Job response.
        """
        request_data = {
            "dataset_id": dataset_id,
            "config": config,
            "remove_outliers": remove_outliers,
            "do_tune": do_tune,
        }
        if n_trials:
            request_data["n_trials"] = n_trials
        if additional_tag:
            request_data["additional_tag"] = additional_tag
        if dataset_name:
            request_data["dataset_name"] = dataset_name

        response = self._request("POST", "/api/v1/training/jobs", json=request_data)
        return cast(TrainingJobResponse, TrainingJobResponse.model_validate(response))

    def get_training_job_status(self, job_id: str) -> TrainingJobStatusResponse:
        """Get training job status.

        Args:
            job_id (str): Job ID.

        Returns:
            TrainingJobStatusResponse: Job status.
        """
        response = self._request("GET", f"/api/v1/training/jobs/{job_id}")
        return cast(TrainingJobStatusResponse, TrainingJobStatusResponse.model_validate(response))

    def list_training_jobs(
        self, limit: int = 50, offset: int = 0, status: Optional[str] = None
    ) -> List[TrainingJobSummary]:
        """List training jobs.

        Args:
            limit (int): Maximum items.
            offset (int): Items to skip.
            status (Optional[str]): Status filter.

        Returns:
            List[TrainingJobSummary]: List of jobs.
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status

        response = self._request("GET", "/api/v1/training/jobs", params=params)
        jobs_data = response.get("data", {}).get("jobs", [])
        return [cast(TrainingJobSummary, TrainingJobSummary.model_validate(j)) for j in jobs_data]

    def start_workflow(
        self,
        df: pd.DataFrame,
        provider: str = "openai",
        preset: Optional[str] = None,
    ) -> WorkflowStartResponse:
        """Start configuration workflow.

        Args:
            df (pd.DataFrame): Input data.
            provider (str): LLM provider.
            preset (Optional[str]): Preset prompt.

        Returns:
            WorkflowStartResponse: Workflow start response.
        """
        sample_df = df.head(50)
        df_json = sample_df.to_json(orient="records", date_format="iso")

        request_data = {
            "data": df_json,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "dataset_size": len(df),
            "provider": provider,
        }
        if preset:
            request_data["preset"] = preset

        response = self._request("POST", "/api/v1/workflow/start", json=request_data)
        return cast(WorkflowStartResponse, WorkflowStartResponse.model_validate(response))

    def get_workflow_state(self, workflow_id: str) -> WorkflowStateResponse:
        """Get workflow state.

        Args:
            workflow_id (str): Workflow ID.

        Returns:
            WorkflowStateResponse: Workflow state.
        """
        response = self._request("GET", f"/api/v1/workflow/{workflow_id}")
        return cast(WorkflowStateResponse, WorkflowStateResponse.model_validate(response))

    def confirm_classification(
        self, workflow_id: str, modifications: Dict[str, Any]
    ) -> WorkflowProgressResponse:
        """Confirm classification phase.

        Args:
            workflow_id (str): Workflow ID.
            modifications (Dict[str, Any]): Classification modifications.

        Returns:
            WorkflowProgressResponse: Progress response.
        """
        from src.api.dto.workflow import (
            ClassificationConfirmationRequest,
            ClassificationModifications,
        )

        request_data = ClassificationConfirmationRequest(
            modifications=ClassificationModifications(
                targets=modifications.get("targets", []),
                features=modifications.get("features", []),
                ignore=modifications.get("ignore", []),
            )
        )

        response = self._request(
            "POST",
            f"/api/v1/workflow/{workflow_id}/confirm/classification",
            json=request_data.model_dump(),
        )
        return cast(WorkflowProgressResponse, WorkflowProgressResponse.model_validate(response))

    def confirm_encoding(
        self, workflow_id: str, modifications: Dict[str, Any]
    ) -> WorkflowProgressResponse:
        """Confirm encoding phase.

        Args:
            workflow_id (str): Workflow ID.
            modifications (Dict[str, Any]): Encoding modifications.

        Returns:
            WorkflowProgressResponse: Progress response.
        """
        from src.api.dto.workflow import EncodingConfirmationRequest, EncodingModifications

        encodings = {}
        for col, enc_config in modifications.get("encodings", {}).items():
            from src.api.dto.workflow import EncodingConfig

            encodings[col] = EncodingConfig(
                type=enc_config.get("type"),
                mapping=enc_config.get("mapping"),
                reasoning=enc_config.get("reasoning"),
            )

        optional_encodings = {}
        for col, opt_enc_config in modifications.get("optional_encodings", {}).items():
            from src.api.dto.workflow import OptionalEncodingConfig

            optional_encodings[col] = OptionalEncodingConfig(
                type=opt_enc_config.get("type"),
                params=opt_enc_config.get("params", {}),
            )

        request_data = EncodingConfirmationRequest(
            modifications=EncodingModifications(
                encodings=encodings,
                optional_encodings=optional_encodings,
            )
        )

        response = self._request(
            "POST",
            f"/api/v1/workflow/{workflow_id}/confirm/encoding",
            json=request_data.model_dump(),
        )
        return cast(WorkflowProgressResponse, WorkflowProgressResponse.model_validate(response))

    def finalize_configuration(
        self, workflow_id: str, config_updates: Dict[str, Any]
    ) -> WorkflowCompleteResponse:
        """Finalize configuration.

        Args:
            workflow_id (str): Workflow ID.
            config_updates (Dict[str, Any]): Configuration updates.

        Returns:
            WorkflowCompleteResponse: Complete response.
        """
        from src.api.dto.workflow import (
            ConfigurationFinalizationRequest,
            FeatureConfig,
            Hyperparameters,
        )

        features = [
            FeatureConfig(name=f["name"], monotone_constraint=f["monotone_constraint"])
            for f in config_updates.get("features", [])
        ]

        request_data = ConfigurationFinalizationRequest(
            features=features,
            quantiles=config_updates.get("quantiles", []),
            hyperparameters=Hyperparameters(
                training=config_updates.get("hyperparameters", {}).get("training", {}),
                cv=config_updates.get("hyperparameters", {}).get("cv", {}),
            ),
            location_settings=config_updates.get("location_settings"),
        )

        response = self._request(
            "POST",
            f"/api/v1/workflow/{workflow_id}/finalize",
            json=request_data.model_dump(),
        )
        return cast(WorkflowCompleteResponse, WorkflowCompleteResponse.model_validate(response))

    def get_data_summary(self, df: pd.DataFrame) -> DataSummaryResponse:
        """Get data summary.

        Args:
            df (pd.DataFrame): Input data.

        Returns:
            DataSummaryResponse: Data summary.
        """
        df_json = df.to_json(orient="records", date_format="iso")
        response = self._request("POST", "/api/v1/analytics/data-summary", json={"data": df_json})
        return cast(DataSummaryResponse, DataSummaryResponse.model_validate(response))

    def get_feature_importance(
        self, run_id: str, target: str, quantile: float
    ) -> FeatureImportanceResponse:
        """Get feature importance.

        Args:
            run_id (str): MLflow run ID.
            target (str): Target column.
            quantile (float): Quantile value.

        Returns:
            FeatureImportanceResponse: Feature importance.
        """
        params = {"target": target, "quantile": quantile}
        response = self._request(
            "GET",
            f"/api/v1/models/{run_id}/analytics/feature-importance",
            params=params,
        )
        return cast(FeatureImportanceResponse, FeatureImportanceResponse.model_validate(response))


def get_api_client() -> Optional[APIClient]:
    """Get API client instance if API is enabled.

    Returns:
        Optional[APIClient]: API client or None if disabled.
    """
    use_api_val = get_env_var("USE_API", "false") or "false"
    use_api = use_api_val.lower() in ("true", "1", "yes")
    if not use_api:
        return None
    return APIClient()
