"""Unit tests for MCP tool handlers."""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.api.exceptions import InvalidInputError as APIInvalidInputError
from src.api.exceptions import ModelNotFoundError as APIModelNotFoundError
from src.api.mcp.handlers import MCPToolHandler
from src.services.inference_service import InvalidInputError, ModelNotFoundError


@pytest.fixture
def handler():
    """Create MCPToolHandler instance. Returns: MCPToolHandler: Handler instance."""
    return MCPToolHandler()


@pytest.fixture
def mock_model():
    """Create mock model. Returns: MagicMock: Mock model."""
    model = MagicMock()
    model.targets = ["BaseSalary", "TotalComp"]
    model.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    model.feature_names = ["Level", "Location", "YearsOfExperience"]
    model.ranked_encoders = {"Level": MagicMock(mapping={"L3": 0, "L4": 1, "L5": 2})}
    model.proximity_encoders = {"Location": MagicMock()}
    return model


@pytest.fixture
def mock_schema():
    """Create mock schema. Returns: MagicMock: Mock schema."""
    schema = MagicMock()
    schema.ranked_features = ["Level"]
    schema.proximity_features = ["Location"]
    schema.numerical_features = ["YearsOfExperience"]
    schema.all_feature_names = ["Level", "Location", "YearsOfExperience"]
    schema.targets = ["BaseSalary", "TotalComp"]
    schema.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    return schema


def test_handle_list_models(handler):
    """Test _handle_list_models."""
    handler.model_registry.list_models = MagicMock(
        return_value=[
            {
                "run_id": "test123",
                "start_time": datetime(2024, 1, 1, 10, 0, 0),
                "tags.model_type": "XGBoost",
                "metrics.cv_mean_score": 0.85,
                "tags.dataset_name": "test_data",
                "tags.additional_tag": None,
            }
        ]
    )

    result = asyncio.run(handler._handle_list_models({"limit": 10, "offset": 0}))

    assert "models" in result
    assert "total" in result
    assert "limit" in result
    assert "offset" in result
    assert len(result["models"]) == 1
    assert result["models"][0]["run_id"] == "test123"


def test_handle_list_models_with_experiment_filter(handler):
    """Test _handle_list_models with experiment filter."""
    handler.model_registry.list_models = MagicMock(
        return_value=[
            {
                "run_id": "test123",
                "start_time": datetime(2024, 1, 1, 10, 0, 0),
                "tags.model_type": "XGBoost",
                "tags.experiment_name": "experiment1",
                "metrics.cv_mean_score": 0.85,
                "tags.dataset_name": "test_data",
                "tags.additional_tag": None,
            },
            {
                "run_id": "test456",
                "start_time": datetime(2024, 1, 2, 10, 0, 0),
                "tags.model_type": "XGBoost",
                "tags.experiment_name": "experiment2",
                "metrics.cv_mean_score": 0.90,
                "tags.dataset_name": "test_data2",
                "tags.additional_tag": None,
            },
        ]
    )

    result = asyncio.run(
        handler._handle_list_models({"limit": 50, "offset": 0, "experiment_name": "experiment1"})
    )

    assert len(result["models"]) == 1
    assert result["models"][0]["run_id"] == "test123"


def test_handle_get_model_details(handler, mock_model, mock_schema):
    """Test _handle_get_model_details."""
    handler.inference_service.load_model = MagicMock(return_value=mock_model)
    handler.inference_service.get_model_schema = MagicMock(return_value=mock_schema)
    handler.model_registry.list_models = MagicMock(
        return_value=[
            {
                "run_id": "test123",
                "start_time": datetime(2024, 1, 1, 10, 0, 0),
                "tags.model_type": "XGBoost",
                "metrics.cv_mean_score": 0.85,
                "tags.dataset_name": "test_data",
                "tags.additional_tag": None,
            }
        ]
    )

    result = asyncio.run(handler._handle_get_model_details({"run_id": "test123"}))

    assert "run_id" in result
    assert "metadata" in result
    assert "model_schema" in result
    assert result["run_id"] == "test123"
    assert result["metadata"]["run_id"] == "test123"


def test_handle_get_model_details_not_found(handler):
    """Test _handle_get_model_details with non-existent model."""
    handler.inference_service.load_model = MagicMock(side_effect=ModelNotFoundError("Not found"))

    with pytest.raises(APIModelNotFoundError):
        asyncio.run(handler._handle_get_model_details({"run_id": "nonexistent"}))


def test_handle_get_model_schema(handler, mock_model, mock_schema):
    """Test _handle_get_model_schema."""
    handler.inference_service.load_model = MagicMock(return_value=mock_model)
    handler.inference_service.get_model_schema = MagicMock(return_value=mock_schema)

    result = asyncio.run(handler._handle_get_model_schema({"run_id": "test123"}))

    assert "run_id" in result
    assert "model_schema" in result
    assert result["run_id"] == "test123"


def test_handle_get_model_schema_not_found(handler):
    """Test _handle_get_model_schema with non-existent model."""
    handler.inference_service.load_model = MagicMock(side_effect=ModelNotFoundError("Not found"))

    with pytest.raises(APIModelNotFoundError):
        asyncio.run(handler._handle_get_model_schema({"run_id": "nonexistent"}))


def test_handle_predict_salary(handler, mock_model):
    """Test _handle_predict_salary."""
    from src.services.inference_service import PredictionResult

    handler.inference_service.load_model = MagicMock(return_value=mock_model)
    handler.inference_service.predict = MagicMock(
        return_value=PredictionResult(
            predictions={"BaseSalary": {"p10": 150000.0, "p50": 180000.0, "p90": 220000.0}},
            metadata={"model_run_id": "test123"},
        )
    )

    result = asyncio.run(
        handler._handle_predict_salary(
            {"run_id": "test123", "features": {"Level": "L5", "Location": "San Francisco"}}
        )
    )

    assert "predictions" in result
    assert "metadata" in result
    assert "BaseSalary" in result["predictions"]
    assert result["metadata"]["model_run_id"] == "test123"


def test_handle_predict_salary_model_not_found(handler):
    """Test _handle_predict_salary with non-existent model."""
    handler.inference_service.load_model = MagicMock(side_effect=ModelNotFoundError("Not found"))

    with pytest.raises(APIModelNotFoundError):
        asyncio.run(
            handler._handle_predict_salary({"run_id": "nonexistent", "features": {"Level": "L5"}})
        )


def test_handle_predict_salary_invalid_input(handler, mock_model):
    """Test _handle_predict_salary with invalid input."""
    handler.inference_service.load_model = MagicMock(return_value=mock_model)
    handler.inference_service.predict = MagicMock(side_effect=InvalidInputError("Invalid features"))

    with pytest.raises(APIInvalidInputError):
        asyncio.run(handler._handle_predict_salary({"run_id": "test123", "features": {}}))


def test_handle_get_training_status(handler):
    """Test _handle_get_training_status."""
    with patch("src.api.routers.training.get_training_job_status") as mock_get_status:
        from src.api.dto.training import TrainingJobStatusResponse

        mock_get_status.return_value = TrainingJobStatusResponse(
            job_id="job123",
            status="COMPLETED",
            progress=1.0,
            logs=["Training completed"],
            submitted_at=datetime(2024, 1, 1, 10, 0, 0),
            completed_at=datetime(2024, 1, 1, 11, 0, 0),
            result=None,
            error=None,
            run_id="model123",
        )

        result = asyncio.run(handler._handle_get_training_status({"job_id": "job123"}))

        assert "job_id" in result
        assert "status" in result
        assert result["status"] == "COMPLETED"


def test_handle_start_configuration_workflow(handler):
    """Test _handle_start_configuration_workflow."""
    with patch("src.api.routers.workflow.start_workflow") as mock_start:
        from src.api.dto.workflow import WorkflowStartResponse, WorkflowState

        test_df = pd.DataFrame({"Salary": [100000], "Level": ["L3"]})
        df_json = test_df.to_json(orient="records", date_format="iso")

        mock_start.return_value = WorkflowStartResponse(
            workflow_id="workflow123",
            phase="classification",
            state=WorkflowState(
                phase="classification",
                status="success",
                current_result={"targets": ["Salary"], "features": ["Level"]},
            ),
        )

        result = asyncio.run(
            handler._handle_start_configuration_workflow(
                {
                    "data": df_json,
                    "columns": ["Salary", "Level"],
                    "dtypes": {"Salary": "int64", "Level": "object"},
                    "dataset_size": 1,
                    "provider": "openai",
                }
            )
        )

        assert "workflow_id" in result or "phase" in result


def test_handle_start_configuration_workflow_with_preset(handler):
    """Test _handle_start_configuration_workflow with preset."""
    with patch("src.api.routers.workflow.start_workflow") as mock_start:
        from src.api.dto.workflow import WorkflowStartResponse, WorkflowState

        test_df = pd.DataFrame({"Salary": [100000]})
        df_json = test_df.to_json(orient="records", date_format="iso")

        mock_start.return_value = WorkflowStartResponse(
            workflow_id="workflow123",
            phase="classification",
            state=WorkflowState(phase="classification", status="success", current_result={}),
        )

        asyncio.run(
            handler._handle_start_configuration_workflow(
                {
                    "data": df_json,
                    "columns": ["Salary"],
                    "dtypes": {"Salary": "int64"},
                    "dataset_size": 1,
                    "provider": "openai",
                    "preset": "salary",
                }
            )
        )

        mock_start.assert_called_once()
        call_args = mock_start.call_args[0][0]
        assert call_args["preset"] == "salary"


def test_handle_confirm_classification(handler):
    """Test _handle_confirm_classification."""
    with patch("src.api.routers.workflow.confirm_classification") as mock_confirm:
        from src.api.dto.workflow import WorkflowProgressResponse

        mock_confirm.return_value = WorkflowProgressResponse(
            workflow_id="workflow123", phase="encoding", result={}
        )

        result = asyncio.run(
            handler._handle_confirm_classification(
                {
                    "workflow_id": "workflow123",
                    "modifications": {"targets": ["Salary"], "features": ["Level"], "ignore": []},
                }
            )
        )

        assert "workflow_id" in result or "phase" in result


def test_handle_confirm_encoding(handler):
    """Test _handle_confirm_encoding."""
    with patch("src.api.routers.workflow.confirm_encoding") as mock_confirm:
        from src.api.dto.workflow import WorkflowProgressResponse

        mock_confirm.return_value = WorkflowProgressResponse(
            workflow_id="workflow123", phase="configuration", result={}
        )

        result = asyncio.run(
            handler._handle_confirm_encoding(
                {
                    "workflow_id": "workflow123",
                    "modifications": {
                        "encodings": {
                            "Level": {
                                "type": "ordinal",
                                "mapping": {"L3": 0, "L4": 1},
                                "reasoning": "Test",
                            }
                        },
                        "optional_encodings": {},
                    },
                }
            )
        )

        assert "workflow_id" in result or "phase" in result


def test_handle_confirm_encoding_with_optional(handler):
    """Test _handle_confirm_encoding with optional encodings."""
    with patch("src.api.routers.workflow.confirm_encoding") as mock_confirm:
        from src.api.dto.workflow import WorkflowProgressResponse

        mock_confirm.return_value = WorkflowProgressResponse(
            workflow_id="workflow123", phase="configuration", result={}
        )

        asyncio.run(
            handler._handle_confirm_encoding(
                {
                    "workflow_id": "workflow123",
                    "modifications": {
                        "encodings": {},
                        "optional_encodings": {
                            "Location": {"type": "cost_of_living", "params": {}}
                        },
                    },
                }
            )
        )

        mock_confirm.assert_called_once()


def test_handle_finalize_configuration(handler):
    """Test _handle_finalize_configuration."""
    with patch("src.api.routers.workflow.finalize_configuration") as mock_finalize:
        from src.api.dto.workflow import WorkflowCompleteResponse

        mock_finalize.return_value = WorkflowCompleteResponse(
            workflow_id="workflow123",
            phase="complete",
            final_config={"model": {"targets": ["Salary"]}},
        )

        result = asyncio.run(
            handler._handle_finalize_configuration(
                {
                    "workflow_id": "workflow123",
                    "config_updates": {
                        "features": [{"name": "Level", "monotone_constraint": 1}],
                        "quantiles": [0.1, 0.5, 0.9],
                        "hyperparameters": {"training": {}, "cv": {}},
                    },
                }
            )
        )

        assert "workflow_id" in result or "config" in result


def test_handle_finalize_configuration_with_location_settings(handler):
    """Test _handle_finalize_configuration with location settings."""
    with patch("src.api.routers.workflow.finalize_configuration") as mock_finalize:
        from src.api.dto.workflow import WorkflowCompleteResponse

        mock_finalize.return_value = WorkflowCompleteResponse(
            workflow_id="workflow123", phase="complete", final_config={}
        )

        asyncio.run(
            handler._handle_finalize_configuration(
                {
                    "workflow_id": "workflow123",
                    "config_updates": {
                        "features": [{"name": "Level", "monotone_constraint": 1}],
                        "quantiles": [0.1, 0.5, 0.9],
                        "hyperparameters": {"training": {}, "cv": {}},
                        "location_settings": {"max_distance_km": 100},
                    },
                }
            )
        )

        mock_finalize.assert_called_once()
        call_args = mock_finalize.call_args[0][1]
        assert call_args.location_settings == {"max_distance_km": 100}


def test_handle_get_feature_importance(handler):
    """Test _handle_get_feature_importance."""
    with patch("src.api.routers.analytics.get_feature_importance") as mock_get_importance:
        from src.api.dto.analytics import FeatureImportance, FeatureImportanceResponse

        mock_get_importance.return_value = FeatureImportanceResponse(
            features=[
                FeatureImportance(name="Level", gain=0.5),
                FeatureImportance(name="Location", gain=0.3),
            ]
        )

        result = asyncio.run(
            handler._handle_get_feature_importance(
                {"run_id": "test123", "target": "BaseSalary", "quantile": 0.5}
            )
        )

        assert "features" in result
        assert len(result["features"]) == 2


def test_handle_tool_call_unknown_tool(handler):
    """Test handle_tool_call with unknown tool."""
    with pytest.raises(ValueError, match="Unknown tool"):
        asyncio.run(handler.handle_tool_call("unknown_tool", {}))


def test_handle_start_training_dataset_not_found(handler):
    """Test _handle_start_training with non-existent dataset."""
    with patch("src.api.storage.DatasetStorage") as MockStorage:
        mock_storage = MagicMock()
        mock_storage.get.return_value = None
        MockStorage.return_value = mock_storage

        with pytest.raises(ValueError, match="Dataset.*not found"):
            asyncio.run(handler._handle_start_training({"dataset_id": "nonexistent", "config": {}}))


def test_handle_start_training(handler):
    """Test _handle_start_training."""
    from src.api.dto.training import TrainingJobResponse

    with (
        patch("src.api.storage.DatasetStorage") as MockStorage,
        patch("src.api.routers.training.start_training") as mock_start_training,
    ):
        mock_start_training.return_value = TrainingJobResponse(job_id="job123", status="QUEUED")

        mock_dataset = MagicMock()
        mock_storage = MagicMock()
        mock_storage.get.return_value = mock_dataset
        MockStorage.return_value = mock_storage

        result = asyncio.run(
            handler._handle_start_training(
                {
                    "dataset_id": "dataset123",
                    "config": {"model": {"targets": ["Salary"]}},
                    "remove_outliers": True,
                    "do_tune": False,
                }
            )
        )

        assert "job_id" in result
        assert result["job_id"] == "job123"


def test_handle_start_training_with_all_options(handler):
    """Test _handle_start_training with all optional parameters."""
    from src.api.dto.training import TrainingJobResponse

    call_args_capture = []

    with (
        patch("src.api.storage.DatasetStorage") as MockStorage,
        patch("src.api.routers.training.start_training") as mock_start_training,
    ):

        async def capture_and_return(request, user, training_service):
            call_args_capture.append(request)
            return TrainingJobResponse(job_id="job123", status="QUEUED")

        mock_start_training.side_effect = capture_and_return

        mock_dataset = MagicMock()
        mock_storage = MagicMock()
        mock_storage.get.return_value = mock_dataset
        MockStorage.return_value = mock_storage

        asyncio.run(
            handler._handle_start_training(
                {
                    "dataset_id": "dataset123",
                    "config": {"model": {"targets": ["Salary"]}},
                    "remove_outliers": False,
                    "do_tune": True,
                    "n_trials": 50,
                    "additional_tag": "test_tag",
                    "dataset_name": "test_dataset",
                }
            )
        )

        assert len(call_args_capture) == 1
        request = call_args_capture[0]
        assert request.n_trials == 50
        assert request.additional_tag == "test_tag"
        assert request.dataset_name == "test_dataset"
