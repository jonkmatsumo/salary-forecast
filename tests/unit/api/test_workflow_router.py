"""Unit tests for workflow router endpoints."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.api.dto.workflow import (
    ClassificationConfirmationRequest,
    ClassificationModifications,
    ConfigurationFinalizationRequest,
    EncodingConfirmationRequest,
    EncodingModifications,
    EncodingConfig,
    OptionalEncodingConfig,
    FeatureConfig,
    Hyperparameters,
)
from src.api.exceptions import InvalidInputError, WorkflowNotFoundError
from src.api.routers.workflow import (
    _workflow_storage,
    confirm_classification,
    confirm_encoding,
    finalize_configuration,
    get_workflow_state,
    get_workflow_service,
    start_workflow,
)
from src.services.workflow_service import WorkflowService


@pytest.fixture(autouse=True)
def clear_workflow_storage():
    """Clear workflow storage before each test."""
    _workflow_storage.clear()
    yield
    _workflow_storage.clear()


class TestStartWorkflow:
    """Tests for start_workflow endpoint."""

    @patch("src.api.routers.workflow.WorkflowService")
    @patch("pandas.read_json")
    def test_start_workflow_error_status(self, mock_read_json, mock_service_class):
        """Test error status from workflow start raises InvalidInputError."""
        from src.api.dto.workflow import WorkflowStartRequest

        mock_service = MagicMock()
        mock_service.start_workflow.return_value = {
            "status": "error",
            "error": "Test error message",
        }
        mock_service_class.return_value = mock_service

        mock_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        mock_read_json.return_value = mock_df

        request = WorkflowStartRequest(
            data=json.dumps([{"col1": 1, "col2": "a"}]),
            columns=["col1", "col2"],
            dtypes={"col1": "int64", "col2": "object"},
            dataset_size=2,
            provider="openai",
        )

        with pytest.raises(InvalidInputError) as exc_info:
            asyncio.run(start_workflow(request, user="test_user"))

        assert "Test error message" in str(exc_info.value.message)

    @patch("src.api.routers.workflow.WorkflowService")
    @patch("pandas.read_json")
    def test_start_workflow_stored_in_storage(self, mock_read_json, mock_service_class):
        """Test workflow is stored in _workflow_storage."""
        from src.api.dto.workflow import WorkflowStartRequest

        mock_service = MagicMock()
        mock_service.start_workflow.return_value = {
            "status": "success",
            "phase": "classification",
            "data": {"targets": ["target1"], "features": ["feat1"]},
        }
        mock_service_class.return_value = mock_service

        mock_df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        mock_read_json.return_value = mock_df

        request = WorkflowStartRequest(
            data=json.dumps([{"col1": 1, "col2": "a"}]),
            columns=["col1", "col2"],
            dtypes={"col1": "int64", "col2": "object"},
            dataset_size=2,
            provider="openai",
        )

        response = asyncio.run(start_workflow(request, user="test_user"))

        assert response.workflow_id in _workflow_storage
        assert _workflow_storage[response.workflow_id] == mock_service
        assert response.phase == "classification"

    @patch("src.api.routers.workflow.WorkflowService")
    @patch("pandas.read_json")
    def test_start_workflow_workflow_state_construction(self, mock_read_json, mock_service_class):
        """Test WorkflowStartResponse is returned with correct phase."""
        from src.api.dto.workflow import WorkflowStartRequest

        mock_service = MagicMock()
        mock_service.start_workflow.return_value = {
            "status": "success",
            "phase": "classification",
            "data": {"targets": ["target1"]},
        }
        mock_service_class.return_value = mock_service

        mock_df = pd.DataFrame({"col1": [1]})
        mock_read_json.return_value = mock_df

        request = WorkflowStartRequest(
            data=json.dumps([{"col1": 1}]),
            columns=["col1"],
            dtypes={"col1": "int64"},
            dataset_size=1,
        )

        response = asyncio.run(start_workflow(request, user="test_user"))

        assert response.workflow_id is not None
        assert response.phase == "classification"
        assert response.state.phase == "classification"
        assert response.state.status == "success"


class TestGetWorkflowState:
    """Tests for get_workflow_state endpoint."""

    def test_get_workflow_state_not_found(self):
        """Test workflow not found raises WorkflowNotFoundError."""
        with pytest.raises(WorkflowNotFoundError) as exc_info:
            asyncio.run(get_workflow_state("nonexistent", user="test_user"))

        assert "nonexistent" in str(exc_info.value.message)
        assert exc_info.value.status_code == 404

    def test_get_workflow_state_success(self):
        """Test successful state retrieval."""
        mock_service = MagicMock(spec=WorkflowService)
        mock_service.get_current_state.return_value = {
            "phase": "encoding",
            "status": "success",
            "data": {"encodings": {}},
        }
        mock_service.current_state = {"phase": "encoding"}
        mock_service.workflow = MagicMock()

        workflow_id = "test_workflow_123"
        _workflow_storage[workflow_id] = mock_service

        response = asyncio.run(get_workflow_state(workflow_id, user="test_user"))

        assert response.workflow_id == workflow_id
        assert response.phase == "encoding"
        assert response.state == mock_service.current_state
        assert response.current_result == {"encodings": {}}

    def test_get_workflow_state_no_workflow_attribute(self):
        """Test state retrieval when service has no workflow attribute."""
        mock_service = MagicMock(spec=WorkflowService)
        mock_service.get_current_state.return_value = {
            "phase": "classification",
            "status": "success",
            "data": {},
        }
        mock_service.workflow = None

        workflow_id = "test_workflow_456"
        _workflow_storage[workflow_id] = mock_service

        response = asyncio.run(get_workflow_state(workflow_id, user="test_user"))

        assert response.workflow_id == workflow_id
        assert response.phase == "classification"
        assert response.state == {}


class TestConfirmClassification:
    """Tests for confirm_classification endpoint."""

    def test_confirm_classification_workflow_not_found(self):
        """Test workflow not found raises WorkflowNotFoundError."""
        request = ClassificationConfirmationRequest(
            modifications=ClassificationModifications(
                targets=["target1"],
                features=["feat1"],
                ignore=[],
            )
        )

        with pytest.raises(WorkflowNotFoundError) as exc_info:
            asyncio.run(confirm_classification("nonexistent", request, user="test_user"))

        assert "nonexistent" in str(exc_info.value.message)

    def test_confirm_classification_error_status(self):
        """Test error status from confirmation raises InvalidInputError."""
        mock_service = MagicMock(spec=WorkflowService)
        mock_service.confirm_classification.return_value = {
            "status": "error",
            "error": "Classification failed",
        }

        workflow_id = "test_workflow_789"
        _workflow_storage[workflow_id] = mock_service

        request = ClassificationConfirmationRequest(
            modifications=ClassificationModifications(
                targets=["target1"],
                features=["feat1"],
                ignore=[],
            )
        )

        with pytest.raises(InvalidInputError) as exc_info:
            asyncio.run(confirm_classification(workflow_id, request, user="test_user"))

        assert "Classification failed" in str(exc_info.value.message)

    def test_confirm_classification_success(self):
        """Test successful classification confirmation."""
        mock_service = MagicMock(spec=WorkflowService)
        mock_service.confirm_classification.return_value = {
            "status": "success",
            "data": {"encodings": {}},
        }

        workflow_id = "test_workflow_success"
        _workflow_storage[workflow_id] = mock_service

        request = ClassificationConfirmationRequest(
            modifications=ClassificationModifications(
                targets=["target1", "target2"],
                features=["feat1", "feat2"],
                ignore=["ignore1"],
            )
        )

        response = asyncio.run(confirm_classification(workflow_id, request, user="test_user"))

        assert response.workflow_id == workflow_id
        assert response.phase == "encoding"
        assert response.result == {"encodings": {}}

        mock_service.confirm_classification.assert_called_once_with({
            "targets": ["target1", "target2"],
            "features": ["feat1", "feat2"],
            "ignore": ["ignore1"],
        })


class TestConfirmEncoding:
    """Tests for confirm_encoding endpoint."""

    def test_confirm_encoding_workflow_not_found(self):
        """Test workflow not found raises WorkflowNotFoundError."""
        request = EncodingConfirmationRequest(
            modifications=EncodingModifications(
                encodings={},
                optional_encodings={},
            )
        )

        with pytest.raises(WorkflowNotFoundError) as exc_info:
            asyncio.run(confirm_encoding("nonexistent", request, user="test_user"))

        assert "nonexistent" in str(exc_info.value.message)

    def test_confirm_encoding_modifications_parsing(self):
        """Test encoding modifications parsing with type, mapping, and reasoning."""
        mock_service = MagicMock(spec=WorkflowService)
        mock_service.confirm_encoding.return_value = {
            "status": "success",
            "data": {"config": {}},
        }

        workflow_id = "test_encoding_parsing"
        _workflow_storage[workflow_id] = mock_service

        request = EncodingConfirmationRequest(
            modifications=EncodingModifications(
                encodings={
                    "col1": EncodingConfig(
                        type="ordinal",
                        mapping={"low": 0, "high": 1},
                        reasoning="Test reasoning",
                    ),
                    "col2": EncodingConfig(
                        type="proximity",
                        mapping=None,
                        reasoning=None,
                    ),
                },
                optional_encodings={
                    "col3": OptionalEncodingConfig(
                        type="cost_of_living",
                        params={"key": "value"},
                    ),
                },
            )
        )

        response = asyncio.run(confirm_encoding(workflow_id, request, user="test_user"))

        assert response.workflow_id == workflow_id
        assert response.phase == "configuration"

        call_args = mock_service.confirm_encoding.call_args[0][0]
        assert call_args["encodings"]["col1"]["type"] == "ordinal"
        assert call_args["encodings"]["col1"]["mapping"] == {"low": 0, "high": 1}
        assert call_args["encodings"]["col1"]["reasoning"] == "Test reasoning"
        assert call_args["encodings"]["col2"]["type"] == "proximity"
        assert call_args["encodings"]["col2"]["mapping"] == {}
        assert call_args["encodings"]["col2"]["reasoning"] == ""
        assert call_args["optional_encodings"]["col3"]["type"] == "cost_of_living"
        assert call_args["optional_encodings"]["col3"]["params"] == {"key": "value"}

    def test_confirm_encoding_error_status(self):
        """Test error status from confirmation raises InvalidInputError."""
        mock_service = MagicMock(spec=WorkflowService)
        mock_service.confirm_encoding.return_value = {
            "status": "error",
            "error": "Encoding failed",
        }

        workflow_id = "test_encoding_error"
        _workflow_storage[workflow_id] = mock_service

        request = EncodingConfirmationRequest(
            modifications=EncodingModifications(
                encodings={},
                optional_encodings={},
            )
        )

        with pytest.raises(InvalidInputError) as exc_info:
            asyncio.run(confirm_encoding(workflow_id, request, user="test_user"))

        assert "Encoding failed" in str(exc_info.value.message)

    def test_confirm_encoding_success(self):
        """Test successful encoding confirmation."""
        mock_service = MagicMock(spec=WorkflowService)
        mock_service.confirm_encoding.return_value = {
            "status": "success",
            "data": {"features": []},
        }

        workflow_id = "test_encoding_success"
        _workflow_storage[workflow_id] = mock_service

        request = EncodingConfirmationRequest(
            modifications=EncodingModifications(
                encodings={
                    "col1": EncodingConfig(type="numeric", mapping=None, reasoning=None),
                },
                optional_encodings={},
            )
        )

        response = asyncio.run(confirm_encoding(workflow_id, request, user="test_user"))

        assert response.workflow_id == workflow_id
        assert response.phase == "configuration"
        assert response.result == {"features": []}


class TestFinalizeConfiguration:
    """Tests for finalize_configuration endpoint."""

    def test_finalize_configuration_workflow_not_found(self):
        """Test workflow not found raises WorkflowNotFoundError."""
        request = ConfigurationFinalizationRequest(
            features=[FeatureConfig(name="feat1", monotone_constraint=1)],
            quantiles=[0.5],
            hyperparameters=Hyperparameters(training={}, cv={}),
        )

        with pytest.raises(WorkflowNotFoundError) as exc_info:
            asyncio.run(finalize_configuration("nonexistent", request, user="test_user"))

        assert "nonexistent" in str(exc_info.value.message)

    def test_finalize_configuration_no_final_config(self):
        """Test no final config available raises InvalidInputError."""
        mock_service = MagicMock(spec=WorkflowService)
        mock_service.get_final_config.return_value = None

        workflow_id = "test_no_config"
        _workflow_storage[workflow_id] = mock_service

        request = ConfigurationFinalizationRequest(
            features=[FeatureConfig(name="feat1", monotone_constraint=1)],
            quantiles=[0.5],
            hyperparameters=Hyperparameters(training={}, cv={}),
        )

        with pytest.raises(InvalidInputError) as exc_info:
            asyncio.run(finalize_configuration(workflow_id, request, user="test_user"))

        assert "No final configuration available" in str(exc_info.value.message)

    def test_finalize_configuration_success_with_location_settings(self):
        """Test successful finalization with location_settings."""
        mock_service = MagicMock(spec=WorkflowService)
        mock_service.get_final_config.return_value = {
            "model": {
                "targets": ["target1"],
                "features": [],
                "quantiles": [],
                "hyperparameters": {},
            },
        }

        workflow_id = "test_finalize_success"
        _workflow_storage[workflow_id] = mock_service

        request = ConfigurationFinalizationRequest(
            features=[
                FeatureConfig(name="feat1", monotone_constraint=1),
                FeatureConfig(name="feat2", monotone_constraint=-1),
            ],
            quantiles=[0.1, 0.5, 0.9],
            hyperparameters=Hyperparameters(
                training={"max_depth": 6},
                cv={"nfold": 5},
            ),
            location_settings={"max_distance_km": 50},
        )

        response = asyncio.run(finalize_configuration(workflow_id, request, user="test_user"))

        assert response.workflow_id == workflow_id
        assert response.phase == "complete"
        assert len(response.final_config["model"]["features"]) == 2
        assert response.final_config["model"]["features"][0]["name"] == "feat1"
        assert response.final_config["model"]["features"][0]["monotone_constraint"] == 1
        assert response.final_config["model"]["quantiles"] == [0.1, 0.5, 0.9]
        assert response.final_config["model"]["hyperparameters"]["training"]["max_depth"] == 6
        assert response.final_config["model"]["hyperparameters"]["cv"]["nfold"] == 5
        assert response.final_config["location_settings"]["max_distance_km"] == 50

    def test_finalize_configuration_success_without_location_settings(self):
        """Test successful finalization without location_settings."""
        mock_service = MagicMock(spec=WorkflowService)
        mock_service.get_final_config.return_value = {
            "model": {
                "targets": ["target1"],
                "features": [],
                "quantiles": [],
                "hyperparameters": {},
            },
        }

        workflow_id = "test_finalize_no_location"
        _workflow_storage[workflow_id] = mock_service

        request = ConfigurationFinalizationRequest(
            features=[FeatureConfig(name="feat1", monotone_constraint=0)],
            quantiles=[0.5],
            hyperparameters=Hyperparameters(training={"eta": 0.1}, cv={"nfold": 3}),
            location_settings=None,
        )

        response = asyncio.run(finalize_configuration(workflow_id, request, user="test_user"))

        assert response.workflow_id == workflow_id
        assert response.phase == "complete"
        assert "location_settings" not in response.final_config

    def test_finalize_configuration_updates_features_and_quantiles(self):
        """Test that features and quantiles are correctly updated in final config."""
        mock_service = MagicMock(spec=WorkflowService)
        mock_service.get_final_config.return_value = {
            "model": {
                "targets": ["target1"],
                "features": [{"name": "old_feat", "monotone_constraint": 0}],
                "quantiles": [0.25, 0.75],
                "hyperparameters": {"training": {}, "cv": {}},
            },
        }

        workflow_id = "test_update_config"
        _workflow_storage[workflow_id] = mock_service

        request = ConfigurationFinalizationRequest(
            features=[
                FeatureConfig(name="new_feat1", monotone_constraint=1),
                FeatureConfig(name="new_feat2", monotone_constraint=-1),
            ],
            quantiles=[0.1, 0.25, 0.5, 0.75, 0.9],
            hyperparameters=Hyperparameters(
                training={"max_depth": 8},
                cv={"nfold": 10},
            ),
        )

        response = asyncio.run(finalize_configuration(workflow_id, request, user="test_user"))

        final_features = response.final_config["model"]["features"]
        assert len(final_features) == 2
        assert final_features[0]["name"] == "new_feat1"
        assert final_features[1]["name"] == "new_feat2"
        assert response.final_config["model"]["quantiles"] == [0.1, 0.25, 0.5, 0.75, 0.9]

