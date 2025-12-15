"""Tests for workflow node tracking and observability."""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.workflow import (
    build_final_config_node,
    classify_columns_node,
    configure_model_node,
    encode_features_node,
    validate_input_node,
)


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    llm = MagicMock()
    return llm


@pytest.fixture
def sample_state():
    """Create a sample workflow state."""
    return {
        "df_json": '{"col1": [1, 2], "col2": ["a", "b"]}',
        "columns": ["col1", "col2"],
        "dtypes": {"col1": "int64", "col2": "object"},
        "dataset_size": 2,
        "preset": None,
    }


def test_validate_input_sets_current_node(mock_llm, sample_state):
    """Test that validate_input_node sets current_node."""
    with patch("src.agents.workflow.detect_prompt_injection") as mock_detect:
        mock_detect.return_value = {"is_suspicious": False}

        result = validate_input_node(sample_state, mock_llm)

        assert "current_node" in result
        assert result["current_node"] is None  # Should be None after completion


def test_classify_columns_sets_current_node(mock_llm, sample_state):
    """Test that classify_columns_node sets current_node."""
    with patch("src.agents.workflow.run_column_classifier_sync") as mock_classify:
        mock_classify.return_value = {
            "targets": ["col1"],
            "features": ["col2"],
            "ignore": [],
            "column_types": {},
            "reasoning": "Test",
        }
        with patch("src.agents.workflow.compute_correlation_matrix") as mock_corr:
            mock_corr.invoke.return_value = "correlation_data"

            result = classify_columns_node(sample_state, mock_llm)

            assert "current_node" in result
            assert result["current_node"] == "classifying_columns"


def test_encode_features_sets_current_node(mock_llm, sample_state):
    """Test that encode_features_node sets current_node."""
    sample_state["column_classification"] = {
        "targets": ["col1"],
        "features": ["col2"],
        "ignore": [],
    }
    sample_state["location_columns"] = []

    with patch("src.agents.workflow.run_feature_encoder_sync") as mock_encode:
        mock_encode.return_value = {"encodings": {"col2": {"type": "numeric"}}, "summary": "Test"}

        result = encode_features_node(sample_state, mock_llm)

        assert "current_node" in result
        assert result["current_node"] == "evaluating_features"


def test_configure_model_sets_current_node(mock_llm, sample_state):
    """Test that configure_model_node sets current_node."""
    sample_state["column_classification"] = {
        "targets": ["col1"],
        "features": ["col2"],
        "ignore": [],
    }
    sample_state["feature_encodings"] = {"encodings": {"col2": {"type": "numeric"}}}

    with patch("src.agents.workflow.run_model_configurator_sync") as mock_config:
        mock_config.return_value = {
            "features": [{"name": "col2", "monotone_constraint": 0}],
            "quantiles": [0.1, 0.5, 0.9],
            "hyperparameters": {},
            "reasoning": "Test",
        }

        result = configure_model_node(sample_state, mock_llm)

        assert "current_node" in result
        assert result["current_node"] == "configuring_model"


def test_build_final_config_sets_current_node(sample_state):
    """Test that build_final_config_node sets current_node."""
    sample_state["column_classification"] = {
        "targets": ["col1"],
        "features": ["col2"],
        "ignore": [],
    }
    sample_state["feature_encodings"] = {"encodings": {}}
    sample_state["model_config"] = {
        "features": [{"name": "col2", "monotone_constraint": 0}],
        "quantiles": [0.1, 0.5, 0.9],
    }
    sample_state["location_columns"] = []
    sample_state["location_settings"] = {"max_distance_km": 50}

    result = build_final_config_node(sample_state)

    assert "current_node" in result
    assert result["current_node"] == "building_config"


def test_ui_progress_message_helper():
    """Test the UI progress message helper function."""
    from src.app.config_ui import _get_progress_message

    # Test with None service
    assert _get_progress_message(None) == "Processing..."

    # Test with service but no workflow
    mock_service = MagicMock()
    mock_service.workflow = None
    assert _get_progress_message(mock_service) == "Processing..."

    # Test with different node states
    mock_service.workflow = MagicMock()
    mock_service.workflow.current_state = {"current_node": "classifying_columns"}
    assert _get_progress_message(mock_service) == "Classifying columns..."

    mock_service.workflow.current_state = {"current_node": "evaluating_features"}
    assert _get_progress_message(mock_service) == "Evaluating features..."

    mock_service.workflow.current_state = {"current_node": "configuring_model"}
    assert _get_progress_message(mock_service) == "Configuring model..."

    mock_service.workflow.current_state = {"current_node": "building_config"}
    assert _get_progress_message(mock_service) == "Building final configuration..."
