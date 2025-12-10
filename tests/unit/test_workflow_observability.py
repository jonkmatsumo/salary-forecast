"""Tests for workflow observability logging."""

import unittest
from unittest.mock import MagicMock, patch
from langchain_core.language_models import BaseChatModel

from src.agents.workflow import (
    WorkflowState,
    classify_columns_node,
    encode_features_node,
    configure_model_node,
    ConfigWorkflow,
    log_workflow_state_transition,
)


class TestWorkflowObservability(unittest.TestCase):
    """Tests for workflow state logging."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock(spec=BaseChatModel)
        self.state: WorkflowState = {
            "df_json": '{"col1": [1, 2, 3]}',
            "columns": ["col1"],
            "dtypes": {"col1": "int64"},
            "dataset_size": 3,
            "current_phase": "starting"
        }
    
    @patch("src.agents.workflow.run_column_classifier_sync")
    @patch("src.agents.workflow.log_workflow_state_transition")
    def test_classify_columns_node_logs_state(self, mock_log, mock_classify):
        """Test that state is logged before/after classification."""
        mock_classify.return_value = {
            "targets": ["col1"],
            "features": [],
            "locations": [],
            "ignore": [],
            "reasoning": "Test"
        }
        
        classify_columns_node(self.state, self.mock_llm)
        
        self.assertEqual(mock_log.call_count, 2)
        self.assertIn("classify_columns_before", mock_log.call_args_list[0][0][0])
        self.assertIn("classify_columns_after", mock_log.call_args_list[1][0][0])
    
    @patch("src.agents.workflow.run_feature_encoder_sync")
    @patch("src.agents.workflow.log_workflow_state_transition")
    def test_encode_features_node_logs_state(self, mock_log, mock_encode):
        """Test that state is logged before/after encoding."""
        mock_encode.return_value = {
            "encodings": {"col1": {"type": "numeric"}},
            "summary": "Test"
        }
        
        state_with_classification: WorkflowState = {
            **self.state,
            "column_classification": {"features": ["col1"]},
            "location_columns": []
        }
        
        encode_features_node(state_with_classification, self.mock_llm)
        
        self.assertEqual(mock_log.call_count, 2)
        self.assertIn("encode_features_before", mock_log.call_args_list[0][0][0])
        self.assertIn("encode_features_after", mock_log.call_args_list[1][0][0])
    
    @patch("src.agents.workflow.run_model_configurator_sync")
    @patch("src.agents.workflow.log_workflow_state_transition")
    def test_configure_model_node_logs_state(self, mock_log, mock_configure):
        """Test that state is logged before/after configuration."""
        mock_configure.return_value = {
            "features": ["col1"],
            "quantiles": [0.5],
            "hyperparameters": {},
            "reasoning": "Test"
        }
        
        state_with_encoding: WorkflowState = {
            **self.state,
            "column_classification": {"targets": ["target"]},
            "feature_encodings": {"encodings": {}}
        }
        
        configure_model_node(state_with_encoding, self.mock_llm)
        
        self.assertEqual(mock_log.call_count, 2)
        self.assertIn("configure_model_before", mock_log.call_args_list[0][0][0])
        self.assertIn("configure_model_after", mock_log.call_args_list[1][0][0])
    
    @patch("src.agents.workflow.compile_workflow")
    @patch("src.agents.workflow.log_workflow_state_transition")
    def test_config_workflow_logs_transitions(self, mock_log, mock_compile):
        """Test that ConfigWorkflow methods log transitions."""
        mock_workflow = MagicMock()
        mock_state_snapshot = MagicMock()
        mock_state_snapshot.values = {"current_phase": "classification"}
        mock_workflow.get_state.return_value = mock_state_snapshot
        mock_workflow.stream.return_value = []
        mock_compile.return_value = mock_workflow
        
        workflow = ConfigWorkflow(self.mock_llm)
        workflow.start("{}", ["col1"], {"col1": "int64"}, 3)
        
        mock_log.assert_called()
        call_args = mock_log.call_args
        self.assertIn("ConfigWorkflow.start", call_args[0][0])

