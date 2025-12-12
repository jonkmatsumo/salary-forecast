import unittest
from unittest.mock import MagicMock, patch

from langchain_core.language_models import BaseChatModel

from src.agents.workflow import (
    PromptInjectionError,
    WorkflowState,
    create_workflow_graph,
    validate_input_node,
)


class TestValidateInputNode(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock(spec=BaseChatModel)
        self.clean_state: WorkflowState = {
            "df_json": '{"col1": [1, 2, 3]}',
            "columns": ["col1"],
            "dtypes": {"col1": "int64"},
            "dataset_size": 3,
        }

    @patch("src.agents.workflow.detect_prompt_injection")
    def test_validate_input_node_clean_data(self, mock_detect):
        """Test that valid data passes through."""
        mock_detect.return_value = {
            "is_suspicious": False,
            "confidence": 0.1,
            "reasoning": "Data is clean",
            "suspicious_content": "",
        }

        result = validate_input_node(self.clean_state, self.mock_llm)

        self.assertEqual(result, {"current_node": None})
        mock_detect.assert_called_once_with(
            self.mock_llm, self.clean_state["df_json"], self.clean_state["columns"]
        )

    @patch("src.agents.workflow.detect_prompt_injection")
    def test_validate_input_node_detects_injection(self, mock_detect):
        """Test that suspicious data raises PromptInjectionError."""
        mock_detect.return_value = {
            "is_suspicious": True,
            "confidence": 0.95,
            "reasoning": "Contains malicious instructions",
            "suspicious_content": "ignore previous instructions",
        }

        with self.assertRaises(PromptInjectionError) as context:
            validate_input_node(self.clean_state, self.mock_llm)

        self.assertGreater(context.exception.confidence, 0.8)
        self.assertIn("malicious", context.exception.reasoning.lower())
        self.assertIn("ignore", context.exception.suspicious_content.lower())

    @patch("src.agents.workflow.detect_prompt_injection")
    def test_validate_input_node_error_details(self, mock_detect):
        """Test that error contains proper details."""
        mock_detect.return_value = {
            "is_suspicious": True,
            "confidence": 0.9,
            "reasoning": "Test reasoning",
            "suspicious_content": "test content",
        }

        with self.assertRaises(PromptInjectionError) as context:
            validate_input_node(self.clean_state, self.mock_llm)

        error = context.exception
        self.assertEqual(error.confidence, 0.9)
        self.assertEqual(error.reasoning, "Test reasoning")
        self.assertEqual(error.suspicious_content, "test content")
        self.assertIn("prompt injection", str(error).lower())

    @patch("src.agents.workflow.detect_prompt_injection")
    def test_validate_input_node_state_preservation(self, mock_detect):
        """Test that state is unchanged on pass."""
        mock_detect.return_value = {
            "is_suspicious": False,
            "confidence": 0.1,
            "reasoning": "OK",
            "suspicious_content": "",
        }

        original_state = self.clean_state.copy()
        result = validate_input_node(self.clean_state, self.mock_llm)

        self.assertEqual(result, {"current_node": None})
        self.assertEqual(self.clean_state, original_state)

    @patch("src.agents.workflow.detect_prompt_injection")
    def test_validate_input_node_empty_df_json(self, mock_detect):
        """Test handling of empty df_json."""
        empty_state: WorkflowState = {"df_json": "", "columns": [], "dtypes": {}, "dataset_size": 0}

        result = validate_input_node(empty_state, self.mock_llm)

        mock_detect.assert_not_called()
        self.assertEqual(result, {"current_node": None})

    @patch("src.agents.workflow.detect_prompt_injection")
    def test_validate_input_node_detection_error(self, mock_detect):
        """Test handling of errors during detection."""
        mock_detect.side_effect = Exception("Detection error")

        result = validate_input_node(self.clean_state, self.mock_llm)

        self.assertEqual(result, {})


class TestWorkflowGraphValidation(unittest.TestCase):
    """Tests for workflow graph integration with validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock(spec=BaseChatModel)

    def test_workflow_graph_includes_validation(self):
        """Test that validation node is included in the graph."""
        graph = create_workflow_graph(self.mock_llm)

        nodes = graph.nodes
        self.assertIn("validate_input", nodes)

    def test_workflow_graph_validation_to_classification(self):
        """Test that edge exists from validation to classification."""
        graph = create_workflow_graph(self.mock_llm)

        edges = graph.edges
        validation_edges = [
            (source, target) for source, target in edges if source == "validate_input"
        ]

        self.assertTrue(any(target == "classify_columns" for source, target in validation_edges))

    def test_workflow_graph_entry_point(self):
        """Test that validation is the entry point."""
        graph = create_workflow_graph(self.mock_llm)

        # Verify validate_input node exists
        nodes = graph.nodes
        self.assertIn("validate_input", nodes)

        # Verify validate_input has expected outgoing edge to classify_columns
        edges = graph.edges
        has_outgoing_to_classify = any(
            source == "validate_input" and target == "classify_columns" for source, target in edges
        )
        self.assertTrue(
            has_outgoing_to_classify, "validate_input should have edge to classify_columns"
        )
