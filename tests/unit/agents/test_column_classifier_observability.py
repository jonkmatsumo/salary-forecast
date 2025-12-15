"""Tests for column classifier observability logging."""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from src.agents.column_classifier import run_column_classifier_sync


class TestColumnClassifierObservability(unittest.TestCase):
    """Tests for column classifier observability."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock(spec=BaseChatModel)
        self.df_json = '{"col1": [1, 2, 3]}'
        self.columns = ["col1"]
        self.dtypes = {"col1": "int64"}

    @patch("src.agents.column_classifier.load_prompt")
    @patch("src.agents.column_classifier.log_llm_tool_call")
    @patch("src.agents.column_classifier.log_tool_result")
    @patch("src.agents.column_classifier.log_llm_follow_up")
    def test_column_classifier_logs_tool_calls(
        self, mock_follow_up, mock_result, mock_tool_call, mock_load_prompt
    ):
        """Test that tool calls are logged."""
        mock_load_prompt.return_value = "System prompt"

        tool_response = AIMessage(content="")
        tool_response.tool_calls = [
            {
                "name": "compute_correlation_matrix",
                "args": {"df_json": self.df_json, "columns": None},
                "id": "call_1",
            }
        ]

        final_response = AIMessage(content='{"targets": [], "features": ["col1"]}')
        final_response.tool_calls = []

        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = [tool_response, final_response]
        self.mock_llm.bind_tools.return_value = mock_agent

        run_column_classifier_sync(self.mock_llm, self.df_json, self.columns, self.dtypes)

        mock_tool_call.assert_called()
        call_args = mock_tool_call.call_args
        self.assertEqual(call_args[0][0], "column_classifier")
        self.assertEqual(call_args[0][1], "compute_correlation_matrix")

    @patch("src.agents.column_classifier.load_prompt")
    @patch("src.agents.column_classifier.log_tool_result")
    def test_column_classifier_logs_tool_results(self, mock_result, mock_load_prompt):
        """Test that tool results are logged."""
        mock_load_prompt.return_value = "System prompt"

        tool_response = AIMessage(content="")
        tool_response.tool_calls = [
            {
                "name": "get_column_statistics",
                "args": {"df_json": self.df_json, "column": "col1"},
                "id": "call_1",
            }
        ]

        final_response = AIMessage(content='{"targets": [], "features": ["col1"]}')
        final_response.tool_calls = []

        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = [tool_response, final_response]
        self.mock_llm.bind_tools.return_value = mock_agent

        run_column_classifier_sync(self.mock_llm, self.df_json, self.columns, self.dtypes)

        mock_result.assert_called()
        call_args = mock_result.call_args
        self.assertEqual(call_args[0][0], "column_classifier")
        self.assertEqual(call_args[0][1], "get_column_statistics")

    @patch("src.agents.column_classifier.load_prompt")
    @patch("src.agents.column_classifier.log_llm_follow_up")
    def test_column_classifier_logs_follow_up(self, mock_follow_up, mock_load_prompt):
        """Test that follow-up messages are logged."""
        mock_load_prompt.return_value = "System prompt"

        tool_response = AIMessage(content="")
        tool_response.tool_calls = [
            {
                "name": "detect_column_dtype",
                "args": {"df_json": self.df_json, "column": "col1"},
                "id": "call_1",
            }
        ]

        final_response = AIMessage(content='{"targets": [], "features": ["col1"]}')
        final_response.tool_calls = []

        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = [tool_response, final_response]
        self.mock_llm.bind_tools.return_value = mock_agent

        run_column_classifier_sync(self.mock_llm, self.df_json, self.columns, self.dtypes)

        mock_follow_up.assert_called()
        call_args = mock_follow_up.call_args
        self.assertEqual(call_args[0][0], "column_classifier")

    @patch("src.agents.column_classifier.load_prompt")
    @patch("src.agents.column_classifier.log_agent_interaction")
    def test_column_classifier_logs_interaction(self, mock_interaction, mock_load_prompt):
        """Test that complete interaction is logged."""
        mock_load_prompt.return_value = "System prompt"

        final_response = AIMessage(content='{"targets": ["col1"], "features": []}')
        final_response.tool_calls = []

        mock_agent = MagicMock()
        mock_agent.invoke.return_value = final_response
        self.mock_llm.bind_tools.return_value = mock_agent

        run_column_classifier_sync(self.mock_llm, self.df_json, self.columns, self.dtypes)

        mock_interaction.assert_called()
        call_args = mock_interaction.call_args
        self.assertEqual(call_args[0][0], "column_classifier")
        self.assertIn("System prompt", call_args[0][1])
