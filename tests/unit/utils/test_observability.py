"""Tests for observability logging utilities."""

import unittest
from unittest.mock import MagicMock, patch

from src.utils.observability import (
    log_agent_interaction,
    log_llm_follow_up,
    log_llm_tool_call,
    log_tool_result,
    log_workflow_state_transition,
)


class TestObservabilityLogging(unittest.TestCase):
    """Tests for observability logging functions."""

    @patch("src.utils.observability.logger")
    def test_log_llm_tool_call(self, mock_logger):
        """Test that tool call logging works correctly."""
        log_llm_tool_call(
            agent_name="test_agent",
            tool_name="test_tool",
            tool_args={"arg1": "value1", "df_json": "large_data"},
            iteration=1,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        self.assertIn("[OBSERVABILITY]", call_args)
        self.assertIn("agent=test_agent", call_args)
        self.assertIn("tool_call=test_tool", call_args)
        self.assertIn("iteration=1", call_args)
        self.assertIn("[DataFrame JSON, length=", call_args)

    @patch("src.utils.observability.logger")
    def test_log_tool_result(self, mock_logger):
        """Test that tool result logging works correctly."""
        result = "This is a test result" * 10
        log_tool_result(agent_name="test_agent", tool_name="test_tool", result=result, iteration=2)

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        self.assertIn("[OBSERVABILITY]", call_args)
        self.assertIn("agent=test_agent", call_args)
        self.assertIn("tool_result=test_tool", call_args)
        self.assertIn("result_length=", call_args)
        self.assertIn("iteration=2", call_args)

    @patch("src.utils.observability.logger")
    def test_log_tool_result_truncates_long_results(self, mock_logger):
        """Test that long tool results are truncated."""
        long_result = "x" * 1000
        log_tool_result(
            agent_name="test_agent", tool_name="test_tool", result=long_result, iteration=1
        )

        call_args = mock_logger.info.call_args[0][0]
        self.assertIn("result_length=1000", call_args)
        self.assertIn("[truncated]", call_args)

    @patch("src.utils.observability.logger")
    def test_log_llm_follow_up(self, mock_logger):
        """Test that follow-up message logging works."""
        mock_messages = [MagicMock(), MagicMock()]
        mock_messages[-1].__class__.__name__ = "ToolMessage"

        log_llm_follow_up(agent_name="test_agent", messages=mock_messages, iteration=3)

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        self.assertIn("[OBSERVABILITY]", call_args)
        self.assertIn("agent=test_agent", call_args)
        self.assertIn("follow_up", call_args)
        self.assertIn("message_count=2", call_args)
        self.assertIn("iteration=3", call_args)

    @patch("src.utils.observability.logger")
    def test_log_agent_interaction(self, mock_logger):
        """Test that agent interaction logging works."""
        system_prompt = "System prompt"
        user_prompt = "User prompt"
        final_response = "Final response"

        log_agent_interaction(
            agent_name="test_agent",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            final_response=final_response,
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        self.assertIn("[OBSERVABILITY]", call_args)
        self.assertIn("agent=test_agent", call_args)
        self.assertIn("interaction_complete", call_args)
        self.assertIn("system_prompt_length=13", call_args)
        self.assertIn("user_prompt_length=11", call_args)
        self.assertIn("response_length=14", call_args)

    @patch("src.utils.observability.logger")
    def test_log_agent_interaction_truncates_long_content(self, mock_logger):
        """Test that long content is truncated in interaction logs."""
        long_prompt = "x" * 500
        log_agent_interaction(
            agent_name="test_agent",
            system_prompt=long_prompt,
            user_prompt="short",
            final_response="short",
        )

        call_args = mock_logger.info.call_args[0][0]
        self.assertIn("[truncated]", call_args)

    @patch("src.utils.observability.logger")
    def test_log_workflow_state_transition(self, mock_logger):
        """Test that workflow state transition logging works."""
        state = {"current_phase": "classification", "error": None, "df_json": "data"}

        log_workflow_state_transition("test_node", state)

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        self.assertIn("[OBSERVABILITY]", call_args)
        self.assertIn("workflow", call_args)
        self.assertIn("node=test_node", call_args)
        self.assertIn("phase=classification", call_args)
        self.assertIn("has_error=False", call_args)

    @patch("src.utils.observability.logger")
    def test_log_workflow_state_transition_with_error(self, mock_logger):
        """Test state transition logging with error."""
        state = {"current_phase": "error", "error": "Something went wrong"}

        log_workflow_state_transition("error_node", state)

        call_args = mock_logger.info.call_args[0][0]
        self.assertIn("has_error=True", call_args)
        self.assertIn("phase=error", call_args)

    def test_log_level_is_info(self):
        """Test that logs are at INFO level."""
        with patch("src.utils.observability.logger") as mock_logger:
            log_llm_tool_call("agent", "tool", {}, 1)
            mock_logger.info.assert_called()
            mock_logger.debug.assert_not_called()
            mock_logger.warning.assert_not_called()
