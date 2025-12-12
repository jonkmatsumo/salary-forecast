"""Tests for model configurator observability logging."""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from src.agents.model_configurator import run_model_configurator_sync


class TestModelConfiguratorObservability(unittest.TestCase):
    """Tests for model configurator observability."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock(spec=BaseChatModel)
        self.targets = ["target_col"]
        self.encodings = {"encodings": {"col1": {"type": "numeric"}}}

    @patch("src.agents.model_configurator.load_prompt")
    @patch("src.agents.model_configurator.log_agent_interaction")
    def test_model_configurator_logs_interaction(self, mock_interaction, mock_load_prompt):
        """Test that model configurator logs interaction (no tools)."""
        mock_load_prompt.return_value = "System prompt"

        response = AIMessage(
            content='{"features": ["col1"], "quantiles": [0.5], "hyperparameters": {}}'
        )
        self.mock_llm.invoke.return_value = response

        run_model_configurator_sync(self.mock_llm, self.targets, self.encodings)

        mock_interaction.assert_called_once()
        call_args = mock_interaction.call_args
        self.assertEqual(call_args[0][0], "model_configurator")
        self.assertIn("System prompt", call_args[0][1])
