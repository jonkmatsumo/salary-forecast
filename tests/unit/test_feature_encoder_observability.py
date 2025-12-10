"""Tests for feature encoder observability logging."""

import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel

from src.agents.feature_encoder import run_feature_encoder_sync


class TestFeatureEncoderObservability(unittest.TestCase):
    """Tests for feature encoder observability."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock(spec=BaseChatModel)
        self.df_json = '{"col1": [1, 2, 3]}'
        self.features = ["col1"]
        self.dtypes = {"col1": "int64"}
    
    @patch("src.agents.feature_encoder.load_prompt")
    @patch("src.agents.feature_encoder.log_llm_tool_call")
    @patch("src.agents.feature_encoder.log_tool_result")
    @patch("src.agents.feature_encoder.log_llm_follow_up")
    @patch("src.agents.feature_encoder.log_agent_interaction")
    def test_feature_encoder_logs_interactions(
        self, mock_interaction, mock_follow_up, mock_result, mock_tool_call, mock_load_prompt
    ):
        """Test that feature encoder logs all interactions properly."""
        mock_load_prompt.return_value = "System prompt"
        
        tool_response = AIMessage(content="")
        tool_response.tool_calls = [{
            "name": "get_unique_value_counts",
            "args": {"df_json": self.df_json, "column": "col1"},
            "id": "call_1"
        }]
        
        final_response = AIMessage(content='{"encodings": {"col1": {"type": "numeric"}}}')
        final_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = [tool_response, final_response]
        self.mock_llm.bind_tools.return_value = mock_agent
        
        run_feature_encoder_sync(self.mock_llm, self.df_json, self.features, self.dtypes)
        
        mock_tool_call.assert_called()
        mock_result.assert_called()
        mock_follow_up.assert_called()
        mock_interaction.assert_called()
        
        interaction_call = mock_interaction.call_args
        self.assertEqual(interaction_call[0][0], "feature_encoder")

