"""Tests for feature encoder agent."""

import unittest
import json
from unittest.mock import MagicMock, patch
import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel

from src.agents.feature_encoder import (
    get_feature_encoder_tools,
    create_feature_encoder_agent,
    build_encoding_prompt,
    parse_encoding_response,
    run_feature_encoder_sync,
)
from src.utils.prompt_loader import load_prompt


class TestGetFeatureEncoderTools(unittest.TestCase):
    """Tests for get_feature_encoder_tools function."""
    
    def test_returns_correct_tools(self):
        """Test that correct tools are returned."""
        tools = get_feature_encoder_tools()
        
        self.assertEqual(len(tools), 3)
        tool_names = [tool.name for tool in tools]
        
        self.assertIn("get_unique_value_counts", tool_names)
        self.assertIn("detect_ordinal_patterns", tool_names)
        self.assertIn("get_column_statistics", tool_names)


class TestCreateFeatureEncoderAgent(unittest.TestCase):
    """Tests for create_feature_encoder_agent function."""
    
    def test_binds_tools_correctly(self):
        """Test that tools are bound to LLM."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_bound = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_bound)
        
        agent = create_feature_encoder_agent(mock_llm)
        
        mock_llm.bind_tools.assert_called_once()
        tools_arg = mock_llm.bind_tools.call_args[0][0]
        self.assertEqual(len(tools_arg), 3)


class TestBuildEncodingPrompt(unittest.TestCase):
    """Tests for build_encoding_prompt function."""
    
    def test_prompt_includes_all_features(self):
        """Test that prompt includes all feature information."""
        df_json = '{"Level": ["L3"], "Location": ["NY"]}'
        features = ["Level", "Location"]
        dtypes = {"Level": "object", "Location": "object"}
        
        prompt = build_encoding_prompt(df_json, features, dtypes)
        
        self.assertIn("Level", prompt)
        self.assertIn("Location", prompt)
        self.assertIn(df_json, prompt)
    
    def test_prompt_with_empty_features(self):
        """Test prompt with empty features list."""
        df_json = '{}'
        features = []
        dtypes = {}
        
        prompt = build_encoding_prompt(df_json, features, dtypes)
        
        # Should still create a valid prompt
        self.assertIn("Feature Columns to Encode", prompt)
        self.assertIn(df_json, prompt)
    
    def test_prompt_formatting(self):
        """Test prompt formatting."""
        df_json = '{"A": [1]}'
        features = ["A"]
        dtypes = {"A": "int64"}
        
        prompt = build_encoding_prompt(df_json, features, dtypes)
        
        # Should include encoding instructions
        self.assertIn("numeric", prompt)
        self.assertIn("ordinal", prompt)
        self.assertIn("onehot", prompt)
        self.assertIn("proximity", prompt)
        self.assertIn("label", prompt)


class TestParseEncodingResponse(unittest.TestCase):
    """Tests for parse_encoding_response function."""
    
    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        response = json.dumps({
            "encodings": {
                "Level": {"type": "ordinal", "mapping": {"L1": 0, "L2": 1}},
                "Location": {"type": "proximity"}
            },
            "summary": "Test summary"
        })
        
        result = parse_encoding_response(response)
        
        self.assertIn("encodings", result)
        self.assertEqual(result["encodings"]["Level"]["type"], "ordinal")
        self.assertEqual(result["summary"], "Test summary")
    
    def test_parse_different_encoding_types(self):
        """Test parsing with different encoding types."""
        response = json.dumps({
            "encodings": {
                "numeric_col": {"type": "numeric"},
                "ordinal_col": {"type": "ordinal", "mapping": {"A": 0, "B": 1}},
                "onehot_col": {"type": "onehot"},
                "proximity_col": {"type": "proximity"},
                "label_col": {"type": "label"}
            }
        })
        
        result = parse_encoding_response(response)
        
        encodings = result["encodings"]
        self.assertEqual(encodings["numeric_col"]["type"], "numeric")
        self.assertEqual(encodings["ordinal_col"]["type"], "ordinal")
        self.assertEqual(encodings["onehot_col"]["type"], "onehot")
        self.assertEqual(encodings["proximity_col"]["type"], "proximity")
        self.assertEqual(encodings["label_col"]["type"], "label")
    
    def test_parse_ordinal_mappings(self):
        """Test parsing with ordinal mappings."""
        response = json.dumps({
            "encodings": {
                "Level": {
                    "type": "ordinal",
                    "mapping": {"Junior": 0, "Mid": 1, "Senior": 2},
                    "reasoning": "Natural hierarchy"
                }
            }
        })
        
        result = parse_encoding_response(response)
        
        level_enc = result["encodings"]["Level"]
        self.assertEqual(level_enc["type"], "ordinal")
        self.assertEqual(level_enc["mapping"]["Junior"], 0)
        self.assertEqual(level_enc["mapping"]["Senior"], 2)
    
    def test_parse_missing_keys(self):
        """Test parsing with missing keys."""
        response = json.dumps({
            "encodings": {
                "Level": {"type": "ordinal"}
                # Missing mapping, summary
            }
        })
        
        result = parse_encoding_response(response)
        
        self.assertIn("encodings", result)
        if "summary" not in result:
            # parse_encoding_response adds default summary
            pass
    
    def test_parse_invalid_json(self):
        """Test parsing with invalid JSON."""
        response = "This is not valid JSON!"
        
        result = parse_encoding_response(response)
        
        self.assertEqual(result["encodings"], {})
        self.assertIn("Failed to parse", result["summary"])
        self.assertIn("raw_response", result)
    
    def test_parse_without_encodings_key(self):
        """Test parsing when encodings is at root level."""
        response = json.dumps({
            "Level": {"type": "ordinal", "mapping": {"L1": 0}},
            "Location": {"type": "proximity"}
        })
        
        result = parse_encoding_response(response)
        
        # Should wrap in encodings key
        self.assertIn("encodings", result)
        self.assertIn("Level", result["encodings"])


class TestRunFeatureEncoderSync(unittest.TestCase):
    """Tests for run_feature_encoder_sync function."""
    
    @patch("src.agents.feature_encoder.load_prompt")
    def test_empty_features_list(self, mock_load_prompt):
        """Test with empty features list."""
        mock_llm = MagicMock(spec=BaseChatModel)
        
        result = run_feature_encoder_sync(
            mock_llm,
            '{}',
            [],
            {}
        )
        
        self.assertEqual(result["encodings"], {})
        self.assertEqual(result["summary"], "No features to encode")
        # Should not call LLM
        mock_llm.bind_tools.assert_not_called()
    
    @patch("src.agents.feature_encoder.load_prompt")
    def test_successful_encoding_no_tools(self, mock_load_prompt):
        """Test successful encoding without tool calls."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = AIMessage(content=json.dumps({
            "encodings": {
                "Level": {"type": "ordinal", "mapping": {"L1": 0, "L2": 1}}
            },
            "summary": "Test summary"
        }))
        mock_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_agent
        
        df = pd.DataFrame({"Level": ["L1", "L2"]})
        result = run_feature_encoder_sync(
            mock_llm,
            df.to_json(),
            ["Level"],
            {"Level": "object"}
        )
        
        self.assertIn("Level", result["encodings"])
        self.assertEqual(result["encodings"]["Level"]["type"], "ordinal")
    
    @patch("src.agents.feature_encoder.load_prompt")
    def test_tool_calling_for_each_feature(self, mock_load_prompt):
        """Test tool calling for each feature."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        
        # First: tool call for Level
        tool_call_1 = AIMessage(content="")
        tool_call_1.tool_calls = [{
            "name": "detect_ordinal_patterns",
            "args": {"df_json": "{}", "column": "Level"},
            "id": "call_1"
        }]
        
        # Second: tool call for Location
        tool_call_2 = AIMessage(content="")
        tool_call_2.tool_calls = [{
            "name": "get_unique_value_counts",
            "args": {"df_json": "{}", "column": "Location"},
            "id": "call_2"
        }]
        
        # Final: encoding result
        final_response = AIMessage(content=json.dumps({
            "encodings": {
                "Level": {"type": "ordinal"},
                "Location": {"type": "proximity"}
            },
            "summary": "Done"
        }))
        final_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = [tool_call_1, tool_call_2, final_response]
        mock_llm.bind_tools.return_value = mock_agent
        
        # Mock tools via get_feature_encoder_tools
        with patch("src.agents.feature_encoder.get_feature_encoder_tools") as mock_get_tools:
            mock_ordinal_tool = MagicMock()
            mock_ordinal_tool.name = "detect_ordinal_patterns"
            mock_ordinal_tool.invoke = MagicMock(return_value='{"is_ordinal": true}')
            
            mock_counts_tool = MagicMock()
            mock_counts_tool.name = "get_unique_value_counts"
            mock_counts_tool.invoke = MagicMock(return_value='{"total_unique_values": 10}')
            
            mock_get_tools.return_value = [mock_ordinal_tool, mock_counts_tool]
            
            df = pd.DataFrame({"Level": ["L1"], "Location": ["NY"]})
            result = run_feature_encoder_sync(
                mock_llm,
                df.to_json(),
                ["Level", "Location"],
                {"Level": "object", "Location": "object"}
            )
            
            # Should have called tools
            mock_ordinal_tool.invoke.assert_called_once()
            mock_counts_tool.invoke.assert_called_once()
            # Should return final result
            self.assertIn("Level", result["encodings"])
            self.assertIn("Location", result["encodings"])
    
    @patch("src.agents.feature_encoder.load_prompt")
    def test_max_iterations_limit(self, mock_load_prompt):
        """Test max_iterations limit."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        
        tool_call_response = AIMessage(content="")
        tool_call_response.tool_calls = [{
            "name": "detect_ordinal_patterns",
            "args": {"df_json": "{}", "column": "Level"},
            "id": "call_1"
        }]
        
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = tool_call_response
        mock_llm.bind_tools.return_value = mock_agent
        
        with patch("src.agents.feature_encoder.detect_ordinal_patterns") as mock_tool:
            mock_tool.invoke.return_value = '{"result": "test"}'
            
            df = pd.DataFrame({"Level": ["L1"]})
            result = run_feature_encoder_sync(
                mock_llm,
                df.to_json(),
                ["Level"],
                {"Level": "object"},
                max_iterations=3
            )
            
            # Should stop after max_iterations
            self.assertEqual(mock_agent.invoke.call_count, 3)
            self.assertIn("encodings", result)
    
    @patch("src.agents.feature_encoder.load_prompt")
    def test_all_numeric_features(self, mock_load_prompt):
        """Test with all numeric features."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = AIMessage(content=json.dumps({
            "encodings": {
                "Age": {"type": "numeric"},
                "Years": {"type": "numeric"}
            },
            "summary": "All numeric"
        }))
        mock_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_agent
        
        df = pd.DataFrame({"Age": [25, 30], "Years": [5, 10]})
        result = run_feature_encoder_sync(
            mock_llm,
            df.to_json(),
            ["Age", "Years"],
            {"Age": "int64", "Years": "int64"}
        )
        
        self.assertEqual(result["encodings"]["Age"]["type"], "numeric")
        self.assertEqual(result["encodings"]["Years"]["type"], "numeric")
    
    @patch("src.agents.feature_encoder.load_prompt")
    def test_all_categorical_features(self, mock_load_prompt):
        """Test with all categorical features."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = AIMessage(content=json.dumps({
            "encodings": {
                "Category": {"type": "onehot"},
                "Status": {"type": "label"}
            },
            "summary": "All categorical"
        }))
        mock_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_agent
        
        df = pd.DataFrame({"Category": ["A", "B"], "Status": ["Active", "Inactive"]})
        result = run_feature_encoder_sync(
            mock_llm,
            df.to_json(),
            ["Category", "Status"],
            {"Category": "object", "Status": "object"}
        )
        
        self.assertEqual(result["encodings"]["Category"]["type"], "onehot")
        self.assertEqual(result["encodings"]["Status"]["type"], "label")
    
    @patch("src.agents.feature_encoder.load_prompt")
    def test_error_handling(self, mock_load_prompt):
        """Test error handling."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = Exception("LLM error")
        mock_llm.bind_tools.return_value = mock_agent
        
        df = pd.DataFrame({"Level": ["L1"]})
        
        with self.assertRaises(Exception):
            run_feature_encoder_sync(
                mock_llm,
                df.to_json(),
                ["Level"],
                {"Level": "object"}
            )


class TestEncodingTypeValidation(unittest.TestCase):
    """Tests for encoding type validation."""
    
    def test_each_encoding_type(self):
        """Test each encoding type is valid."""
        valid_types = ["numeric", "ordinal", "onehot", "proximity", "label"]
        
        for enc_type in valid_types:
            response = json.dumps({
                "encodings": {
                    "test_col": {"type": enc_type}
                }
            })
            
            result = parse_encoding_response(response)
            self.assertEqual(result["encodings"]["test_col"]["type"], enc_type)
    
    def test_ordinal_mapping_validation(self):
        """Test ordinal mapping validation."""
        response = json.dumps({
            "encodings": {
                "Level": {
                    "type": "ordinal",
                    "mapping": {"L1": 0, "L2": 1, "L3": 2}
                }
            }
        })
        
        result = parse_encoding_response(response)
        
        mapping = result["encodings"]["Level"]["mapping"]
        # Verify mapping is a dict with string keys and int values
        self.assertIsInstance(mapping, dict)
        self.assertEqual(mapping["L1"], 0)
        self.assertEqual(mapping["L3"], 2)


class TestFeatureEncoderPreset(unittest.TestCase):
    """Tests for preset prompt loading in feature encoder."""
    
    @patch("src.agents.feature_encoder.load_prompt")
    def test_preset_prompt_loaded(self, mock_load_prompt):
        """Test that preset prompt is loaded and appended to system prompt."""
        system_prompt = "System prompt"
        preset_content = "**Domain Specific Instructions:**\nUse cost of living for locations"
        
        mock_load_prompt.side_effect = lambda name: {
            "agents/feature_encoder_system": system_prompt,
            "presets/salary": preset_content
        }[name]
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = AIMessage(content=json.dumps({
            "encodings": {"Location": {"type": "proximity"}},
            "summary": "Test"
        }))
        mock_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_agent
        
        df = pd.DataFrame({"Location": ["NY"]})
        result = run_feature_encoder_sync(
            mock_llm,
            df.to_json(),
            ["Location"],
            {"Location": "object"},
            preset="salary"
        )
        
        # Verify both system prompt and preset were loaded
        self.assertEqual(mock_load_prompt.call_count, 2)
        mock_load_prompt.assert_any_call("agents/feature_encoder_system")
        mock_load_prompt.assert_any_call("presets/salary")
    
    @patch("src.agents.feature_encoder.load_prompt")
    def test_preset_none_does_not_load(self, mock_load_prompt):
        """Test that preset=None does not attempt to load preset."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = AIMessage(content=json.dumps({
            "encodings": {},
            "summary": "Test"
        }))
        mock_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_agent
        
        df = pd.DataFrame({"A": [1]})
        result = run_feature_encoder_sync(
            mock_llm,
            df.to_json(),
            ["A"],
            {"A": "int64"},
            preset=None
        )
        
        # Should only load system prompt, not preset
        mock_load_prompt.assert_called_once_with("agents/feature_encoder_system")


if __name__ == "__main__":
    unittest.main()

