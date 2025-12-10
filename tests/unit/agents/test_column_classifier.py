import unittest
import json
from unittest.mock import MagicMock, patch, Mock
import pandas as pd
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.language_models import BaseChatModel

from src.agents.column_classifier import (
    get_column_classifier_tools,
    create_column_classifier_agent,
    build_classification_prompt,
    parse_classification_response,
    run_column_classifier_sync,
)
from src.utils.prompt_loader import load_prompt


class TestGetColumnClassifierTools(unittest.TestCase):
    def test_returns_correct_tools(self):
        tools = get_column_classifier_tools()
        
        self.assertEqual(len(tools), 3)
        tool_names = [tool.name for tool in tools]
        
        self.assertIn("compute_correlation_matrix", tool_names)
        self.assertIn("get_column_statistics", tool_names)
        self.assertIn("detect_column_dtype", tool_names)


class TestCreateColumnClassifierAgent(unittest.TestCase):
    def test_binds_tools_correctly(self):
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_bound = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_bound)
        
        agent = create_column_classifier_agent(mock_llm)
        
        mock_llm.bind_tools.assert_called_once()
        tools_arg = mock_llm.bind_tools.call_args[0][0]
        self.assertEqual(len(tools_arg), 3)


class TestBuildClassificationPrompt(unittest.TestCase):
    def test_prompt_includes_all_columns(self):
        df_json = '{"A": [1, 2], "B": [3, 4]}'
        columns = ["A", "B"]
        dtypes = {"A": "int64", "B": "int64"}
        
        prompt = build_classification_prompt(df_json, columns, dtypes)
        
        self.assertIn("A", prompt)
        self.assertIn("B", prompt)
        self.assertIn("int64", prompt)
        self.assertIn(df_json, prompt)
    
    def test_prompt_with_special_characters(self):
        df_json = '{"col-name": [1], "col.name": [2]}'
        columns = ["col-name", "col.name"]
        dtypes = {"col-name": "int64", "col.name": "int64"}
        
        prompt = build_classification_prompt(df_json, columns, dtypes)
        
        self.assertIn("col-name", prompt)
        self.assertIn("col.name", prompt)
    
    def test_prompt_with_missing_dtypes(self):
        df_json = '{"A": [1, 2]}'
        columns = ["A"]
        dtypes = {}
        
        prompt = build_classification_prompt(df_json, columns, dtypes)
        
        self.assertIn("unknown", prompt)


class TestParseClassificationResponse(unittest.TestCase):
    def test_parse_valid_json(self):
        response = json.dumps({
            "targets": ["Salary"],
            "features": ["Level", "Location"],
            "ignore": ["ID"],
            "reasoning": "Test reasoning"
        })
        
        result = parse_classification_response(response)
        
        self.assertEqual(result["targets"], ["Salary"])
        self.assertEqual(result["features"], ["Level", "Location"])
        self.assertEqual(result["ignore"], ["ID"])
        self.assertEqual(result["reasoning"], "Test reasoning")
    
    def test_parse_json_code_block(self):
        response = """Here is my classification:
```json
{
    "targets": ["Price"],
    "features": ["Size"],
    "ignore": [],
    "reasoning": "Price is the target"
}
```"""
        
        result = parse_classification_response(response)
        
        self.assertEqual(result["targets"], ["Price"])
        self.assertEqual(result["features"], ["Size"])
    
    def test_parse_plain_json(self):
        response = '{"targets": ["A"], "features": ["B"], "ignore": [], "reasoning": "test"}'
        
        result = parse_classification_response(response)
        
        self.assertEqual(result["targets"], ["A"])
        self.assertEqual(result["features"], ["B"])
    
    def test_parse_invalid_json(self):
        response = "This is not valid JSON at all!"
        
        result = parse_classification_response(response)
        
        self.assertEqual(result["targets"], [])
        self.assertEqual(result["features"], [])
        self.assertEqual(result["ignore"], [])
        self.assertIn("Failed to parse", result["reasoning"])
        self.assertIn("raw_response", result)
    
    def test_parse_missing_keys(self):
        response = json.dumps({
            "targets": ["Salary"]
        })
        
        result = parse_classification_response(response)
        
        self.assertEqual(result["targets"], ["Salary"])
        self.assertEqual(result["features"], [])
        self.assertEqual(result["ignore"], [])
        self.assertEqual(result["reasoning"], "No reasoning provided")
    
    def test_parse_extra_keys(self):
        response = json.dumps({
            "targets": ["Salary"],
            "features": ["Level"],
            "ignore": ["ID"],
            "reasoning": "Test",
            "extra_key": "extra_value"
        })
        
        result = parse_classification_response(response)
        
        self.assertEqual(result["targets"], ["Salary"])
        self.assertEqual(result["extra_key"], "extra_value")


class TestRunColumnClassifierSync(unittest.TestCase):
    @patch("src.agents.column_classifier.load_prompt")
    def test_successful_classification_no_tools(self, mock_load_prompt):
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = AIMessage(content=json.dumps({
            "targets": ["Salary"],
            "features": ["Level"],
            "ignore": ["ID"],
            "reasoning": "Test"
        }))
        mock_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_agent
        
        df = pd.DataFrame({"Salary": [100], "Level": ["L3"], "ID": [1]})
        result = run_column_classifier_sync(
            mock_llm,
            df.to_json(),
            ["Salary", "Level", "ID"],
            {"Salary": "int64", "Level": "object", "ID": "int64"}
        )
        
        self.assertEqual(result["targets"], ["Salary"])
        self.assertEqual(result["features"], ["Level"])
        self.assertEqual(result["ignore"], ["ID"])
    
    @patch("src.agents.column_classifier.load_prompt")
    @patch("src.agents.column_classifier.get_column_classifier_tools")
    def test_tool_calling_loop(self, mock_get_tools, mock_load_prompt):
        mock_load_prompt.return_value = "System prompt"
        
        mock_tool = MagicMock()
        mock_tool.name = "detect_column_dtype"
        mock_tool.invoke = MagicMock(return_value='{"semantic_type": "categorical"}')
        mock_get_tools.return_value = [mock_tool]
        
        mock_llm = MagicMock(spec=BaseChatModel)
        
        tool_call_response = AIMessage(content="")
        tool_call_response.tool_calls = [{
            "name": "detect_column_dtype",
            "args": {"df_json": "{}", "column": "Level"},
            "id": "call_1"
        }]
        
        final_response = AIMessage(content=json.dumps({
            "targets": ["Salary"],
            "features": ["Level"],
            "ignore": [],
            "reasoning": "Done"
        }))
        final_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = [tool_call_response, final_response]
        mock_llm.bind_tools.return_value = mock_agent
        
        df = pd.DataFrame({"Salary": [100], "Level": ["L3"]})
        result = run_column_classifier_sync(
            mock_llm,
            df.to_json(),
            ["Salary", "Level"],
            {"Salary": "int64", "Level": "object"}
        )
        
        mock_tool.invoke.assert_called_once()
        self.assertEqual(result["targets"], ["Salary"])
    
    @patch("src.agents.column_classifier.load_prompt")
    def test_max_iterations_limit(self, mock_load_prompt):
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        
        tool_call_response = AIMessage(content="")
        tool_call_response.tool_calls = [{
            "name": "detect_column_dtype",
            "args": {"df_json": "{}", "column": "Level"},
            "id": "call_1"
        }]
        
        final_response = AIMessage(content=json.dumps({
            "targets": [],
            "features": ["Level"],
            "ignore": [],
            "reasoning": "Max iterations reached"
        }))
        final_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = [tool_call_response, final_response]
        mock_llm.bind_tools.return_value = mock_agent
        
        with patch("src.agents.column_classifier.detect_column_dtype") as mock_tool:
            mock_tool.invoke.return_value = '{"result": "test"}'
            
            df = pd.DataFrame({"Level": ["L3"]})
            result = run_column_classifier_sync(
                mock_llm,
                df.to_json(),
                ["Level"],
                {"Level": "object"},
                max_iterations=2
            )
            
            # Should stop after max_iterations
            self.assertEqual(mock_agent.invoke.call_count, 2)
            # Should return parsed response (even if incomplete)
            self.assertIn("targets", result)
    
    @patch("src.agents.column_classifier.load_prompt")
    def test_error_handling_llm_fails(self, mock_load_prompt):
        """Test error handling when LLM fails."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = Exception("LLM error")
        mock_llm.bind_tools.return_value = mock_agent
        
        df = pd.DataFrame({"A": [1]})
        
        with self.assertRaises(Exception):
            run_column_classifier_sync(
                mock_llm,
                df.to_json(),
                ["A"],
                {"A": "int64"}
            )
    
    @patch("src.agents.column_classifier.load_prompt")
    def test_error_handling_tool_fails(self, mock_load_prompt):
        """Test error handling when tool fails."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        
        tool_call_response = AIMessage(content="")
        tool_call_response.tool_calls = [{
            "name": "detect_column_dtype",
            "args": {"df_json": "{}", "column": "Level"},
            "id": "call_1"
        }]
        
        final_response = AIMessage(content=json.dumps({
            "targets": [],
            "features": ["Level"],
            "ignore": [],
            "reasoning": "Tool failed but continuing"
        }))
        final_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = [tool_call_response, final_response]
        mock_llm.bind_tools.return_value = mock_agent
        
        with patch("src.agents.column_classifier.detect_column_dtype") as mock_tool:
            mock_tool.invoke.side_effect = Exception("Tool error")
            
            df = pd.DataFrame({"Level": ["L3"]})
            # Should handle tool error gracefully
            result = run_column_classifier_sync(
                mock_llm,
                df.to_json(),
                ["Level"],
                {"Level": "object"}
            )
            
            # Should still return a result
            self.assertIn("features", result)
    
    @patch("src.agents.column_classifier.load_prompt")
    def test_empty_dataframe(self, mock_load_prompt):
        """Test with empty DataFrame."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = AIMessage(content=json.dumps({
            "targets": [],
            "features": [],
            "ignore": [],
            "reasoning": "Empty dataset"
        }))
        mock_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_agent
        
        df = pd.DataFrame()
        result = run_column_classifier_sync(
            mock_llm,
            df.to_json(),
            [],
            {}
        )
        
        self.assertEqual(result["targets"], [])
        self.assertEqual(result["features"], [])
    
    @patch("src.agents.column_classifier.load_prompt")
    def test_single_column_dataframe(self, mock_load_prompt):
        """Test with single column DataFrame."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = AIMessage(content=json.dumps({
            "targets": ["Salary"],
            "features": [],
            "ignore": [],
            "reasoning": "Single column"
        }))
        mock_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_agent
        
        df = pd.DataFrame({"Salary": [100, 200]})
        result = run_column_classifier_sync(
            mock_llm,
            df.to_json(),
            ["Salary"],
            {"Salary": "int64"}
        )
        
        self.assertEqual(result["targets"], ["Salary"])


class TestColumnClassifierWithTools(unittest.TestCase):
    """Tests for column classifier using real tools with mocked LLM."""
    
    @patch("src.agents.column_classifier.load_prompt")
    def test_end_to_end_with_mock_llm(self, mock_load_prompt):
        """Test end-to-end with mock LLM and real tools."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        
        # Simulate tool call then final response
        tool_response = AIMessage(content="")
        tool_response.tool_calls = [{
            "name": "get_column_statistics",
            "args": {"df_json": "{}", "column": "Salary"},
            "id": "call_1"
        }]
        
        final_response = AIMessage(content=json.dumps({
            "targets": ["Salary"],
            "features": ["Level"],
            "ignore": ["ID"],
            "reasoning": "Salary is numeric and suitable as target"
        }))
        final_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = [tool_response, final_response]
        mock_llm.bind_tools.return_value = mock_agent
        
        df = pd.DataFrame({
            "Salary": [100000, 150000],
            "Level": ["L3", "L4"],
            "ID": [1, 2]
        })
        
        result = run_column_classifier_sync(
            mock_llm,
            df.to_json(),
            ["Salary", "Level", "ID"],
            {"Salary": "int64", "Level": "object", "ID": "int64"}
        )
        
        self.assertEqual(result["targets"], ["Salary"])
        self.assertEqual(result["features"], ["Level"])
        self.assertEqual(result["ignore"], ["ID"])


class TestColumnClassifierPreset(unittest.TestCase):
    """Tests for preset prompt loading in column classifier."""
    
    @patch("src.agents.column_classifier.load_prompt")
    def test_preset_prompt_loaded(self, mock_load_prompt):
        """Test that preset prompt is loaded and appended to system prompt."""
        system_prompt = "System prompt"
        preset_content = "**Domain Specific Instructions (Salary Forecasting):**\n1. Targets: Look for salary columns"
        
        mock_load_prompt.side_effect = lambda name: {
            "agents/column_classifier_system": system_prompt,
            "presets/salary": preset_content
        }[name]
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = AIMessage(content=json.dumps({
            "targets": ["Salary"],
            "features": [],
            "ignore": [],
            "reasoning": "Test"
        }))
        mock_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_agent
        
        df = pd.DataFrame({"Salary": [100000]})
        result = run_column_classifier_sync(
            mock_llm,
            df.to_json(),
            ["Salary"],
            {"Salary": "int64"},
            preset="salary"
        )
        
        # Verify both system prompt and preset were loaded
        self.assertEqual(mock_load_prompt.call_count, 2)
        mock_load_prompt.assert_any_call("agents/column_classifier_system")
        mock_load_prompt.assert_any_call("presets/salary")
    
    @patch("src.agents.column_classifier.load_prompt")
    def test_preset_none_does_not_load(self, mock_load_prompt):
        """Test that preset=None does not attempt to load preset."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = AIMessage(content=json.dumps({
            "targets": [],
            "features": [],
            "ignore": [],
            "reasoning": "Test"
        }))
        mock_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_agent
        
        df = pd.DataFrame({"A": [1]})
        result = run_column_classifier_sync(
            mock_llm,
            df.to_json(),
            ["A"],
            {"A": "int64"},
            preset=None
        )
        
        # Should only load system prompt, not preset
        mock_load_prompt.assert_called_once_with("agents/column_classifier_system")
    
    @patch("src.agents.column_classifier.load_prompt")
    def test_preset_invalid_handles_gracefully(self, mock_load_prompt):
        """Test that invalid preset name is handled gracefully."""
        mock_load_prompt.side_effect = lambda name: {
            "agents/column_classifier_system": "System prompt"
        }.get(name, FileNotFoundError(f"Prompt file not found: {name}"))
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = AIMessage(content=json.dumps({
            "targets": [],
            "features": [],
            "ignore": [],
            "reasoning": "Test"
        }))
        mock_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_response
        mock_llm.bind_tools.return_value = mock_agent
        
        df = pd.DataFrame({"A": [1]})
        # Should not raise exception, just log warning
        result = run_column_classifier_sync(
            mock_llm,
            df.to_json(),
            ["A"],
            {"A": "int64"},
            preset="invalid_preset"
        )
        
        # Should still work
        self.assertIn("targets", result)


if __name__ == "__main__":
    unittest.main()

