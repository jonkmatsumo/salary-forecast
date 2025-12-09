"""Tests for column classifier agent."""

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


class TestGetColumnClassifierTools(unittest.TestCase):
    """Tests for get_column_classifier_tools function."""
    
    def test_returns_correct_tools(self):
        """Test that correct tools are returned."""
        tools = get_column_classifier_tools()
        
        self.assertEqual(len(tools), 3)
        tool_names = [tool.name for tool in tools]
        
        self.assertIn("compute_correlation_matrix", tool_names)
        self.assertIn("get_column_statistics", tool_names)
        self.assertIn("detect_column_dtype", tool_names)


class TestCreateColumnClassifierAgent(unittest.TestCase):
    """Tests for create_column_classifier_agent function."""
    
    def test_binds_tools_correctly(self):
        """Test that tools are bound to LLM."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_bound = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_bound)
        
        agent = create_column_classifier_agent(mock_llm)
        
        mock_llm.bind_tools.assert_called_once()
        tools_arg = mock_llm.bind_tools.call_args[0][0]
        self.assertEqual(len(tools_arg), 3)


class TestBuildClassificationPrompt(unittest.TestCase):
    """Tests for build_classification_prompt function."""
    
    def test_prompt_includes_all_columns(self):
        """Test that prompt includes all column information."""
        df_json = '{"A": [1, 2], "B": [3, 4]}'
        columns = ["A", "B"]
        dtypes = {"A": "int64", "B": "int64"}
        
        prompt = build_classification_prompt(df_json, columns, dtypes)
        
        self.assertIn("A", prompt)
        self.assertIn("B", prompt)
        self.assertIn("int64", prompt)
        self.assertIn(df_json, prompt)
    
    def test_prompt_with_special_characters(self):
        """Test prompt formatting with special characters."""
        df_json = '{"col-name": [1], "col.name": [2]}'
        columns = ["col-name", "col.name"]
        dtypes = {"col-name": "int64", "col.name": "int64"}
        
        prompt = build_classification_prompt(df_json, columns, dtypes)
        
        # Should handle special characters in column names
        self.assertIn("col-name", prompt)
        self.assertIn("col.name", prompt)
    
    def test_prompt_with_missing_dtypes(self):
        """Test prompt with missing dtype information."""
        df_json = '{"A": [1, 2]}'
        columns = ["A"]
        dtypes = {}  # Missing dtype
        
        prompt = build_classification_prompt(df_json, columns, dtypes)
        
        # Should use 'unknown' for missing dtypes
        self.assertIn("unknown", prompt)


class TestParseClassificationResponse(unittest.TestCase):
    """Tests for parse_classification_response function."""
    
    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
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
        """Test parsing JSON in code block."""
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
        """Test parsing plain JSON without code blocks."""
        response = '{"targets": ["A"], "features": ["B"], "ignore": [], "reasoning": "test"}'
        
        result = parse_classification_response(response)
        
        self.assertEqual(result["targets"], ["A"])
        self.assertEqual(result["features"], ["B"])
    
    def test_parse_invalid_json(self):
        """Test parsing invalid JSON (error handling)."""
        response = "This is not valid JSON at all!"
        
        result = parse_classification_response(response)
        
        self.assertEqual(result["targets"], [])
        self.assertEqual(result["features"], [])
        self.assertEqual(result["ignore"], [])
        self.assertIn("Failed to parse", result["reasoning"])
        self.assertIn("raw_response", result)
    
    def test_parse_missing_keys(self):
        """Test parsing with missing keys (defaults)."""
        response = json.dumps({
            "targets": ["Salary"]
            # Missing features, ignore, reasoning
        })
        
        result = parse_classification_response(response)
        
        self.assertEqual(result["targets"], ["Salary"])
        self.assertEqual(result["features"], [])  # Default
        self.assertEqual(result["ignore"], [])  # Default
        self.assertEqual(result["reasoning"], "No reasoning provided")  # Default
    
    def test_parse_extra_keys(self):
        """Test parsing with extra keys."""
        response = json.dumps({
            "targets": ["Salary"],
            "features": ["Level"],
            "ignore": ["ID"],
            "reasoning": "Test",
            "extra_key": "extra_value"
        })
        
        result = parse_classification_response(response)
        
        # Should include extra keys
        self.assertEqual(result["targets"], ["Salary"])
        self.assertEqual(result["extra_key"], "extra_value")


class TestRunColumnClassifierSync(unittest.TestCase):
    """Tests for run_column_classifier_sync function."""
    
    @patch("src.agents.column_classifier.load_prompt")
    def test_successful_classification_no_tools(self, mock_load_prompt):
        """Test successful classification without tool calls."""
        mock_load_prompt.return_value = "System prompt"
        
        # Mock LLM that returns final answer immediately
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
        """Test tool calling loop with multiple iterations."""
        mock_load_prompt.return_value = "System prompt"
        
        # Create a mock tool
        mock_tool = MagicMock()
        mock_tool.name = "detect_column_dtype"
        mock_tool.invoke = MagicMock(return_value='{"semantic_type": "categorical"}')
        mock_get_tools.return_value = [mock_tool]
        
        mock_llm = MagicMock(spec=BaseChatModel)
        
        # First response: tool call
        tool_call_response = AIMessage(content="")
        tool_call_response.tool_calls = [{
            "name": "detect_column_dtype",
            "args": {"df_json": "{}", "column": "Level"},
            "id": "call_1"
        }]
        
        # Second response: final answer
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
        
        # Should have called tool
        mock_tool.invoke.assert_called_once()
        # Should return final result
        self.assertEqual(result["targets"], ["Salary"])
    
    @patch("src.agents.column_classifier.load_prompt")
    def test_max_iterations_limit(self, mock_load_prompt):
        """Test max_iterations limit."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        
        # Always return tool calls (infinite loop scenario)
        tool_call_response = AIMessage(content="")
        tool_call_response.tool_calls = [{
            "name": "detect_column_dtype",
            "args": {"df_json": "{}", "column": "Level"},
            "id": "call_1"
        }]
        
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = tool_call_response
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


class TestColumnClassifierIntegration(unittest.TestCase):
    """Integration tests for column classifier."""
    
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


if __name__ == "__main__":
    unittest.main()

