"""Integration tests for end-to-end workflow."""

import unittest
import json
from unittest.mock import MagicMock, patch
import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel

from src.services.workflow_service import WorkflowService
from src.agents.column_classifier import run_column_classifier_sync
from src.agents.feature_encoder import run_feature_encoder_sync
from src.agents.model_configurator import run_model_configurator_sync
from src.agents.tools import compute_correlation_matrix, detect_ordinal_patterns


class TestEndToEndWorkflow(unittest.TestCase):
    """End-to-end workflow integration tests."""
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_full_workflow_mock_llm(self, mock_workflow_class, mock_get_llm):
        """Test full workflow from DataFrame to final config with mock LLM."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_llm
        
        # Mock workflow that simulates full execution
        mock_workflow = MagicMock()
        
        # Start returns classification
        mock_workflow.start.return_value = {
            "column_classification": {
                "targets": ["Salary"],
                "features": ["Level", "Location"],
                "ignore": ["ID"],
                "reasoning": "Salary is target"
            },
            "current_phase": "classification"
        }
        mock_workflow.get_current_phase.return_value = "classification"
        
        # Confirm classification returns encoding
        mock_workflow.confirm_classification.return_value = {
            "feature_encodings": {
                "encodings": {
                    "Level": {"type": "ordinal", "mapping": {"L1": 0, "L2": 1}},
                    "Location": {"type": "proximity"}
                },
                "summary": "Encoded features"
            },
            "current_phase": "encoding"
        }
        
        # Confirm encoding returns final config
        mock_workflow.confirm_encoding.return_value = {
            "model_config": {
                "features": [{"name": "Level", "monotone_constraint": 1}],
                "quantiles": [0.1, 0.5, 0.9],
                "hyperparameters": {
                    "training": {"max_depth": 6},
                    "cv": {"nfold": 5}
                },
                "reasoning": "Model configured"
            },
            "final_config": {
                "model": {
                    "targets": ["Salary"],
                    "features": [{"name": "Level", "monotone_constraint": 1}],
                    "quantiles": [0.1, 0.5, 0.9]
                },
                "mappings": {},
                "feature_engineering": {}
            },
            "current_phase": "complete"
        }
        mock_workflow.get_final_config.return_value = {
            "model": {"targets": ["Salary"]}
        }
        
        mock_workflow_class.return_value = mock_workflow
        
        # Create service and run workflow
        service = WorkflowService(provider="openai")
        
        df = pd.DataFrame({
            "ID": [1, 2],
            "Level": ["L1", "L2"],
            "Location": ["NY", "SF"],
            "Salary": [100000, 150000]
        })
        
        # Start workflow
        result1 = service.start_workflow(df)
        self.assertEqual(result1["phase"], "classification")
        
        # Confirm classification
        result2 = service.confirm_classification()
        self.assertEqual(result2["phase"], "encoding")
        
        # Confirm encoding
        result3 = service.confirm_encoding()
        self.assertEqual(result3["phase"], "configuration")
        
        # Get final config
        final_config = service.get_final_config()
        self.assertIsNotNone(final_config)
        self.assertEqual(final_config["model"]["targets"], ["Salary"])
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_workflow_with_user_modifications(self, mock_workflow_class, mock_get_llm):
        """Test workflow with user modifications at each phase."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.start.return_value = {
            "column_classification": {"targets": ["A"], "features": ["B"]},
            "current_phase": "classification"
        }
        mock_workflow.get_current_phase.return_value = "classification"
        mock_workflow.confirm_classification.return_value = {
            "feature_encodings": {},
            "current_phase": "encoding"
        }
        mock_workflow.confirm_encoding.return_value = {
            "model_config": {},
            "final_config": {},
            "current_phase": "complete"
        }
        mock_workflow.get_final_config.return_value = {}
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        df = pd.DataFrame({"A": [1], "B": [2]})
        
        # Start and modify classification
        service.start_workflow(df)
        modifications = {"targets": ["ModifiedTarget"], "features": ["ModifiedFeature"]}
        service.confirm_classification(modifications)
        
        # Verify modifications were passed
        call_args = mock_workflow.confirm_classification.call_args[0]
        self.assertEqual(call_args[0], modifications)
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_workflow_error_recovery(self, mock_workflow_class, mock_get_llm):
        """Test workflow error recovery."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.start.side_effect = Exception("Start failed")
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        df = pd.DataFrame({"A": [1]})
        
        result = service.start_workflow(df)
        
        # Should handle error gracefully
        self.assertEqual(result["status"], "error")
        self.assertIn("error", result)


class TestServiceIntegration(unittest.TestCase):
    """Tests for service integration."""
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.config_generator.ConfigGenerator")
    def test_workflow_service_with_fallback(self, mock_config_gen, mock_get_llm):
        """Test WorkflowService with ConfigGenerator fallback."""
        # This test verifies that services can work together
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_gen = MagicMock()
        mock_gen.generate_config_template.return_value = {
            "model": {"targets": ["Salary"]}
        }
        mock_config_gen.return_value = mock_gen
        
        # WorkflowService should work independently
        service = WorkflowService(provider="openai")
        self.assertIsNotNone(service)


class TestToolIntegration(unittest.TestCase):
    """Tests for tool integration with agents."""
    
    @patch("src.agents.column_classifier.load_prompt")
    def test_agents_using_real_tools(self, mock_load_prompt):
        """Test agents using real tools with real DataFrames."""
        mock_load_prompt.return_value = "System prompt"
        
        # Create real DataFrame
        df = pd.DataFrame({
            "Salary": [100000, 150000, 200000],
            "Level": ["L3", "L4", "L5"],
            "Years": [5, 10, 15]
        })
        
        # Mock LLM that uses tools
        mock_llm = MagicMock(spec=BaseChatModel)
        
        # First response: tool call
        tool_response = AIMessage(content="")
        tool_response.tool_calls = [{
            "name": "compute_correlation_matrix",
            "args": {"df_json": df.to_json(), "columns": None},
            "id": "call_1"
        }]
        
        # Second response: final answer
        final_response = AIMessage(content=json.dumps({
            "targets": ["Salary"],
            "features": ["Level", "Years"],
            "ignore": [],
            "reasoning": "Used correlation tool"
        }))
        final_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = [tool_response, final_response]
        mock_llm.bind_tools.return_value = mock_agent
        
        # Run classifier - should use real tools
        result = run_column_classifier_sync(
            mock_llm,
            df.to_json(),
            ["Salary", "Level", "Years"],
            {"Salary": "int64", "Level": "object", "Years": "int64"}
        )
        
        # Should have called tool and got result
        self.assertIn("targets", result)
        # Tool should have been invoked with real data
        self.assertTrue(mock_agent.invoke.called)
    
    def test_tool_results_influence_decisions(self):
        """Test that tool results influence agent decisions."""
        # This is more of a conceptual test - in real usage, tools provide
        # data that agents use to make decisions
        
        df = pd.DataFrame({
            "Level": ["L1", "L2", "L3", "L4", "L5"]
        })
        
        # Use real ordinal detection tool
        result = detect_ordinal_patterns.invoke({
            "df_json": df.to_json(),
            "column": "Level"
        })
        
        result_dict = json.loads(result)
        
        # Tool should detect ordinal pattern
        self.assertTrue(result_dict["is_ordinal"])
        # This result would influence agent to recommend ordinal encoding
    
    @patch("src.agents.column_classifier.load_prompt")
    def test_llm_tool_call_with_escaped_json(self, mock_load_prompt):
        """Test that tools handle escaped JSON from LLM tool calls."""
        mock_load_prompt.return_value = "System prompt"
        
        df = pd.DataFrame({
            "Level": {"0": "E6", "1": "E6", "2": "E3", "3": "E3"},
            "Salary": {"0": 100000, "1": 150000, "2": 120000, "3": 130000}
        })
        
        normal_json = df.to_json()
        escaped_json = json.dumps(normal_json)
        
        mock_llm = MagicMock(spec=BaseChatModel)
        
        tool_response = AIMessage(content="")
        tool_response.tool_calls = [{
            "name": "detect_column_dtype",
            "args": {"df_json": escaped_json, "column": "Level"},
            "id": "call_1"
        }]
        
        final_response = AIMessage(content=json.dumps({
            "targets": ["Salary"],
            "features": ["Level"],
            "ignore": [],
            "reasoning": "Used dtype detection tool"
        }))
        final_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = [tool_response, final_response]
        mock_llm.bind_tools.return_value = mock_agent
        
        result = run_column_classifier_sync(
            mock_llm,
            df.to_json(),
            ["Level", "Salary"],
            {"Level": "object", "Salary": "int64"}
        )
        
        self.assertIn("targets", result)
        self.assertTrue(mock_agent.invoke.called)
    
    @patch("src.agents.column_classifier.load_prompt")
    def test_multiple_tool_calls_with_escaped_json(self, mock_load_prompt):
        """Test multiple tool calls with escaped JSON."""
        mock_load_prompt.return_value = "System prompt"
        
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [2, 4, 6, 8, 10],
            "C": ["x", "y", "z", "x", "y"]
        })
        
        normal_json = df.to_json()
        escaped_json = json.dumps(normal_json)
        
        mock_llm = MagicMock(spec=BaseChatModel)
        
        tool_response_1 = AIMessage(content="")
        tool_response_1.tool_calls = [{
            "name": "compute_correlation_matrix",
            "args": {"df_json": escaped_json, "columns": None},
            "id": "call_1"
        }]
        
        tool_response_2 = AIMessage(content="")
        tool_response_2.tool_calls = [{
            "name": "get_column_statistics",
            "args": {"df_json": escaped_json, "column": "A"},
            "id": "call_2"
        }]
        
        final_response = AIMessage(content=json.dumps({
            "targets": ["A"],
            "features": ["B", "C"],
            "ignore": [],
            "reasoning": "Used multiple tools"
        }))
        final_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = [tool_response_1, tool_response_2, final_response]
        mock_llm.bind_tools.return_value = mock_agent
        
        result = run_column_classifier_sync(
            mock_llm,
            df.to_json(),
            ["A", "B", "C"],
            {"A": "int64", "B": "int64", "C": "object"}
        )
        
        self.assertIn("targets", result)
        self.assertEqual(mock_agent.invoke.call_count, 3)
    
    def test_real_world_escaped_json_from_issue(self):
        """Test with the exact escaped JSON format from the issue."""
        escaped_json = '{\\"Level\\": {\\"0\\": \\"E6\\", \\"1\\": \\"E6\\", \\"2\\": \\"E3\\", \\"3\\": \\"E3\\"}}'
        
        from src.agents.tools import detect_column_dtype
        
        result = detect_column_dtype.invoke({
            "df_json": escaped_json,
            "column": "Level"
        })
        result_dict = json.loads(result)
        
        self.assertEqual(result_dict["column"], "Level")
        self.assertNotIn("error", result_dict)
        self.assertIn("semantic_type", result_dict)
    
    @patch("src.agents.column_classifier.load_prompt")
    def test_tool_error_handling_in_agents(self, mock_load_prompt):
        """Test tool error handling in agents."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        
        # Agent calls tool that fails
        tool_response = AIMessage(content="")
        tool_response.tool_calls = [{
            "name": "get_column_statistics",
            "args": {"df_json": "{}", "column": "Invalid"},
            "id": "call_1"
        }]
        
        # Agent continues with final answer despite tool error
        final_response = AIMessage(content=json.dumps({
            "targets": [],
            "features": [],
            "ignore": [],
            "reasoning": "Tool failed but continued"
        }))
        final_response.tool_calls = []
        
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = [tool_response, final_response]
        mock_llm.bind_tools.return_value = mock_agent
        
        # Mock tool to raise error
        with patch("src.agents.column_classifier.get_column_statistics") as mock_tool:
            mock_tool.invoke.side_effect = Exception("Tool error")
            
            df = pd.DataFrame({"A": [1]})
            result = run_column_classifier_sync(
                mock_llm,
                df.to_json(),
                ["A"],
                {"A": "int64"}
            )
            
            # Should still return a result (graceful degradation)
            self.assertIn("targets", result)


if __name__ == "__main__":
    unittest.main()

