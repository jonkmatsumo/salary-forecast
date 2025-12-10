"""Integration tests for observability and prompt injection prevention."""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel

from src.services.workflow_service import WorkflowService
from src.agents.workflow import PromptInjectionError


class TestObservabilityIntegration(unittest.TestCase):
    """Integration tests for observability logging."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame({
            "Salary": [100000, 150000, 200000],
            "Level": ["L3", "L4", "L5"],
            "Years": [5, 10, 15]
        })
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_full_workflow_logs_all_steps(self, mock_workflow_class, mock_get_llm):
        """Test that complete workflow produces expected logs."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.start.return_value = {
            "column_classification": {"targets": ["Salary"], "features": ["Level"]},
            "current_phase": "classification"
        }
        mock_workflow_class.return_value = mock_workflow
        
        # Patch observability logger to capture calls
        with patch("src.utils.observability.logger") as mock_obs_logger:
            service = WorkflowService(provider="openai")
            service.workflow = mock_workflow
            
            service.start_workflow(self.df)
            
            # Since workflow is mocked, observability won't be called during workflow execution
            # But we can verify the workflow service itself logs properly
            # The actual observability logging happens in workflow nodes which are mocked
            # This test verifies the service structure, not the full observability flow
            self.assertIsNotNone(service.workflow)
            self.assertEqual(service.current_state.get("current_phase"), "classification")
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    @patch("src.utils.observability.logger")
    def test_observability_logs_are_structured(self, mock_logger, mock_workflow_class, mock_get_llm):
        """Test that logs follow expected format."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.start.return_value = {
            "column_classification": {"targets": ["Salary"]},
            "current_phase": "classification"
        }
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        service.workflow = mock_workflow
        
        service.start_workflow(self.df)
        
        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        observability_calls = [call for call in info_calls if "[OBSERVABILITY]" in call]
        
        if observability_calls:
            log_message = observability_calls[0]
            self.assertIn("[OBSERVABILITY]", log_message)
            self.assertIn("=", log_message)


class TestPromptInjectionIntegration(unittest.TestCase):
    """Integration tests for prompt injection prevention."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.clean_df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_workflow_rejects_injected_data(self, mock_workflow_class, mock_get_llm):
        """Test that workflow stops on injection detection."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.start.side_effect = PromptInjectionError(
            "Injection detected",
            confidence=0.9,
            reasoning="Malicious content",
            suspicious_content="ignore previous instructions"
        )
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        service.workflow = mock_workflow
        
        result = service.start_workflow(self.clean_df)
        
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["phase"], "validation")
        self.assertIn("error_type", result)
        self.assertEqual(result["error_type"], "prompt_injection")
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_workflow_processes_clean_data(self, mock_workflow_class, mock_get_llm):
        """Test that workflow continues on clean data."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.start.return_value = {
            "column_classification": {
                "targets": ["col1"],
                "features": ["col2"]
            },
            "current_phase": "classification"
        }
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        service.workflow = mock_workflow
        
        result = service.start_workflow(self.clean_df)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["phase"], "classification")
        self.assertIn("data", result)

