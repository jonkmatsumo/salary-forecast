"""Tests for workflow service prompt injection error handling."""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from langchain_core.language_models import BaseChatModel

from src.services.workflow_service import WorkflowService
from src.agents.workflow import PromptInjectionError


class TestWorkflowServiceInjection(unittest.TestCase):
    """Tests for prompt injection error handling in workflow service."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_start_workflow_detects_injection(self, mock_workflow_class, mock_get_llm):
        """Test that workflow returns error response when injection detected."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.start.side_effect = PromptInjectionError(
            "Injection detected",
            confidence=0.95,
            reasoning="Contains malicious instructions",
            suspicious_content="ignore previous instructions"
        )
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        service.workflow = mock_workflow
        
        result = service.start_workflow(self.df)
        
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["phase"], "validation")
        self.assertIn("error", result)
        self.assertIn("manipulate", result["error"].lower())
        self.assertEqual(result["error_type"], "prompt_injection")
        self.assertEqual(result["confidence"], 0.95)
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    @patch("src.services.workflow_service.logger")
    def test_start_workflow_logs_detection(self, mock_logger, mock_workflow_class, mock_get_llm):
        """Test that detection event is logged."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.start.side_effect = PromptInjectionError(
            "Injection detected",
            confidence=0.9,
            reasoning="Test",
            suspicious_content="test"
        )
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        service.workflow = mock_workflow
        
        service.start_workflow(self.df)
        
        mock_logger.warning.assert_called()
        mock_logger.info.assert_called()
        
        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        self.assertTrue(any("[OBSERVABILITY]" in str(call) for call in info_calls))
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_start_workflow_user_friendly_error(self, mock_workflow_class, mock_get_llm):
        """Test that error message is user-friendly."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.start.side_effect = PromptInjectionError(
            "Technical error",
            confidence=0.85,
            reasoning="Technical reasoning",
            suspicious_content="technical content"
        )
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        service.workflow = mock_workflow
        
        result = service.start_workflow(self.df)
        
        error_message = result["error"]
        self.assertNotIn("Technical error", error_message)
        self.assertIn("uploaded data", error_message.lower())
        self.assertIn("security", error_message.lower())
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_start_workflow_continues_on_clean_data(self, mock_workflow_class, mock_get_llm):
        """Test that normal flow continues for clean data."""
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
        
        result = service.start_workflow(self.df)
        
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["phase"], "classification")
        self.assertIn("data", result)

