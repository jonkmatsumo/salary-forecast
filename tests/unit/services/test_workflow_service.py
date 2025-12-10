"""Tests for WorkflowService."""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.services.workflow_service import WorkflowService, get_workflow_providers, create_workflow_service


class TestWorkflowService(unittest.TestCase):
    """Test cases for WorkflowService."""
    
    @patch("src.services.workflow_service.get_langchain_llm")
    def test_init_success(self, mock_get_llm):
        """Test successful initialization."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        service = WorkflowService(provider="openai")
        
        self.assertIsNotNone(service.llm)
        mock_get_llm.assert_called_once_with(provider="openai", model=None)
    
    @patch("src.services.workflow_service.get_langchain_llm")
    def test_init_with_custom_model(self, mock_get_llm):
        """Test initialization with custom model."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        service = WorkflowService(provider="openai", model="gpt-4")
        
        mock_get_llm.assert_called_once_with(provider="openai", model="gpt-4")
    
    @patch("src.services.workflow_service.get_langchain_llm")
    def test_init_with_gemini(self, mock_get_llm):
        """Test initialization with Gemini provider."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        service = WorkflowService(provider="gemini")
        
        mock_get_llm.assert_called_once_with(provider="gemini", model=None)
    
    @patch("src.services.workflow_service.get_langchain_llm")
    def test_init_failure(self, mock_get_llm):
        """Test initialization failure."""
        mock_get_llm.side_effect = ValueError("Invalid API key")
        
        with self.assertRaises(ValueError):
            WorkflowService(provider="openai")
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_start_workflow(self, mock_workflow_class, mock_get_llm):
        """Test starting a workflow."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        # Mock workflow
        mock_workflow = MagicMock()
        mock_workflow.start.return_value = {
            "column_classification": {
                "targets": ["Salary"],
                "features": ["Level", "Location"],
                "ignore": ["ID"],
                "reasoning": "Test reasoning"
            },
            "classification_confirmed": False,
            "current_phase": "classification"
        }
        mock_workflow.get_current_phase.return_value = "classification"
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        
        df = pd.DataFrame({
            "ID": [1, 2],
            "Level": ["L3", "L4"],
            "Location": ["NY", "SF"],
            "Salary": [100000, 150000]
        })
        
        result = service.start_workflow(df)
        
        self.assertEqual(result["phase"], "classification")
        self.assertEqual(result["status"], "success")
        self.assertIn("data", result)
        self.assertEqual(result["data"]["targets"], ["Salary"])
        
        # Verify workflow.start was called
        mock_workflow.start.assert_called_once()
        call_kwargs = mock_workflow.start.call_args[1]
        self.assertIsNone(call_kwargs.get("preset"))
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_start_workflow_with_preset(self, mock_workflow_class, mock_get_llm):
        """Test starting a workflow with preset prompt."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.start.return_value = {
            "column_classification": {
                "targets": ["Salary"],
                "features": ["Level"],
                "ignore": [],
                "reasoning": "Test reasoning"
            },
            "classification_confirmed": False,
            "current_phase": "classification",
            "preset": "salary"
        }
        mock_workflow.get_current_phase.return_value = "classification"
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        
        df = pd.DataFrame({
            "Level": ["L3", "L4"],
            "Salary": [100000, 150000]
        })
        
        result = service.start_workflow(df, preset="salary")
        
        # Verify preset was passed to workflow
        mock_workflow.start.assert_called_once()
        call_kwargs = mock_workflow.start.call_args[1]
        self.assertEqual(call_kwargs.get("preset"), "salary")
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_start_workflow_with_optional_encodings(self, mock_workflow_class, mock_get_llm):
        """Test that optional_encodings are initialized in workflow state."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.start.return_value = {
            "column_classification": {
                "targets": ["Salary"],
                "features": ["Level"],
                "ignore": []
            },
            "classification_confirmed": False,
            "current_phase": "classification",
            "optional_encodings": {}
        }
        mock_workflow.get_current_phase.return_value = "classification"
        mock_workflow.current_state = mock_workflow.start.return_value
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        
        df = pd.DataFrame({
            "Level": ["L3"],
            "Salary": [100000]
        })
        
        result = service.start_workflow(df)
        
        # Verify optional_encodings is in state
        self.assertIn("optional_encodings", mock_workflow.current_state)
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_confirm_classification_with_optional_encodings(self, mock_workflow_class, mock_get_llm):
        """Test confirming classification with optional_encodings modifications."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.current_state = {
            "column_classification": {"targets": [], "features": []},
            "classification_confirmed": False
        }
        mock_workflow.confirm_classification.return_value = {
            "feature_encodings": {"encodings": {}},
            "encodings_confirmed": False,
            "current_phase": "encoding",
            "optional_encodings": {
                "Location": {"type": "cost_of_living", "params": {}}
            }
        }
        mock_workflow.get_current_phase.return_value = "encoding"
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        service.workflow = mock_workflow
        
        modifications = {
            "targets": ["Salary"],
            "optional_encodings": {
                "Location": {"type": "cost_of_living", "params": {}}
            }
        }
        
        result = service.confirm_classification(modifications)
        
        # Verify modifications were passed
        mock_workflow.confirm_classification.assert_called_once()
        call_args = mock_workflow.confirm_classification.call_args[0][0]
        self.assertIn("optional_encodings", call_args)
        self.assertEqual(call_args["optional_encodings"]["Location"]["type"], "cost_of_living")
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_confirm_encoding_with_optional_encodings(self, mock_workflow_class, mock_get_llm):
        """Test confirming encoding with optional_encodings modifications."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.current_state = {
            "feature_encodings": {"encodings": {}},
            "encodings_confirmed": False
        }
        mock_workflow.confirm_encoding.return_value = {
            "model_config": {},
            "current_phase": "configuration",
            "optional_encodings": {
                "Date": {"type": "normalize_recent", "params": {}}
            }
        }
        mock_workflow.get_current_phase.return_value = "configuration"
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        service.workflow = mock_workflow
        
        modifications = {
            "encodings": {"Level": {"type": "ordinal", "mapping": {}}},
            "optional_encodings": {
                "Date": {"type": "normalize_recent", "params": {}}
            }
        }
        
        result = service.confirm_encoding(modifications)
        
        # Verify modifications were passed
        mock_workflow.confirm_encoding.assert_called_once()
        call_args = mock_workflow.confirm_encoding.call_args[0][0]
        self.assertIn("optional_encodings", call_args)
        self.assertEqual(call_args["optional_encodings"]["Date"]["type"], "normalize_recent")
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_start_workflow_different_sizes(self, mock_workflow_class, mock_get_llm):
        """Test start_workflow with various DataFrame sizes."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.start.return_value = {
            "column_classification": {"targets": [], "features": [], "ignore": []},
            "current_phase": "classification"
        }
        mock_workflow.get_current_phase.return_value = "classification"
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        
        # Test with different sizes
        for size in [10, 100, 1000]:
            df = pd.DataFrame({"A": list(range(size))})
            result = service.start_workflow(df, sample_size=50)
            self.assertEqual(result["phase"], "classification")
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_start_workflow_different_sample_sizes(self, mock_workflow_class, mock_get_llm):
        """Test start_workflow with different sample sizes."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.start.return_value = {
            "column_classification": {"targets": [], "features": []},
            "current_phase": "classification"
        }
        mock_workflow.get_current_phase.return_value = "classification"
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        
        df = pd.DataFrame({"A": list(range(1000))})
        
        for sample_size in [10, 50, 100]:
            result = service.start_workflow(df, sample_size=sample_size)
            self.assertEqual(result["phase"], "classification")
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_confirm_classification_with_modifications(self, mock_workflow_class, mock_get_llm):
        """Test confirm_classification with modifications."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.confirm_classification.return_value = {
            "feature_encodings": {"encodings": {}},
            "current_phase": "encoding"
        }
        mock_workflow.get_current_phase.return_value = "encoding"
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        service.workflow = mock_workflow
        
        modifications = {
            "targets": ["Salary"],
            "features": ["Level"]
        }
        
        result = service.confirm_classification(modifications)
        
        self.assertEqual(result["phase"], "encoding")
        mock_workflow.confirm_classification.assert_called_once()
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_confirm_classification_without_modifications(self, mock_workflow_class, mock_get_llm):
        """Test confirm_classification without modifications."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.confirm_classification.return_value = {
            "feature_encodings": {},
            "current_phase": "encoding"
        }
        mock_workflow.get_current_phase.return_value = "encoding"
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        service.workflow = mock_workflow
        
        result = service.confirm_classification()
        
        self.assertEqual(result["phase"], "encoding")
        mock_workflow.confirm_classification.assert_called_once_with(None)
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_confirm_encoding_with_modifications(self, mock_workflow_class, mock_get_llm):
        """Test confirm_encoding with modifications."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.confirm_encoding.return_value = {
            "model_config": {},
            "final_config": {"model": {"targets": ["Salary"]}},
            "current_phase": "complete"
        }
        mock_workflow.get_current_phase.return_value = "complete"
        mock_workflow.get_final_config.return_value = {"model": {"targets": ["Salary"]}}
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        service.workflow = mock_workflow
        
        modifications = {
            "encodings": {"Level": {"type": "ordinal"}}
        }
        
        result = service.confirm_encoding(modifications)
        
        self.assertEqual(result["phase"], "configuration")
        mock_workflow.confirm_encoding.assert_called_once()
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_confirm_encoding_without_modifications(self, mock_workflow_class, mock_get_llm):
        """Test confirm_encoding without modifications."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.confirm_encoding.return_value = {
            "model_config": {},
            "final_config": {},
            "current_phase": "complete"
        }
        mock_workflow.get_current_phase.return_value = "complete"
        mock_workflow.get_final_config.return_value = {}
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        service.workflow = mock_workflow
        
        result = service.confirm_encoding()
        
        self.assertEqual(result["phase"], "configuration")
        mock_workflow.confirm_encoding.assert_called_once_with(None)
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_get_final_config_when_complete(self, mock_workflow_class, mock_get_llm):
        """Test get_final_config when workflow is complete."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.get_current_phase.return_value = "complete"
        mock_workflow.get_final_config.return_value = {
            "model": {"targets": ["Salary"]}
        }
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        service.workflow = mock_workflow
        
        config = service.get_final_config()
        
        self.assertIsNotNone(config)
        self.assertEqual(config["model"]["targets"], ["Salary"])
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_is_complete_at_each_phase(self, mock_workflow_class, mock_get_llm):
        """Test is_complete at each phase."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        service.workflow = mock_workflow
        
        phases = ["not_started", "classification", "encoding", "configuration", "complete"]
        expected = [False, False, False, False, True]
        
        for phase, expected_result in zip(phases, expected):
            mock_workflow.get_current_phase.return_value = phase
            self.assertEqual(service.is_complete(), expected_result)
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_get_current_state_at_each_phase(self, mock_workflow_class, mock_get_llm):
        """Test get_current_state at each phase."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.get_current_phase.return_value = "classification"
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        service.workflow = mock_workflow
        service.current_state = {
            "column_classification": {"targets": ["Salary"]},
            "current_phase": "classification"
        }
        
        state = service.get_current_state()
        
        self.assertEqual(state["phase"], "classification")
        self.assertIn("data", state)
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_format_phase_result_classification(self, mock_workflow_class, mock_get_llm):
        """Test _format_phase_result for classification phase."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_workflow_class.return_value = MagicMock()
        
        service = WorkflowService(provider="openai")
        service.current_state = {
            "column_classification": {
                "targets": ["Salary"],
                "features": ["Level"],
                "ignore": ["ID"],
                "reasoning": "Test"
            },
            "classification_confirmed": False,
            "current_phase": "classification"
        }
        
        result = service._format_phase_result("classification")
        
        self.assertEqual(result["phase"], "classification")
        self.assertIn("data", result)
        self.assertEqual(result["data"]["targets"], ["Salary"])
        self.assertFalse(result["confirmed"])
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_format_phase_result_encoding(self, mock_workflow_class, mock_get_llm):
        """Test _format_phase_result for encoding phase."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_workflow_class.return_value = MagicMock()
        
        service = WorkflowService(provider="openai")
        service.current_state = {
            "feature_encodings": {
                "encodings": {"Level": {"type": "ordinal"}},
                "summary": "Test summary"
            },
            "encodings_confirmed": False,
            "current_phase": "encoding"
        }
        
        result = service._format_phase_result("encoding")
        
        self.assertEqual(result["phase"], "encoding")
        self.assertIn("data", result)
        self.assertIn("encodings", result["data"])
        self.assertFalse(result["confirmed"])
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_format_phase_result_configuration(self, mock_workflow_class, mock_get_llm):
        """Test _format_phase_result for configuration phase."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_workflow_class.return_value = MagicMock()
        
        service = WorkflowService(provider="openai")
        service.current_state = {
            "model_config": {
                "features": [{"name": "Level", "monotone_constraint": 1}],
                "quantiles": [0.1, 0.5, 0.9],
                "hyperparameters": {},
                "reasoning": "Test"
            },
            "final_config": {"model": {"targets": ["Salary"]}},
            "current_phase": "complete"
        }
        
        result = service._format_phase_result("configuration")
        
        self.assertEqual(result["phase"], "configuration")
        self.assertIn("data", result)
        self.assertIn("final_config", result)
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_error_state_formatting(self, mock_workflow_class, mock_get_llm):
        """Test error state formatting."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_workflow_class.return_value = MagicMock()
        
        service = WorkflowService(provider="openai")
        service.current_state = {
            "error": "Test error",
            "current_phase": "classification"
        }
        
        result = service._format_phase_result("classification")
        
        self.assertEqual(result["status"], "error")
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Test error")
    
    @patch("src.services.workflow_service.get_langchain_llm")
    @patch("src.services.workflow_service.ConfigWorkflow")
    def test_workflow_start_failure(self, mock_workflow_class, mock_get_llm):
        """Test workflow start failure."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_workflow = MagicMock()
        mock_workflow.start.side_effect = Exception("Start failed")
        mock_workflow_class.return_value = mock_workflow
        
        service = WorkflowService(provider="openai")
        
        df = pd.DataFrame({"A": [1]})
        result = service.start_workflow(df)
        
        self.assertEqual(result["status"], "error")
        self.assertIn("error", result)
    
    @patch("src.services.workflow_service.get_langchain_llm")
    def test_classification_confirmation_failure(self, mock_get_llm):
        """Test classification confirmation failure."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        service = WorkflowService(provider="openai")
        service.workflow = MagicMock()
        service.workflow.confirm_classification.side_effect = Exception("Confirm failed")
        
        result = service.confirm_classification()
        
        self.assertEqual(result["status"], "error")
        self.assertIn("error", result)
    
    @patch("src.services.workflow_service.get_langchain_llm")
    def test_encoding_confirmation_failure(self, mock_get_llm):
        """Test encoding confirmation failure."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        service = WorkflowService(provider="openai")
        service.workflow = MagicMock()
        service.workflow.confirm_encoding.side_effect = Exception("Confirm failed")
        
        result = service.confirm_encoding()
        
        self.assertEqual(result["status"], "error")
        self.assertIn("error", result)
    
    @patch("src.services.workflow_service.get_langchain_llm")
    def test_service_with_no_workflow_started(self, mock_get_llm):
        """Test service methods when workflow not started."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        service = WorkflowService(provider="openai")
        
        # These should raise RuntimeError
        with self.assertRaises(RuntimeError):
            service.confirm_classification()
        
        with self.assertRaises(RuntimeError):
            service.confirm_encoding()
    
    @patch("src.services.workflow_service.get_langchain_llm")
    def test_get_current_state_not_started(self, mock_get_llm):
        """Test getting state before workflow starts."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        service = WorkflowService(provider="openai")
        state = service.get_current_state()
        
        self.assertEqual(state["phase"], "not_started")
        self.assertEqual(state["status"], "pending")
    
    @patch("src.services.workflow_service.get_langchain_llm")
    def test_is_complete_false(self, mock_get_llm):
        """Test is_complete when workflow not started."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        service = WorkflowService(provider="openai")
        self.assertFalse(service.is_complete())
    
    @patch("src.services.workflow_service.get_langchain_llm")
    def test_get_final_config_none(self, mock_get_llm):
        """Test get_final_config when workflow not complete."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        service = WorkflowService(provider="openai")
        self.assertIsNone(service.get_final_config())


class TestGetWorkflowProviders(unittest.TestCase):
    """Test cases for get_workflow_providers function."""
    
    @patch("src.services.workflow_service.get_available_providers")
    def test_get_providers(self, mock_get_available):
        """Test getting available providers."""
        mock_get_available.return_value = ["openai", "gemini"]
        
        providers = get_workflow_providers()
        
        self.assertEqual(providers, ["openai", "gemini"])
    
    @patch("src.services.workflow_service.get_available_providers")
    def test_get_providers_empty(self, mock_get_available):
        """Test getting providers when none available."""
        mock_get_available.return_value = []
        
        providers = get_workflow_providers()
        
        self.assertEqual(providers, [])


class TestCreateWorkflowService(unittest.TestCase):
    """Test cases for create_workflow_service factory."""
    
    @patch("src.services.workflow_service.WorkflowService")
    def test_create_workflow_service(self, mock_service_class):
        """Test create_workflow_service factory."""
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        
        result = create_workflow_service(provider="openai", model="gpt-4")
        
        mock_service_class.assert_called_once_with(provider="openai", model="gpt-4")
        self.assertEqual(result, mock_service)


if __name__ == "__main__":
    unittest.main()
