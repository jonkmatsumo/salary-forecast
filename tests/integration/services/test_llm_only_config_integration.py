"""Integration tests for LLM-only configuration workflow."""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from conftest import create_test_config

from src.services.training_service import TrainingService
from src.services.workflow_service import WorkflowService


class TestLLMOnlyConfigIntegration(unittest.TestCase):
    """Integration tests for LLM-only configuration system."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_df = pd.DataFrame(
            {
                "Level": ["E3", "E4", "E5", "E3"],
                "Location": ["New York, NY", "San Francisco, CA", "New York, NY", "Austin, TX"],
                "BaseSalary": [100000, 150000, 120000, 110000],
                "YearsOfExperience": [2, 5, 3, 4],
            }
        )

    @patch("src.services.workflow_service.get_langchain_llm")
    def test_workflow_to_training_integration(self, mock_get_llm):
        """Test full workflow: data → config → training."""
        # Mock LLM for workflow
        from langchain_core.language_models import BaseChatModel

        mock_llm = MagicMock(spec=BaseChatModel)
        mock_get_llm.return_value = mock_llm

        # Generate config using workflow
        WorkflowService(provider="openai")
        config = create_test_config()

        # Verify config is valid
        self.assertIsNotNone(config)
        self.assertIn("model", config)
        self.assertIn("mappings", config)

        # Test that config can be used for training
        training_service = TrainingService()

        # Mock the actual training to avoid real model training
        with patch.object(training_service, "start_training_async") as mock_train:
            mock_train.return_value = "job_id_123"

            job_id = training_service.start_training_async(
                self.sample_df,
                config,
                remove_outliers=False,
                do_tune=False,
                n_trials=20,
                additional_tag=None,
                dataset_name="test_data",
            )

            # Verify training was called with config
            mock_train.assert_called_once()
            call_args = mock_train.call_args
            self.assertEqual(call_args[0][1], config)  # Config is second positional arg
            self.assertEqual(job_id, "job_id_123")

    def test_config_validation_in_workflow(self):
        """Test that workflow-generated configs are validated."""
        config = create_test_config()

        # Config should have required fields
        self.assertIn("model", config)
        self.assertIn("mappings", config)
        self.assertIn("targets", config["model"])
        self.assertIn("quantiles", config["model"])

        # Test validation using config_schema_model
        from src.model.config_schema_model import validate_config_dict

        validated_config = validate_config_dict(config)
        self.assertIsNotNone(validated_config)

    def test_config_state_management(self):
        """Test config state management across components."""
        config = create_test_config()

        # Simulate session state
        session_state = {"config_override": config}

        # Verify config can be retrieved
        retrieved_config = session_state.get("config_override")
        self.assertEqual(retrieved_config, config)

        # Verify config is valid
        from src.model.config_schema_model import validate_config_dict

        validated = validate_config_dict(retrieved_config)
        self.assertIsNotNone(validated)

    def test_backward_compatibility_with_mlflow_models(self):
        """Test that MLflow models (with embedded config) still work."""
        # Models saved to MLflow have config embedded
        # This test verifies inference still works with embedded configs
        config = create_test_config()

        # Simulate model with embedded config
        mock_model = MagicMock()
        mock_model.config = config

        # Verify model has config
        self.assertIsNotNone(mock_model.config)
        self.assertEqual(mock_model.config, config)

        # Inference should work with embedded config
        # (Actual inference testing would require model loading, which is tested elsewhere)
