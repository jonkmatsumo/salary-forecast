import os
import unittest
from unittest.mock import mock_open, patch

from src.utils.prompt_loader import load_prompt


class TestPromptLoader(unittest.TestCase):
    @patch("src.utils.prompt_loader.os.path.exists")
    @patch("src.utils.prompt_loader.open", new_callable=mock_open, read_data="Mock Prompt Content")
    def test_load_prompt_success(self, mock_file, mock_exists):
        mock_exists.return_value = True

        content = load_prompt("test_prompt")

        self.assertEqual(content, "Mock Prompt Content")

        # Verify path logic implies calling exists on a path ending in test_prompt.md
        args, _ = mock_file.call_args
        self.assertTrue(args[0].endswith("test_prompt.md"))

    @patch("src.utils.prompt_loader.os.path.exists")
    def test_load_prompt_not_found(self, mock_exists):
        mock_exists.return_value = False

        with self.assertRaises(FileNotFoundError):
            load_prompt("missing_prompt")

    @patch("src.utils.prompt_loader.os.path.exists")
    @patch(
        "src.utils.prompt_loader.open",
        new_callable=mock_open,
        read_data="Column Classification System Prompt",
    )
    def test_load_column_classifier_prompt(self, mock_file, mock_exists):
        """Test loading column classifier system prompt."""
        mock_exists.return_value = True

        content = load_prompt("agents/column_classifier_system")

        self.assertIn("Column Classification", content or "")
        args, _ = mock_file.call_args
        self.assertTrue(args[0].endswith("column_classifier_system.md"))

    @patch("src.utils.prompt_loader.os.path.exists")
    @patch(
        "src.utils.prompt_loader.open",
        new_callable=mock_open,
        read_data="Feature Encoding System Prompt",
    )
    def test_load_feature_encoder_prompt(self, mock_file, mock_exists):
        """Test loading feature encoder system prompt."""
        mock_exists.return_value = True

        content = load_prompt("agents/feature_encoder_system")

        self.assertIn("Feature Encoding", content or "")
        args, _ = mock_file.call_args
        self.assertTrue(args[0].endswith("feature_encoder_system.md"))

    @patch("src.utils.prompt_loader.os.path.exists")
    @patch(
        "src.utils.prompt_loader.open",
        new_callable=mock_open,
        read_data="Model Configurator System Prompt",
    )
    def test_load_model_configurator_prompt(self, mock_file, mock_exists):
        """Test loading model configurator system prompt."""
        mock_exists.return_value = True

        content = load_prompt("agents/model_configurator_system")

        # Check that the prompt was loaded (content should contain something from the file)
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)
        # The mock returns "Model Configurator System Prompt", so check for that
        self.assertIn("Model Configurator", content or "")
        args, _ = mock_file.call_args
        self.assertTrue(args[0].endswith("model_configurator_system.md"))

    def test_agent_prompt_files_exist(self):
        """Test that agent prompt files exist."""
        import os

        from src.utils.prompt_loader import load_prompt

        # Try to load actual prompts (will fail if files don't exist)
        prompt_files = [
            "agents/column_classifier_system",
            "agents/feature_encoder_system",
            "agents/model_configurator_system",
        ]

        for prompt_name in prompt_files:
            try:
                content = load_prompt(prompt_name)
                # Verify content is not empty
                self.assertGreater(len(content), 0, f"Prompt {prompt_name} is empty")
            except FileNotFoundError:
                self.fail(f"Agent prompt file not found: {prompt_name}")

    def test_agent_prompt_content_validation(self):
        """Test agent prompt content validation."""
        from src.utils.prompt_loader import load_prompt

        # Load and validate column classifier prompt
        try:
            content = load_prompt("agents/column_classifier_system")
            # Should contain key terms
            self.assertIn("target", content.lower() or "")
            self.assertIn("feature", content.lower() or "")
        except FileNotFoundError:
            pass  # Skip if file doesn't exist in test environment

        # Load and validate feature encoder prompt
        try:
            content = load_prompt("agents/feature_encoder_system")
            # Should contain encoding types
            self.assertIn("ordinal", content.lower() or "")
            self.assertIn("numeric", content.lower() or "")
        except FileNotFoundError:
            pass

        # Load and validate model configurator prompt
        try:
            content = load_prompt("agents/model_configurator_system")
            # Should contain hyperparameters
            self.assertIn("hyperparameter", content.lower() or "")
            self.assertIn("monotonic", content.lower() or "")
        except FileNotFoundError:
            pass
