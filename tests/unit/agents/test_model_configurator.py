"""Tests for model configurator agent."""

import unittest
import json
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel

from src.agents.model_configurator import (
    build_configuration_prompt,
    parse_configuration_response,
    get_default_hyperparameters,
    run_model_configurator_sync,
)
from src.utils.prompt_loader import load_prompt


class TestBuildConfigurationPrompt(unittest.TestCase):
    """Tests for build_configuration_prompt function."""
    
    def test_prompt_with_basic_inputs(self):
        """Test prompt building with basic inputs."""
        targets = ["Salary"]
        encodings = {
            "encodings": {
                "Level": {"type": "ordinal"},
                "Location": {"type": "proximity"}
            }
        }
        
        prompt = build_configuration_prompt(targets, encodings, dataset_size=100)
        
        self.assertIn("Salary", prompt)
        self.assertIn("Level", prompt)
        self.assertIn("Location", prompt)
        self.assertIn("100 rows", prompt)
    
    def test_prompt_with_correlation_data(self):
        """Test prompt with correlation data."""
        targets = ["Price"]
        encodings = {"encodings": {}}
        correlation_data = '{"correlations": [{"column_1": "A", "column_2": "B", "correlation": 0.8}]}'
        
        prompt = build_configuration_prompt(
            targets, encodings, correlation_data=correlation_data
        )
        
        self.assertIn("Correlation Data", prompt)
        self.assertIn(correlation_data, prompt)
    
    def test_prompt_with_column_stats(self):
        """Test prompt with column statistics."""
        targets = ["Salary"]
        encodings = {"encodings": {}}
        column_stats = {"Salary": {"mean": 100000, "std": 20000}}
        
        prompt = build_configuration_prompt(
            targets, encodings, column_stats=column_stats
        )
        
        self.assertIn("Column Statistics", prompt)
        self.assertIn("100000", prompt)
    
    def test_prompt_with_different_dataset_sizes(self):
        """Test prompt with different dataset sizes."""
        targets = ["Price"]
        encodings = {"encodings": {}}
        
        for size in [10, 100, 1000, 10000]:
            prompt = build_configuration_prompt(
                targets, encodings, dataset_size=size
            )
            self.assertIn(f"{size} rows", prompt)
    
    def test_prompt_includes_all_context(self):
        """Test prompt includes all context."""
        targets = ["Salary", "Bonus"]
        encodings = {
            "encodings": {
                "Level": {
                    "type": "ordinal",
                    "mapping": {"L1": 0, "L2": 1, "L3": 2}
                }
            }
        }
        correlation_data = '{"test": "data"}'
        column_stats = {"Salary": {"mean": 100000}}
        
        prompt = build_configuration_prompt(
            targets, encodings, correlation_data, column_stats, 500
        )
        
        # Check all sections present
        self.assertIn("Target Columns", prompt)
        self.assertIn("Feature Encodings", prompt)
        self.assertIn("Dataset Size", prompt)
        self.assertIn("Correlation Data", prompt)
        self.assertIn("Column Statistics", prompt)
        self.assertIn("monotonic constraints", prompt)
        self.assertIn("quantiles", prompt)
        self.assertIn("hyperparameters", prompt)
    
    def test_prompt_with_ordinal_mapping(self):
        """Test prompt includes ordinal mapping info."""
        encodings = {
            "encodings": {
                "Level": {
                    "type": "ordinal",
                    "mapping": {"Junior": 0, "Mid": 1, "Senior": 2}
                }
            }
        }
        
        prompt = build_configuration_prompt(["Salary"], encodings)
        
        self.assertIn("Mapping:", prompt)
        self.assertIn("Junior=0", prompt)


class TestParseConfigurationResponse(unittest.TestCase):
    """Tests for parse_configuration_response function."""
    
    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        response = json.dumps({
            "features": [
                {"name": "Level", "monotone_constraint": 1}
            ],
            "quantiles": [0.1, 0.5, 0.9],
            "hyperparameters": {
                "training": {"max_depth": 6},
                "cv": {"nfold": 5}
            },
            "reasoning": "Test reasoning"
        })
        
        result = parse_configuration_response(response)
        
        self.assertEqual(len(result["features"]), 1)
        self.assertEqual(result["features"][0]["name"], "Level")
        self.assertEqual(result["quantiles"], [0.1, 0.5, 0.9])
        self.assertIn("hyperparameters", result)
        self.assertEqual(result["reasoning"], "Test reasoning")
    
    def test_parse_missing_keys_defaults(self):
        """Test parsing with missing keys (defaults)."""
        response = json.dumps({
            "features": []
            # Missing quantiles, hyperparameters, reasoning
        })
        
        result = parse_configuration_response(response)
        
        self.assertEqual(result["features"], [])
        # Should have defaults
        self.assertEqual(result["quantiles"], [0.1, 0.25, 0.5, 0.75, 0.9])
        self.assertIn("hyperparameters", result)
        self.assertEqual(result["reasoning"], "No reasoning provided")
    
    def test_parse_invalid_json(self):
        """Test parsing with invalid JSON."""
        response = "This is not valid JSON!"
        
        result = parse_configuration_response(response)
        
        # Should return defaults
        self.assertEqual(result["features"], [])
        self.assertEqual(result["quantiles"], [0.1, 0.25, 0.5, 0.75, 0.9])
        self.assertIn("hyperparameters", result)
        self.assertIn("Failed to parse", result["reasoning"])
        self.assertIn("raw_response", result)
    
    def test_parse_default_hyperparameters_fallback(self):
        """Test default hyperparameters fallback."""
        response = json.dumps({
            "features": [],
            "quantiles": [0.5]
            # Missing hyperparameters
        })
        
        result = parse_configuration_response(response)
        
        # Should use default hyperparameters
        self.assertIn("hyperparameters", result)
        self.assertIn("training", result["hyperparameters"])
        self.assertIn("cv", result["hyperparameters"])
        self.assertEqual(
            result["hyperparameters"]["training"]["objective"],
            "reg:quantileerror"
        )


class TestGetDefaultHyperparameters(unittest.TestCase):
    """Tests for get_default_hyperparameters function."""
    
    def test_structure(self):
        """Test default hyperparameters structure."""
        defaults = get_default_hyperparameters()
        
        self.assertIn("training", defaults)
        self.assertIn("cv", defaults)
    
    def test_all_required_keys_present(self):
        """Test all required keys are present."""
        defaults = get_default_hyperparameters()
        
        training = defaults["training"]
        self.assertIn("objective", training)
        self.assertIn("tree_method", training)
        self.assertIn("max_depth", training)
        self.assertIn("eta", training)
        self.assertIn("subsample", training)
        self.assertIn("colsample_bytree", training)
        self.assertIn("verbosity", training)
        
        cv = defaults["cv"]
        self.assertIn("num_boost_round", cv)
        self.assertIn("nfold", cv)
        self.assertIn("early_stopping_rounds", cv)
        self.assertIn("verbose_eval", cv)
    
    def test_valid_value_ranges(self):
        """Test valid value ranges."""
        defaults = get_default_hyperparameters()
        
        training = defaults["training"]
        self.assertGreater(training["max_depth"], 0)
        self.assertGreater(training["eta"], 0)
        self.assertLessEqual(training["eta"], 1)
        self.assertGreater(training["subsample"], 0)
        self.assertLessEqual(training["subsample"], 1)
        
        cv = defaults["cv"]
        self.assertGreater(cv["num_boost_round"], 0)
        self.assertGreater(cv["nfold"], 0)
        self.assertGreater(cv["early_stopping_rounds"], 0)


class TestRunModelConfiguratorSync(unittest.TestCase):
    """Tests for run_model_configurator_sync function."""
    
    @patch("src.agents.model_configurator.load_prompt")
    def test_successful_configuration(self, mock_load_prompt):
        """Test successful model configuration."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = AIMessage(content=json.dumps({
            "features": [
                {"name": "Level", "monotone_constraint": 1, "reasoning": "Positive"}
            ],
            "quantiles": [0.1, 0.5, 0.9],
            "hyperparameters": {
                "training": {"max_depth": 6, "eta": 0.1},
                "cv": {"nfold": 5}
            },
            "reasoning": "Test"
        }))
        
        mock_llm.invoke.return_value = mock_response
        
        result = run_model_configurator_sync(
            mock_llm,
            ["Salary"],
            {"encodings": {"Level": {"type": "ordinal"}}}
        )
        
        self.assertEqual(len(result["features"]), 1)
        self.assertEqual(result["features"][0]["name"], "Level")
        self.assertEqual(result["quantiles"], [0.1, 0.5, 0.9])
    
    @patch("src.agents.model_configurator.load_prompt")
    def test_without_tool_calls(self, mock_load_prompt):
        """Test that agent doesn't use tools (synthesis only)."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = AIMessage(content=json.dumps({
            "features": [],
            "quantiles": [0.5],
            "hyperparameters": {}
        }))
        
        mock_llm.invoke.return_value = mock_response
        
        result = run_model_configurator_sync(
            mock_llm,
            ["Price"],
            {"encodings": {}}
        )
        
        # Should process successfully (model configurator doesn't use tools)
        self.assertIn("features", result)
        self.assertIn("quantiles", result)
    
    @patch("src.agents.model_configurator.load_prompt")
    def test_with_correlation_data(self, mock_load_prompt):
        """Test with correlation data."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = AIMessage(content=json.dumps({
            "features": [{"name": "A", "monotone_constraint": 1}],
            "quantiles": [0.5],
            "hyperparameters": {}
        }))
        
        mock_llm.invoke.return_value = mock_response
        
        correlation_data = '{"correlations": []}'
        result = run_model_configurator_sync(
            mock_llm,
            ["Price"],
            {"encodings": {}},
            correlation_data=correlation_data
        )
        
        # Should process successfully
        self.assertIn("features", result)
    
    @patch("src.agents.model_configurator.load_prompt")
    def test_with_column_stats(self, mock_load_prompt):
        """Test with column statistics."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = AIMessage(content=json.dumps({
            "features": [],
            "quantiles": [0.5],
            "hyperparameters": {}
        }))
        
        mock_llm.invoke.return_value = mock_response
        
        column_stats = {"Salary": {"mean": 100000}}
        result = run_model_configurator_sync(
            mock_llm,
            ["Salary"],
            {"encodings": {}},
            column_stats=column_stats
        )
        
        # Should process successfully
        self.assertIn("features", result)
    
    @patch("src.agents.model_configurator.load_prompt")
    def test_error_handling(self, mock_load_prompt):
        """Test error handling."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_llm.invoke.side_effect = Exception("LLM error")
        
        with self.assertRaises(Exception):
            run_model_configurator_sync(
                mock_llm,
                ["Price"],
                {"encodings": {}}
            )


class TestConfigurationValidation(unittest.TestCase):
    """Tests for configuration validation."""
    
    def test_feature_constraints_valid(self):
        """Test feature constraints are valid (-1, 0, 1)."""
        valid_constraints = [-1, 0, 1]
        
        for constraint in valid_constraints:
            response = json.dumps({
                "features": [
                    {"name": "Test", "monotone_constraint": constraint}
                ],
                "quantiles": [0.5],
                "hyperparameters": {}
            })
            
            result = parse_configuration_response(response)
            self.assertEqual(
                result["features"][0]["monotone_constraint"],
                constraint
            )
    
    def test_quantiles_valid_range(self):
        """Test quantiles are in valid range (0-1)."""
        valid_quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        
        response = json.dumps({
            "features": [],
            "quantiles": valid_quantiles,
            "hyperparameters": {}
        })
        
        result = parse_configuration_response(response)
        self.assertEqual(result["quantiles"], valid_quantiles)
    
    def test_hyperparameters_structure(self):
        """Test hyperparameters structure."""
        response = json.dumps({
            "features": [],
            "quantiles": [0.5],
            "hyperparameters": {
                "training": {
                    "objective": "reg:quantileerror",
                    "tree_method": "hist",
                    "max_depth": 6,
                    "eta": 0.1
                },
                "cv": {
                    "num_boost_round": 200,
                    "nfold": 5,
                    "early_stopping_rounds": 20
                }
            }
        })
        
        result = parse_configuration_response(response)
        
        self.assertIn("training", result["hyperparameters"])
        self.assertIn("cv", result["hyperparameters"])
        self.assertEqual(
            result["hyperparameters"]["training"]["objective"],
            "reg:quantileerror"
        )


class TestModelConfiguratorPreset(unittest.TestCase):
    """Tests for preset prompt loading in model configurator."""
    
    @patch("src.agents.model_configurator.load_prompt")
    def test_preset_prompt_loaded(self, mock_load_prompt):
        """Test that preset prompt is loaded and appended to system prompt."""
        system_prompt = "System prompt"
        preset_content = "**Domain Specific Instructions:**\nApply positive monotonicity to experience"
        
        mock_load_prompt.side_effect = lambda name: {
            "agents/model_configurator_system": system_prompt,
            "presets/salary": preset_content
        }[name]
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = AIMessage(content=json.dumps({
            "features": [{"name": "Level", "monotone_constraint": 1}],
            "quantiles": [0.5],
            "hyperparameters": {},
            "reasoning": "Test"
        }))
        
        mock_llm.invoke.return_value = mock_response
        
        result = run_model_configurator_sync(
            mock_llm,
            ["Salary"],
            {"encodings": {}},
            preset="salary"
        )
        
        # Verify both system prompt and preset were loaded
        self.assertEqual(mock_load_prompt.call_count, 2)
        mock_load_prompt.assert_any_call("agents/model_configurator_system")
        mock_load_prompt.assert_any_call("presets/salary")
    
    @patch("src.agents.model_configurator.load_prompt")
    def test_preset_none_does_not_load(self, mock_load_prompt):
        """Test that preset=None does not attempt to load preset."""
        mock_load_prompt.return_value = "System prompt"
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = AIMessage(content=json.dumps({
            "features": [],
            "quantiles": [0.5],
            "hyperparameters": {},
            "reasoning": "Test"
        }))
        
        mock_llm.invoke.return_value = mock_response
        
        result = run_model_configurator_sync(
            mock_llm,
            ["Price"],
            {"encodings": {}},
            preset=None
        )
        
        # Should only load system prompt, not preset
        mock_load_prompt.assert_called_once_with("agents/model_configurator_system")
    
    @patch("src.agents.model_configurator.load_prompt")
    def test_preset_invalid_handles_gracefully(self, mock_load_prompt):
        """Test that invalid preset name is handled gracefully."""
        mock_load_prompt.side_effect = lambda name: {
            "agents/model_configurator_system": "System prompt"
        }.get(name, FileNotFoundError(f"Prompt file not found: {name}"))
        
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_response = AIMessage(content=json.dumps({
            "features": [],
            "quantiles": [0.5],
            "hyperparameters": {},
            "reasoning": "Test"
        }))
        
        mock_llm.invoke.return_value = mock_response
        
        # Should not raise exception, just log warning
        result = run_model_configurator_sync(
            mock_llm,
            ["Price"],
            {"encodings": {}},
            preset="invalid_preset"
        )
        
        # Should still work
        self.assertIn("features", result)


if __name__ == "__main__":
    unittest.main()

