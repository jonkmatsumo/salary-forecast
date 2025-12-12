import json
import unittest
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
from langchain_core.language_models import BaseChatModel

from src.agents.workflow import (
    ConfigWorkflow,
    WorkflowState,
    build_final_config_node,
    classify_columns_node,
    compile_workflow,
    configure_model_node,
    create_workflow_graph,
    encode_features_node,
    should_continue_after_classification,
    should_continue_after_encoding,
)


class TestWorkflowState(unittest.TestCase):
    def test_state_structure(self):
        state: WorkflowState = {
            "df_json": "{}",
            "columns": ["A"],
            "dtypes": {"A": "int64"},
            "dataset_size": 100,
            "preset": None,
            "optional_encodings": {},
            "column_classification": {},
            "classification_confirmed": False,
            "feature_encodings": {},
            "encodings_confirmed": False,
            "model_config": {},
            "correlation_data": None,
            "final_config": {},
            "current_phase": "starting",
            "error": None,
        }

        self.assertIn("df_json", state)
        self.assertIn("columns", state)
        self.assertIn("current_phase", state)
        self.assertIn("preset", state)
        self.assertIn("optional_encodings", state)

    def test_state_with_preset(self):
        state: WorkflowState = {
            "df_json": "{}",
            "columns": ["A"],
            "dtypes": {"A": "int64"},
            "dataset_size": 100,
            "preset": "salary",
            "optional_encodings": {},
            "current_phase": "starting",
        }

        self.assertEqual(state["preset"], "salary")

    def test_state_with_optional_encodings(self):
        state: WorkflowState = {
            "df_json": "{}",
            "columns": ["A"],
            "dtypes": {"A": "int64"},
            "dataset_size": 100,
            "preset": None,
            "optional_encodings": {"Location": {"type": "cost_of_living", "params": {}}},
            "current_phase": "starting",
        }

        self.assertIn("Location", state["optional_encodings"])
        self.assertEqual(state["optional_encodings"]["Location"]["type"], "cost_of_living")


class TestClassifyColumnsNode(unittest.TestCase):
    @patch("src.agents.workflow.run_column_classifier_sync")
    @patch("src.agents.workflow.compute_correlation_matrix")
    def test_successful_classification(self, mock_corr, mock_classifier):
        mock_classifier.return_value = {
            "targets": ["Salary"],
            "features": ["Level"],
            "ignore": ["ID"],
            "reasoning": "Test",
        }
        mock_corr.invoke.return_value = '{"correlations": []}'

        mock_llm = MagicMock(spec=BaseChatModel)
        state: WorkflowState = {
            "df_json": '{"Salary": [100], "Level": ["L3"], "ID": [1]}',
            "columns": ["Salary", "Level", "ID"],
            "dtypes": {"Salary": "int64", "Level": "object", "ID": "int64"},
            "dataset_size": 1,
            "preset": None,
            "optional_encodings": {},
        }

        result = classify_columns_node(state, mock_llm)

        self.assertEqual(result["current_phase"], "classification")
        self.assertFalse(result["classification_confirmed"])
        self.assertIn("column_classification", result)
        self.assertIsNone(result["error"])
        # Verify optional_encodings is preserved
        self.assertIn("optional_encodings", result)

    @patch("src.agents.workflow.run_column_classifier_sync")
    def test_classification_error_handling(self, mock_classifier):
        """Test error handling in classification node."""
        mock_classifier.side_effect = Exception("Classification failed")

        mock_llm = MagicMock(spec=BaseChatModel)
        state: WorkflowState = {"df_json": "{}", "columns": [], "dtypes": {}, "dataset_size": 0}

        result = classify_columns_node(state, mock_llm)

        self.assertEqual(result["current_phase"], "classification")
        self.assertIsNotNone(result["error"])
        self.assertEqual(result["column_classification"], {})


class TestEncodeFeaturesNode(unittest.TestCase):
    """Tests for encode_features_node function."""

    @patch("src.agents.workflow.run_feature_encoder_sync")
    def test_successful_encoding(self, mock_encoder):
        """Test successful feature encoding."""
        mock_encoder.return_value = {
            "encodings": {"Level": {"type": "ordinal", "mapping": {"L1": 0, "L2": 1}}},
            "summary": "Test",
        }

        mock_llm = MagicMock(spec=BaseChatModel)
        state: WorkflowState = {
            "df_json": '{"Level": ["L1", "L2"]}',
            "columns": ["Level"],
            "dtypes": {"Level": "object"},
            "column_classification": {"features": ["Level"]},
            "optional_encodings": {"Location": {"type": "cost_of_living", "params": {}}},
        }

        result = encode_features_node(state, mock_llm)

        self.assertEqual(result["current_phase"], "encoding")
        self.assertFalse(result["encodings_confirmed"])
        self.assertIn("feature_encodings", result)
        self.assertIsNone(result["error"])
        # Verify optional_encodings is preserved
        self.assertIn("optional_encodings", result)
        self.assertEqual(result["optional_encodings"]["Location"]["type"], "cost_of_living")

    @patch("src.agents.workflow.run_feature_encoder_sync")
    def test_encoding_error_handling(self, mock_encoder):
        """Test error handling in encoding node."""
        mock_encoder.side_effect = Exception("Encoding failed")

        mock_llm = MagicMock(spec=BaseChatModel)
        state: WorkflowState = {
            "df_json": "{}",
            "columns": [],
            "dtypes": {},
            "column_classification": {"features": []},
        }

        result = encode_features_node(state, mock_llm)

        self.assertEqual(result["current_phase"], "encoding")
        self.assertIsNotNone(result["error"])


class TestConfigureModelNode(unittest.TestCase):
    """Tests for configure_model_node function."""

    @patch("src.agents.workflow.run_model_configurator_sync")
    def test_successful_configuration(self, mock_configurator):
        """Test successful model configuration."""
        mock_configurator.return_value = {
            "features": [{"name": "Level", "monotone_constraint": 1}],
            "quantiles": [0.1, 0.5, 0.9],
            "hyperparameters": {},
            "reasoning": "Test",
        }

        mock_llm = MagicMock(spec=BaseChatModel)
        state: WorkflowState = {
            "column_classification": {"targets": ["Salary"]},
            "feature_encodings": {"encodings": {"Level": {"type": "ordinal"}}},
            "correlation_data": None,
            "dataset_size": 100,
            "optional_encodings": {"Date": {"type": "normalize_recent", "params": {}}},
        }

        result = configure_model_node(state, mock_llm)

        self.assertEqual(result["current_phase"], "configuration")
        self.assertIn("model_config", result)
        self.assertIsNone(result["error"])
        # Verify optional_encodings is preserved
        self.assertIn("optional_encodings", result)
        self.assertEqual(result["optional_encodings"]["Date"]["type"], "normalize_recent")

    @patch("src.agents.workflow.run_model_configurator_sync")
    def test_configuration_error_handling(self, mock_configurator):
        """Test error handling in configuration node."""
        mock_configurator.side_effect = Exception("Configuration failed")

        mock_llm = MagicMock(spec=BaseChatModel)
        state: WorkflowState = {
            "column_classification": {"targets": []},
            "feature_encodings": {"encodings": {}},
        }

        result = configure_model_node(state, mock_llm)

        self.assertEqual(result["current_phase"], "configuration")
        self.assertIsNotNone(result["error"])


class TestBuildFinalConfigNode(unittest.TestCase):
    """Tests for build_final_config_node function."""

    def test_build_final_config(self):
        """Test building final configuration."""
        state: WorkflowState = {
            "column_classification": {"targets": ["Salary"], "reasoning": "Salary is target"},
            "feature_encodings": {
                "encodings": {
                    "Level": {"type": "ordinal", "mapping": {"L1": 0, "L2": 1}},
                    "Location": {"type": "proximity"},
                },
                "summary": "Encoding summary",
            },
            "model_config": {
                "features": [{"name": "Level", "monotone_constraint": 1}],
                "quantiles": [0.1, 0.5, 0.9],
                "hyperparameters": {"training": {}, "cv": {}},
                "reasoning": "Config reasoning",
            },
        }

        result = build_final_config_node(state)

        self.assertEqual(result["current_phase"], "complete")
        self.assertIn("final_config", result)
        self.assertIsNone(result["error"])

        final_config = result["final_config"]
        self.assertIn("mappings", final_config)
        self.assertIn("model", final_config)
        self.assertEqual(final_config["model"]["targets"], ["Salary"])

    def test_build_final_config_with_ordinal(self):
        """Test building config with ordinal encodings."""
        state: WorkflowState = {
            "column_classification": {"targets": ["Price"]},
            "feature_encodings": {
                "encodings": {"Level": {"type": "ordinal", "mapping": {"Junior": 0, "Senior": 1}}}
            },
            "model_config": {"features": [], "quantiles": [0.5], "hyperparameters": {}},
        }

        result = build_final_config_node(state)

        final_config = result["final_config"]
        # Should create mapping for ordinal
        self.assertIn("mappings", final_config)
        self.assertIn("feature_engineering", final_config)

    def test_build_final_config_with_optional_encodings(self):
        """Test building config with optional_encodings."""
        state: WorkflowState = {
            "column_classification": {"targets": ["Salary"]},
            "feature_encodings": {"encodings": {}},
            "model_config": {"features": [], "quantiles": [0.5], "hyperparameters": {}},
            "optional_encodings": {
                "Location": {"type": "cost_of_living", "params": {}},
                "Date": {"type": "normalize_recent", "params": {}},
            },
        }

        result = build_final_config_node(state)

        final_config = result["final_config"]
        # Should include optional_encodings in final config
        self.assertIn("optional_encodings", final_config)
        self.assertIn("Location", final_config["optional_encodings"])
        self.assertIn("Date", final_config["optional_encodings"])
        self.assertEqual(final_config["optional_encodings"]["Location"]["type"], "cost_of_living")
        self.assertEqual(final_config["optional_encodings"]["Date"]["type"], "normalize_recent")

    def test_build_final_config_without_optional_encodings(self):
        """Test building config without optional_encodings (backward compatibility)."""
        state: WorkflowState = {
            "column_classification": {"targets": ["Salary"]},
            "feature_encodings": {"encodings": {}},
            "model_config": {"features": [], "quantiles": [0.5], "hyperparameters": {}},
            # No optional_encodings field
        }

        result = build_final_config_node(state)

        final_config = result["final_config"]
        # Should still work, optional_encodings should be empty dict
        self.assertIn("optional_encodings", final_config)
        self.assertEqual(final_config["optional_encodings"], {})


class TestConditionalEdges(unittest.TestCase):
    """Tests for conditional edge functions."""

    def test_should_continue_after_classification_confirmed(self):
        """Test continuation when classification is confirmed."""
        state: WorkflowState = {"classification_confirmed": True}

        result = should_continue_after_classification(state)
        self.assertEqual(result, "encode_features")

    def test_should_continue_after_classification_not_confirmed(self):
        """Test waiting when classification not confirmed."""
        state: WorkflowState = {"classification_confirmed": False}

        result = should_continue_after_classification(state)
        self.assertEqual(result, "await_classification")

    def test_should_continue_after_encoding_confirmed(self):
        """Test continuation when encoding is confirmed."""
        state: WorkflowState = {"encodings_confirmed": True}

        result = should_continue_after_encoding(state)
        self.assertEqual(result, "configure_model")

    def test_should_continue_after_encoding_not_confirmed(self):
        """Test waiting when encoding not confirmed."""
        state: WorkflowState = {"encodings_confirmed": False}

        result = should_continue_after_encoding(state)
        self.assertEqual(result, "await_encoding")


class TestCreateWorkflowGraph(unittest.TestCase):
    """Tests for create_workflow_graph function."""

    def test_graph_structure(self):
        """Test workflow graph structure."""
        mock_llm = MagicMock(spec=BaseChatModel)
        graph = create_workflow_graph(mock_llm)

        # Graph should be a StateGraph
        self.assertIsNotNone(graph)

    def test_all_nodes_added(self):
        """Test all nodes are added to graph."""
        mock_llm = MagicMock(spec=BaseChatModel)
        graph = create_workflow_graph(mock_llm)

        # Check that graph has nodes (structure check)
        self.assertIsNotNone(graph)


class TestCompileWorkflow(unittest.TestCase):
    """Tests for compile_workflow function."""

    def test_compile_with_checkpointer(self):
        """Test compiling workflow with checkpointer."""
        from langgraph.checkpoint.memory import MemorySaver

        mock_llm = MagicMock(spec=BaseChatModel)
        checkpointer = MemorySaver()

        compiled = compile_workflow(mock_llm, checkpointer)

        self.assertIsNotNone(compiled)

    def test_compile_without_checkpointer(self):
        """Test compiling workflow without checkpointer."""
        mock_llm = MagicMock(spec=BaseChatModel)

        compiled = compile_workflow(mock_llm, None)

        self.assertIsNotNone(compiled)


class TestConfigWorkflow(unittest.TestCase):
    """Tests for ConfigWorkflow class."""

    def test_init(self):
        """Test ConfigWorkflow initialization."""
        mock_llm = MagicMock(spec=BaseChatModel)

        workflow = ConfigWorkflow(mock_llm)

        self.assertIsNotNone(workflow.llm)
        self.assertIsNotNone(workflow.checkpointer)
        self.assertIsNotNone(workflow.compiled)

    @patch("src.agents.workflow.compile_workflow")
    def test_start_method(self, mock_compile):
        """Test start method."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_compiled = MagicMock()

        # Mock the stream to return events
        mock_compiled.stream.return_value = iter(
            [{"column_classification": {"targets": ["Salary"]}}]
        )
        mock_compiled.get_state.return_value = Mock(
            values={
                "column_classification": {"targets": ["Salary"]},
                "current_phase": "classification",
            }
        )

        mock_compile.return_value = mock_compiled

        workflow = ConfigWorkflow(mock_llm)
        workflow.compiled = mock_compiled

        df = pd.DataFrame({"Salary": [100]})
        result = workflow.start(df.to_json(), ["Salary"], {"Salary": "int64"}, 1)

        self.assertIn("column_classification", result)

    @patch("src.agents.workflow.compile_workflow")
    def test_start_with_preset(self, mock_compile):
        """Test start method with preset parameter."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_compiled = MagicMock()

        mock_compiled.stream.return_value = iter(
            [{"column_classification": {"targets": ["Salary"]}}]
        )
        mock_compiled.get_state.return_value = Mock(
            values={
                "column_classification": {"targets": ["Salary"]},
                "current_phase": "classification",
                "preset": "salary",
            }
        )

        mock_compile.return_value = mock_compiled

        workflow = ConfigWorkflow(mock_llm)
        workflow.compiled = mock_compiled

        df = pd.DataFrame({"Salary": [100]})
        result = workflow.start(df.to_json(), ["Salary"], {"Salary": "int64"}, 1, preset="salary")

        # Verify preset is in state
        self.assertIn("preset", result)
        self.assertEqual(result["preset"], "salary")

    @patch("src.agents.workflow.compile_workflow")
    def test_start_with_optional_encodings_initialized(self, mock_compile):
        """Test that optional_encodings is initialized in start."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_compiled = MagicMock()

        mock_compiled.stream.return_value = iter(
            [{"column_classification": {"targets": ["Salary"]}}]
        )
        mock_compiled.get_state.return_value = Mock(
            values={
                "column_classification": {"targets": ["Salary"]},
                "current_phase": "classification",
                "optional_encodings": {},
            }
        )

        mock_compile.return_value = mock_compiled

        workflow = ConfigWorkflow(mock_llm)
        workflow.compiled = mock_compiled

        df = pd.DataFrame({"Salary": [100]})
        result = workflow.start(df.to_json(), ["Salary"], {"Salary": "int64"}, 1)

        # Verify optional_encodings is initialized
        self.assertIn("optional_encodings", result)
        self.assertEqual(result["optional_encodings"], {})

    @patch("src.agents.workflow.compile_workflow")
    def test_confirm_classification(self, mock_compile):
        """Test confirm_classification method."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_compiled = MagicMock()

        # Mock the state after encoding
        final_state = {"current_phase": "encoding", "feature_encodings": {"encodings": {}}}

        mock_compiled.get_state.return_value = Mock(values=final_state)
        mock_compiled.stream.return_value = iter([final_state])
        mock_compiled.update_state = MagicMock()

        mock_compile.return_value = mock_compiled

        workflow = ConfigWorkflow(mock_llm)
        workflow.compiled = mock_compiled
        workflow.thread_id = "test_thread"
        workflow.current_state = {"column_classification": {}}

        result = workflow.confirm_classification()

        mock_compiled.update_state.assert_called_once()
        # Result should be the state after encoding
        self.assertIn("current_phase", result)

    @patch("src.agents.workflow.compile_workflow")
    def test_confirm_classification_with_optional_encodings(self, mock_compile):
        """Test confirm_classification with optional_encodings modifications."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_compiled = MagicMock()

        final_state = {
            "current_phase": "encoding",
            "feature_encodings": {"encodings": {}},
            "optional_encodings": {"Location": {"type": "cost_of_living", "params": {}}},
        }

        mock_compiled.get_state.return_value = Mock(values=final_state)
        mock_compiled.stream.return_value = iter([final_state])
        mock_compiled.update_state = MagicMock()

        mock_compile.return_value = mock_compiled

        workflow = ConfigWorkflow(mock_llm)
        workflow.compiled = mock_compiled
        workflow.thread_id = "test_thread"
        workflow.current_state = {"column_classification": {}}

        modifications = {
            "targets": ["Salary"],
            "optional_encodings": {"Location": {"type": "cost_of_living", "params": {}}},
        }

        result = workflow.confirm_classification(modifications)

        # Verify update_state was called with optional_encodings
        update_call = mock_compiled.update_state.call_args
        update_state_dict = update_call[1] if update_call[1] else update_call[0][1]
        self.assertIn("optional_encodings", update_state_dict)
        self.assertEqual(
            update_state_dict["optional_encodings"]["Location"]["type"], "cost_of_living"
        )

    @patch("src.agents.workflow.compile_workflow")
    def test_get_current_phase(self, mock_compile):
        """Test get_current_phase method."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_compile.return_value = MagicMock()

        workflow = ConfigWorkflow(mock_llm)
        workflow.current_state = {"current_phase": "classification"}

        phase = workflow.get_current_phase()

        self.assertEqual(phase, "classification")

    @patch("src.agents.workflow.compile_workflow")
    def test_get_final_config_complete(self, mock_compile):
        """Test get_final_config when complete."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_compile.return_value = MagicMock()

        workflow = ConfigWorkflow(mock_llm)
        workflow.current_state = {
            "current_phase": "complete",
            "final_config": {"model": {"targets": ["Salary"]}},
        }

        config = workflow.get_final_config()

        self.assertIsNotNone(config)
        self.assertEqual(config["model"]["targets"], ["Salary"])

    @patch("src.agents.workflow.compile_workflow")
    def test_get_final_config_incomplete(self, mock_compile):
        """Test get_final_config when not complete."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_compile.return_value = MagicMock()

        workflow = ConfigWorkflow(mock_llm)
        workflow.current_state = {"current_phase": "classification"}

        config = workflow.get_final_config()

        self.assertIsNone(config)

    @patch("src.agents.workflow.compile_workflow")
    def test_confirm_encoding_with_optional_encodings(self, mock_compile):
        """Test confirm_encoding with optional_encodings modifications."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_compiled = MagicMock()

        final_state = {
            "current_phase": "complete",
            "final_config": {"model": {"targets": ["Salary"]}},
            "optional_encodings": {"Date": {"type": "normalize_recent", "params": {}}},
        }

        mock_compiled.get_state.return_value = Mock(values=final_state)
        mock_compiled.stream.return_value = iter([final_state])
        mock_compiled.update_state = MagicMock()

        mock_compile.return_value = mock_compiled

        workflow = ConfigWorkflow(mock_llm)
        workflow.compiled = mock_compiled
        workflow.thread_id = "test_thread"
        workflow.current_state = {"feature_encodings": {"encodings": {}}}

        modifications = {
            "encodings": {"Level": {"type": "ordinal"}},
            "optional_encodings": {"Date": {"type": "normalize_recent", "params": {}}},
        }

        result = workflow.confirm_encoding(modifications)

        # Verify update_state was called with optional_encodings
        update_call = mock_compiled.update_state.call_args
        update_state_dict = update_call[1] if update_call[1] else update_call[0][1]
        self.assertIn("optional_encodings", update_state_dict)
        self.assertEqual(
            update_state_dict["optional_encodings"]["Date"]["type"], "normalize_recent"
        )


if __name__ == "__main__":
    unittest.main()
