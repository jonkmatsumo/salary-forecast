"""Tests for LangGraph workflow."""

import unittest
import json
from unittest.mock import MagicMock, patch, Mock
import pandas as pd
from langchain_core.language_models import BaseChatModel

from src.agents.workflow import (
    WorkflowState,
    classify_columns_node,
    encode_features_node,
    configure_model_node,
    build_final_config_node,
    should_continue_after_classification,
    should_continue_after_encoding,
    create_workflow_graph,
    compile_workflow,
    ConfigWorkflow,
)


class TestWorkflowState(unittest.TestCase):
    """Tests for WorkflowState TypedDict."""
    
    def test_state_structure(self):
        """Test WorkflowState structure."""
        state: WorkflowState = {
            "df_json": '{}',
            "columns": ["A"],
            "dtypes": {"A": "int64"},
            "dataset_size": 100,
            "column_classification": {},
            "classification_confirmed": False,
            "feature_encodings": {},
            "encodings_confirmed": False,
            "model_config": {},
            "correlation_data": None,
            "final_config": {},
            "current_phase": "starting",
            "error": None
        }
        
        self.assertIn("df_json", state)
        self.assertIn("columns", state)
        self.assertIn("current_phase", state)


class TestClassifyColumnsNode(unittest.TestCase):
    """Tests for classify_columns_node function."""
    
    @patch("src.agents.workflow.run_column_classifier_sync")
    @patch("src.agents.workflow.compute_correlation_matrix")
    def test_successful_classification(self, mock_corr, mock_classifier):
        """Test successful column classification."""
        mock_classifier.return_value = {
            "targets": ["Salary"],
            "features": ["Level"],
            "ignore": ["ID"],
            "reasoning": "Test"
        }
        mock_corr.invoke.return_value = '{"correlations": []}'
        
        mock_llm = MagicMock(spec=BaseChatModel)
        state: WorkflowState = {
            "df_json": '{"Salary": [100], "Level": ["L3"], "ID": [1]}',
            "columns": ["Salary", "Level", "ID"],
            "dtypes": {"Salary": "int64", "Level": "object", "ID": "int64"},
            "dataset_size": 1
        }
        
        result = classify_columns_node(state, mock_llm)
        
        self.assertEqual(result["current_phase"], "classification")
        self.assertFalse(result["classification_confirmed"])
        self.assertIn("column_classification", result)
        self.assertIsNone(result["error"])
    
    @patch("src.agents.workflow.run_column_classifier_sync")
    def test_classification_error_handling(self, mock_classifier):
        """Test error handling in classification node."""
        mock_classifier.side_effect = Exception("Classification failed")
        
        mock_llm = MagicMock(spec=BaseChatModel)
        state: WorkflowState = {
            "df_json": '{}',
            "columns": [],
            "dtypes": {},
            "dataset_size": 0
        }
        
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
            "encodings": {
                "Level": {"type": "ordinal", "mapping": {"L1": 0, "L2": 1}}
            },
            "summary": "Test"
        }
        
        mock_llm = MagicMock(spec=BaseChatModel)
        state: WorkflowState = {
            "df_json": '{"Level": ["L1", "L2"]}',
            "columns": ["Level"],
            "dtypes": {"Level": "object"},
            "column_classification": {
                "features": ["Level"]
            }
        }
        
        result = encode_features_node(state, mock_llm)
        
        self.assertEqual(result["current_phase"], "encoding")
        self.assertFalse(result["encodings_confirmed"])
        self.assertIn("feature_encodings", result)
        self.assertIsNone(result["error"])
    
    @patch("src.agents.workflow.run_feature_encoder_sync")
    def test_encoding_error_handling(self, mock_encoder):
        """Test error handling in encoding node."""
        mock_encoder.side_effect = Exception("Encoding failed")
        
        mock_llm = MagicMock(spec=BaseChatModel)
        state: WorkflowState = {
            "df_json": '{}',
            "columns": [],
            "dtypes": {},
            "column_classification": {"features": []}
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
            "reasoning": "Test"
        }
        
        mock_llm = MagicMock(spec=BaseChatModel)
        state: WorkflowState = {
            "column_classification": {"targets": ["Salary"]},
            "feature_encodings": {"encodings": {"Level": {"type": "ordinal"}}},
            "correlation_data": None,
            "dataset_size": 100
        }
        
        result = configure_model_node(state, mock_llm)
        
        self.assertEqual(result["current_phase"], "configuration")
        self.assertIn("model_config", result)
        self.assertIsNone(result["error"])
    
    @patch("src.agents.workflow.run_model_configurator_sync")
    def test_configuration_error_handling(self, mock_configurator):
        """Test error handling in configuration node."""
        mock_configurator.side_effect = Exception("Configuration failed")
        
        mock_llm = MagicMock(spec=BaseChatModel)
        state: WorkflowState = {
            "column_classification": {"targets": []},
            "feature_encodings": {"encodings": {}}
        }
        
        result = configure_model_node(state, mock_llm)
        
        self.assertEqual(result["current_phase"], "configuration")
        self.assertIsNotNone(result["error"])


class TestBuildFinalConfigNode(unittest.TestCase):
    """Tests for build_final_config_node function."""
    
    def test_build_final_config(self):
        """Test building final configuration."""
        state: WorkflowState = {
            "column_classification": {
                "targets": ["Salary"],
                "reasoning": "Salary is target"
            },
            "feature_encodings": {
                "encodings": {
                    "Level": {
                        "type": "ordinal",
                        "mapping": {"L1": 0, "L2": 1}
                    },
                    "Location": {
                        "type": "proximity"
                    }
                },
                "summary": "Encoding summary"
            },
            "model_config": {
                "features": [
                    {"name": "Level", "monotone_constraint": 1}
                ],
                "quantiles": [0.1, 0.5, 0.9],
                "hyperparameters": {
                    "training": {},
                    "cv": {}
                },
                "reasoning": "Config reasoning"
            }
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
                "encodings": {
                    "Level": {
                        "type": "ordinal",
                        "mapping": {"Junior": 0, "Senior": 1}
                    }
                }
            },
            "model_config": {
                "features": [],
                "quantiles": [0.5],
                "hyperparameters": {}
            }
        }
        
        result = build_final_config_node(state)
        
        final_config = result["final_config"]
        # Should create mapping for ordinal
        self.assertIn("mappings", final_config)
        self.assertIn("feature_engineering", final_config)


class TestConditionalEdges(unittest.TestCase):
    """Tests for conditional edge functions."""
    
    def test_should_continue_after_classification_confirmed(self):
        """Test continuation when classification is confirmed."""
        state: WorkflowState = {
            "classification_confirmed": True
        }
        
        result = should_continue_after_classification(state)
        self.assertEqual(result, "encode_features")
    
    def test_should_continue_after_classification_not_confirmed(self):
        """Test waiting when classification not confirmed."""
        state: WorkflowState = {
            "classification_confirmed": False
        }
        
        result = should_continue_after_classification(state)
        self.assertEqual(result, "await_classification")
    
    def test_should_continue_after_encoding_confirmed(self):
        """Test continuation when encoding is confirmed."""
        state: WorkflowState = {
            "encodings_confirmed": True
        }
        
        result = should_continue_after_encoding(state)
        self.assertEqual(result, "configure_model")
    
    def test_should_continue_after_encoding_not_confirmed(self):
        """Test waiting when encoding not confirmed."""
        state: WorkflowState = {
            "encodings_confirmed": False
        }
        
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
        mock_compiled.stream.return_value = iter([
            {"column_classification": {"targets": ["Salary"]}}
        ])
        mock_compiled.get_state.return_value = Mock(values={
            "column_classification": {"targets": ["Salary"]},
            "current_phase": "classification"
        })
        
        mock_compile.return_value = mock_compiled
        
        workflow = ConfigWorkflow(mock_llm)
        workflow.compiled = mock_compiled
        
        df = pd.DataFrame({"Salary": [100]})
        result = workflow.start(
            df.to_json(),
            ["Salary"],
            {"Salary": "int64"},
            1
        )
        
        self.assertIn("column_classification", result)
    
    @patch("src.agents.workflow.compile_workflow")
    def test_confirm_classification(self, mock_compile):
        """Test confirm_classification method."""
        mock_llm = MagicMock(spec=BaseChatModel)
        mock_compiled = MagicMock()
        
        # Mock the state after encoding
        final_state = {
            "current_phase": "encoding",
            "feature_encodings": {"encodings": {}}
        }
        
        mock_compiled.get_state.return_value = Mock(values=final_state)
        mock_compiled.stream.return_value = iter([
            final_state
        ])
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
            "final_config": {"model": {"targets": ["Salary"]}}
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


if __name__ == "__main__":
    unittest.main()

