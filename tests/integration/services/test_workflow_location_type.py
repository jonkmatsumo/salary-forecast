"""Integration tests for location type workflow."""
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.agents.workflow import ConfigWorkflow
from src.services.workflow_service import WorkflowService


@pytest.fixture
def sample_df_with_location():
    """Create a sample DataFrame with location column."""
    return pd.DataFrame({
        "Location": ["New York", "Austin", "Seattle"],
        "Salary": [100000, 90000, 95000],
        "Level": ["E4", "E3", "E4"]
    })


def test_location_type_detected_in_classification(sample_df_with_location):
    """Test that location type is detected during classification."""
    with patch("src.services.workflow_service.get_langchain_llm") as mock_llm_getter:
        mock_llm = MagicMock()
        mock_llm_getter.return_value = mock_llm
        
        # Mock the column classifier to return location type
        with patch("src.agents.workflow.run_column_classifier_sync") as mock_classify:
            mock_classify.return_value = {
                "targets": ["Salary"],
                "features": ["Location", "Level"],
                "ignore": [],
                "column_types": {
                    "Location": "location"
                },
                "reasoning": "Location detected as geographic data"
            }
            
            with patch("src.agents.workflow.compute_correlation_matrix") as mock_corr:
                mock_corr.invoke.return_value = "correlation_data"
                
                service = WorkflowService(provider="openai")
                result = service.start_workflow(sample_df_with_location)
                
                # Verify location was detected
                assert service.workflow.current_state.get("column_types", {}).get("Location") == "location"
                assert "Location" in service.workflow.current_state.get("location_columns", [])


def test_location_can_be_target_or_feature(sample_df_with_location):
    """Test that location columns can be assigned as target or feature."""
    with patch("src.services.workflow_service.get_langchain_llm") as mock_llm_getter:
        mock_llm = MagicMock()
        mock_llm_getter.return_value = mock_llm
        
        # Test location as feature
        with patch("src.agents.workflow.run_column_classifier_sync") as mock_classify:
            mock_classify.return_value = {
                "targets": ["Salary"],
                "features": ["Location"],
                "ignore": [],
                "column_types": {"Location": "location"},
                "reasoning": "Location as feature"
            }
            
            with patch("src.agents.workflow.compute_correlation_matrix") as mock_corr:
                mock_corr.invoke.return_value = "correlation_data"
                
                service = WorkflowService(provider="openai")
                result = service.start_workflow(sample_df_with_location)
                
                classification = service.workflow.current_state.get("column_classification", {})
                assert "Location" in classification.get("features", [])
                assert "Location" not in classification.get("targets", [])
                assert classification.get("column_types", {}).get("Location") == "location"
        
        # Test location as target
        with patch("src.agents.workflow.run_column_classifier_sync") as mock_classify:
            mock_classify.return_value = {
                "targets": ["Location"],
                "features": ["Level"],
                "ignore": [],
                "column_types": {"Location": "location"},
                "reasoning": "Location as target"
            }
            
            with patch("src.agents.workflow.compute_correlation_matrix") as mock_corr:
                mock_corr.invoke.return_value = "correlation_data"
                
                service = WorkflowService(provider="openai")
                result = service.start_workflow(sample_df_with_location)
                
                classification = service.workflow.current_state.get("column_classification", {})
                assert "Location" in classification.get("targets", [])
                assert "Location" not in classification.get("features", [])
                assert classification.get("column_types", {}).get("Location") == "location"


def test_location_columns_extracted_from_column_types(sample_df_with_location):
    """Test that location columns are extracted from column_types."""
    with patch("src.services.workflow_service.get_langchain_llm") as mock_llm_getter:
        mock_llm = MagicMock()
        mock_llm_getter.return_value = mock_llm
        
        with patch("src.agents.workflow.run_column_classifier_sync") as mock_classify:
            mock_classify.return_value = {
                "targets": ["Salary"],
                "features": ["Location", "City"],
                "ignore": [],
                "column_types": {
                    "Location": "location",
                    "City": "location"
                },
                "reasoning": "Multiple location columns"
            }
            
            with patch("src.agents.workflow.compute_correlation_matrix") as mock_corr:
                mock_corr.invoke.return_value = "correlation_data"
                
                service = WorkflowService(provider="openai")
                result = service.start_workflow(sample_df_with_location)
                
                location_columns = service.workflow.current_state.get("location_columns", [])
                assert "Location" in location_columns
                assert "City" in location_columns
                assert len(location_columns) == 2

