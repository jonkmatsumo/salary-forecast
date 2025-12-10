import pytest
import unittest
from pydantic import ValidationError
from src.model.config_schema_model import Config, ModelConfig, FeatureConfig

def test_valid_config():
    data = {
        "model": {
            "targets": ["Salary"],
            "features": [{"name": "Exp", "monotone_constraint": 1}]
        }
    }
    config = Config(**data)
    assert config.model.targets == ["Salary"]
    assert config.model.features[0].name == "Exp"

def test_duplicate_features_error():
    data = {
        "model": {
            "targets": ["Salary"],
            "features": [
                {"name": "Exp", "monotone_constraint": 1},
                {"name": "Exp", "monotone_constraint": 0}
            ]
        }
    }
    with pytest.raises(ValidationError) as exc:
        Config(**data)
    assert "Feature names must be unique" in str(exc.value)

def test_invalid_constraint_value():
    data = {
        "model": {
            "features": [{"name": "Exp", "monotone_constraint": 5}]
        }
    }
    with pytest.raises(ValidationError):
        Config(**data)

def test_optional_encodings_field():
    data = {
        "model": {
            "targets": ["Salary"],
            "features": [{"name": "Exp", "monotone_constraint": 1}]
        },
        "optional_encodings": {
            "Location": {"type": "cost_of_living", "params": {}},
            "Date": {"type": "normalize_recent", "params": {}}
        }
    }
    config = Config(**data)
    assert "Location" in config.optional_encodings
    assert config.optional_encodings["Location"]["type"] == "cost_of_living"
    assert "Date" in config.optional_encodings
    assert config.optional_encodings["Date"]["type"] == "normalize_recent"

def test_optional_encodings_backward_compatibility():
    data = {
        "model": {
            "targets": ["Salary"],
            "features": [{"name": "Exp", "monotone_constraint": 1}]
        }
    }
    config = Config(**data)
    assert config.optional_encodings == {}


class TestBackwardCompatibility(unittest.TestCase):
    """Tests for backward compatibility."""
    
    def test_config_without_optional_encodings(self):
        """Test that configs without optional_encodings field work."""
        data = {
            "model": {
                "targets": ["Salary"],
                "features": [{"name": "Exp", "monotone_constraint": 1}]
            }
        }
        config = Config(**data)
        # Should default to empty dict
        self.assertEqual(config.optional_encodings, {})
    
    def test_workflow_state_without_preset(self):
        """Test that workflow state works without preset."""
        from src.agents.workflow import WorkflowState
        
        state: WorkflowState = {
            "df_json": '{}',
            "columns": ["A"],
            "dtypes": {"A": "int64"},
            "dataset_size": 100,
            "current_phase": "starting"
            # No preset field
        }
        
        # Should work fine (preset is optional)
        self.assertIn("columns", state)
    
    def test_workflow_state_without_optional_encodings(self):
        """Test that workflow state works without optional_encodings."""
        from src.agents.workflow import WorkflowState
        
        state: WorkflowState = {
            "df_json": '{}',
            "columns": ["A"],
            "dtypes": {"A": "int64"},
            "dataset_size": 100,
            "current_phase": "starting"
            # No optional_encodings field
        }
        
        # Should work fine (optional_encodings is optional)
        self.assertIn("columns", state)
