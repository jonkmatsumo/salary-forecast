import pytest
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
