"""Pytest configuration and shared test fixtures."""

from typing import Any, Dict

import pytest

from src.model.config_schema_model import Config


def create_test_config() -> Dict[str, Any]:
    """Create a minimal valid test configuration.

    Returns:
        Dict[str, Any]: Test config.
    """
    return {
        "mappings": {
            "levels": {"E3": 0, "E4": 1, "E5": 2},
            "location_targets": {"New York, NY": 1, "San Francisco, CA": 1},
        },
        "location_settings": {"max_distance_km": 50.0},
        "model": {
            "targets": ["BaseSalary"],
            "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
            "sample_weight_k": 1.0,
            "features": [
                {"name": "Level_Enc", "monotone_constraint": 1},
                {"name": "Location_Enc", "monotone_constraint": 0},
            ],
        },
        "optional_encodings": {},
    }


def create_validated_test_config() -> Config:
    """Create a validated Pydantic Config object.

    Returns:
        Config: Validated config.
    """
    return Config.model_validate(create_test_config())


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Pytest fixture providing a test configuration.

    Returns:
        Dict[str, Any]: Test config.
    """
    return create_test_config()


@pytest.fixture
def validated_test_config() -> Config:
    """Pytest fixture providing a validated Config object.

    Returns:
        Config: Validated config.
    """
    return create_validated_test_config()
