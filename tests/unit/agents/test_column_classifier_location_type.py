"""Tests for location type detection and assignment."""
import pytest
from unittest.mock import MagicMock, patch
import json
from src.agents.column_classifier import parse_classification_response


def test_parse_classification_with_column_types():
    """Test parsing classification response with column_types."""
    response = """{
        "targets": ["Salary"],
        "features": ["Location", "Level"],
        "ignore": [],
        "column_types": {
            "Location": "location"
        },
        "reasoning": "Test reasoning"
    }"""
    
    result = parse_classification_response(response)
    
    assert "targets" in result
    assert "features" in result
    assert "Location" in result["features"]
    assert "column_types" in result
    assert result["column_types"]["Location"] == "location"
    assert "locations" not in result or len(result.get("locations", [])) == 0


def test_parse_classification_backward_compatibility():
    """Test backward compatibility with old locations key."""
    response = """{
        "targets": ["Salary"],
        "features": ["Level"],
        "locations": ["Location"],
        "ignore": [],
        "reasoning": "Test reasoning"
    }"""
    
    result = parse_classification_response(response)
    
    assert "column_types" in result
    assert result["column_types"]["Location"] == "location"
    # locations key should be removed after migration
    assert "locations" not in result


def test_location_can_be_target():
    """Test that location columns can be assigned as targets."""
    response = """{
        "targets": ["Location"],
        "features": ["Level"],
        "ignore": [],
        "column_types": {
            "Location": "location"
        },
        "reasoning": "Location is a target"
    }"""
    
    result = parse_classification_response(response)
    
    assert "Location" in result["targets"]
    assert result["column_types"]["Location"] == "location"


def test_location_can_be_feature():
    """Test that location columns can be assigned as features."""
    response = """{
        "targets": ["Salary"],
        "features": ["Location"],
        "ignore": [],
        "column_types": {
            "Location": "location"
        },
        "reasoning": "Location is a feature"
    }"""
    
    result = parse_classification_response(response)
    
    assert "Location" in result["features"]
    assert result["column_types"]["Location"] == "location"


def test_multiple_location_types():
    """Test handling multiple location columns."""
    response = """{
        "targets": ["Salary"],
        "features": ["City", "Region"],
        "ignore": [],
        "column_types": {
            "City": "location",
            "Region": "location"
        },
        "reasoning": "Multiple location columns"
    }"""
    
    result = parse_classification_response(response)
    
    assert result["column_types"]["City"] == "location"
    assert result["column_types"]["Region"] == "location"
    assert "City" in result["features"]
    assert "Region" in result["features"]

