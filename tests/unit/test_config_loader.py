import pytest
import json
from unittest.mock import patch, mock_open
from src.utils.config_loader import load_config, get_config
import src.utils.config_loader as config_loader_module

def test_load_config_success():
    mock_data = {
        "mappings": {},
        "location_settings": {},
        "model": {
            "targets": [],
            "quantiles": []
        }
    }
    mock_json = json.dumps(mock_data)
    
    with patch("builtins.open", mock_open(read_data=mock_json)), \
         patch("os.path.exists", return_value=True):
        
        config_loader_module._CONFIG = None
        
        config = load_config()
        assert config == mock_data
        assert config_loader_module._CONFIG == mock_data

def test_get_config_singleton():
    """Verify singleton pattern prevents redundant file reads."""
    mock_data = {"key": "value"}
    
    config_loader_module._CONFIG = mock_data
    
    with patch("src.utils.config_loader.load_config") as mock_load:
        config = get_config()
        assert config == mock_data
        mock_load.assert_not_called()

def test_load_config_file_not_found():
    """Verify proper error handling when config file is missing."""
    with patch("os.path.exists", return_value=False):
        config_loader_module._CONFIG = None
        
        with pytest.raises(FileNotFoundError):
            load_config()


def test_load_config_invalid_json():
    """Verify proper error handling for malformed JSON files."""
    invalid_json = "{ invalid json }"
    
    with patch("builtins.open", mock_open(read_data=invalid_json)), \
         patch("os.path.exists", return_value=True):
        
        config_loader_module._CONFIG = None
        
        with pytest.raises(json.JSONDecodeError):
            load_config()


def test_load_config_missing_required_keys():
    """Verify validation catches incomplete configuration files."""
    incomplete_config = {
        "mappings": {}
    }
    mock_json = json.dumps(incomplete_config)
    
    with patch("builtins.open", mock_open(read_data=mock_json)), \
         patch("os.path.exists", return_value=True):
        
        config_loader_module._CONFIG = None
        
        with pytest.raises(ValueError) as exc_info:
            load_config()
        
        assert "missing required keys" in str(exc_info.value).lower() or "required" in str(exc_info.value).lower()


def test_get_config_fallback_to_load():
    """Verify get_config automatically loads config on first access."""
    mock_data = {
        "mappings": {},
        "location_settings": {},
        "model": {"targets": [], "quantiles": []}
    }
    mock_json = json.dumps(mock_data)
    
    with patch("builtins.open", mock_open(read_data=mock_json)), \
         patch("os.path.exists", return_value=True):
        
        config_loader_module._CONFIG = None
        
        config = get_config()
        assert config == mock_data
