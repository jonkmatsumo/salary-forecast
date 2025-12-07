import json
import os
from typing import Optional, Dict, Any

_CONFIG: Optional[Dict[str, Any]] = None

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Loads the configuration from a JSON file.
    """
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
        
    # If path is relative, assume it's relative to the project root (where this script might be running from)
    # or relative to this file's location if we want to be more robust.
    # For now, let's assume it's in the project root.
    
    if not os.path.exists(config_path):
        # Try finding it relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config.json")
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)
        
    _validate_config(config)
    _CONFIG = config
        
    return _CONFIG

def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validates the configuration dictionary structure.
    
    Args:
        config (dict): The configuration dictionary.
        
    Raises:
        ValueError: If required keys are missing.
    """
    required_keys = ["mappings", "location_settings", "model"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config missing required key: '{key}'")
            
    # Validate model section
    model_config = config["model"]
    required_model_keys = ["targets", "quantiles"]
    for key in required_model_keys:
        if key not in model_config:
            raise ValueError(f"Config['model'] missing required key: '{key}'")

def get_config() -> Dict[str, Any]:
    """
    Returns the loaded configuration. Loads it if not already loaded.
    """
    if _CONFIG is None:
        return load_config()
    return _CONFIG
