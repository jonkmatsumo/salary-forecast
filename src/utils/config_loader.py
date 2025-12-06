import json
import os

_CONFIG = None

def load_config(config_path="config.json"):
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
        _CONFIG = json.load(f)
        
    return _CONFIG

def get_config():
    """
    Returns the loaded configuration. Loads it if not already loaded.
    """
    if _CONFIG is None:
        return load_config()
    return _CONFIG
