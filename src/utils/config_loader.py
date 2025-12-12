import json
import os
from typing import Any, Dict, Optional

from src.model.config_schema_model import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CONFIG: Optional[Dict[str, Any]] = None


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load the configuration from a JSON file. Args: config_path (str): Config file path. Returns: Dict[str, Any]: Loaded configuration. Raises: FileNotFoundError: If config file not found."""
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    if not os.path.exists(config_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    try:
        # Validate using Pydantic model
        config_model = Config(**config_dict)
        config = config_model.model_dump()
        logger.debug("Config validated successfully using Pydantic model")
    except Exception as e:
        # Fallback to basic validation for backward compatibility
        logger.warning(f"Pydantic validation failed, using basic validation: {e}")
        _validate_config(config_dict)
        config = config_dict

    _CONFIG = config

    return _CONFIG


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate the configuration dictionary structure. Args: config (Dict[str, Any]): Configuration dictionary. Returns: None. Raises: ValueError: If required keys are missing."""
    required_keys = ["mappings", "location_settings", "model"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Config missing required key: '{key}'")

    model_config = config["model"]
    required_model_keys = ["targets", "quantiles"]
    for key in required_model_keys:
        if key not in model_config:
            raise ValueError(f"Config['model'] missing required key: '{key}'")


def get_config() -> Dict[str, Any]:
    """Return the loaded configuration. Loads it if not already loaded. Returns: Dict[str, Any]: Configuration dictionary."""
    if _CONFIG is None:
        return load_config()
    return _CONFIG
