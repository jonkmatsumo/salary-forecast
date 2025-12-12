import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieve an environment variable. Args: key (str): Environment variable name. default (Optional[str]): Default value. Returns: Optional[str]: Environment variable value."""
    return os.getenv(key, default)
