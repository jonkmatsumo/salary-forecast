import logging
import sys
from typing import Optional

def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Configure the root logger with standard formatting and handlers. Args: level (int): Logging level. log_file (Optional[str]): Log file path. Returns: None."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True # Reconfigure if already configured
    )

def get_logger(name: str) -> logging.Logger:
    """Return a logger with the specified name. Args: name (str): Logger name. Returns: logging.Logger: Logger instance."""
    return logging.getLogger(name)
