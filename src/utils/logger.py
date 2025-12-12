import contextvars
import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional

try:
    from pythonjsonlogger import jsonlogger
except ImportError:
    jsonlogger = None

request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)


class RequestIDFilter(logging.Filter):
    """Filter to add request ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        request_id = request_id_var.get()
        record.request_id = request_id if request_id else "none"
        return True


class RequestTracingContext:
    """Context manager for request tracing with correlation IDs."""

    def __init__(self, request_id: Optional[str] = None):
        self.request_id = request_id
        self.token: Optional[contextvars.Token] = None

    def __enter__(self) -> "RequestTracingContext":
        if self.request_id:
            self.token = request_id_var.set(self.request_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.token:
            request_id_var.reset(self.token)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    json_format: bool = False,
    module_levels: Optional[Dict[str, int]] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> None:
    """Configure the root logger with enhanced formatting and handlers. Args: level (int): Default logging level. log_file (Optional[str]): Log file path. json_format (bool): Use JSON format for structured logging. module_levels (Optional[Dict[str, int]]): Per-module log levels. max_bytes (int): Max log file size before rotation. backup_count (int): Number of backup files to keep. Returns: None."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    request_id_filter = RequestIDFilter()

    if json_format and jsonlogger:
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s %(request_id)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        if json_format and not jsonlogger:
            logging.warning("python-json-logger not installed, using standard format")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [request_id=%(request_id)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(request_id_filter)
    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(request_id_filter)
        root_logger.addHandler(file_handler)

    if module_levels:
        for module_name, module_level in module_levels.items():
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(module_level)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the specified name. Args: name (str): Logger name. Returns: logging.Logger: Logger instance."""
    return logging.getLogger(name)


def get_request_id() -> Optional[str]:
    """Get the current request ID from context. Returns: Optional[str]: Current request ID."""
    return request_id_var.get()


def set_request_id(request_id: str) -> None:
    """Set the request ID in the current context. Args: request_id (str): Request ID to set. Returns: None."""
    request_id_var.set(request_id)
