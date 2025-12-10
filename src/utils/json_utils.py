"""
JSON parsing utilities for handling escaped and normalized JSON strings.

Provides defensive parsing for JSON strings that may be double-encoded
or contain escape sequences from LLM tool calls.
"""

import json
from typing import Any, Dict, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


def normalize_json_string(json_str: str, max_depth: int = 5) -> Any:
    """
    Normalize and parse a JSON string that may be escaped or double-encoded.
    
    Attempts multiple parsing strategies and recursively parses until a non-string
    result is obtained (handles double/triple encoding).
    
    Args:
        json_str (str): JSON string that may be escaped or encoded.
        max_depth (int): Maximum recursion depth to prevent infinite loops.
        
    Returns:
        Any: Parsed JSON object (dict, list, or primitive type).
        
    Raises:
        ValueError: If all parsing attempts fail, with detailed error message.
    """
    if not json_str or not isinstance(json_str, str):
        raise ValueError(f"Invalid input: expected non-empty string, got {type(json_str)}")
    
    original_str = json_str
    strategies = []
    
    def _parse_once(s: str) -> Any:
        """Try all parsing strategies once."""
        # Strategy 1: Direct parse
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Remove outer quotes if present
        stripped = s.strip()
        if (stripped.startswith('"') and stripped.endswith('"')) or \
           (stripped.startswith("'") and stripped.endswith("'")):
            try:
                unquoted = stripped[1:-1]
                return json.loads(unquoted)
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Unescape common escape sequences
        try:
            unescaped = s.encode().decode('unicode_escape')
            return json.loads(unescaped)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        
        # Strategy 4: Replace escaped quotes manually
        try:
            manual_unescaped = s.replace('\\"', '"').replace('\\\\', '\\')
            return json.loads(manual_unescaped)
        except json.JSONDecodeError:
            pass
        
        # Strategy 5: Remove outer quotes then unescape
        if (stripped.startswith('"') and stripped.endswith('"')):
            try:
                unquoted = stripped[1:-1]
                unescaped = unquoted.replace('\\"', '"').replace('\\\\', '\\')
                return json.loads(unescaped)
            except json.JSONDecodeError:
                pass
        
        return None
    
    # Try initial parse
    result = _parse_once(json_str)
    if result is None:
        preview = original_str[:200] if len(original_str) > 200 else original_str
        error_msg = (
            f"Failed to parse JSON after all normalization attempts. "
            f"Original length: {len(original_str)}. "
            f"Preview: {preview}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Recursively parse if result is still a string (handles double/triple encoding)
    depth = 0
    while isinstance(result, str) and depth < max_depth:
        try:
            parsed = json.loads(result)
            result = parsed
            depth += 1
            logger.debug(f"Recursively parsed JSON string (depth: {depth})")
        except (json.JSONDecodeError, TypeError):
            # Not a JSON string, return as-is
            break
    
    if isinstance(result, str) and depth >= max_depth:
        logger.warning(f"Reached max recursion depth ({max_depth}), returning string result")
    
    return result


def parse_df_json_safely(df_json: str) -> Dict[str, Any]:
    """
    Safely parse df_json parameter with comprehensive error handling.
    
    Wrapper around normalize_json_string with tool-specific error formatting.
    
    Args:
        df_json (str): JSON string representation of DataFrame.
        
    Returns:
        Dict[str, Any]: Parsed JSON object (dict).
        
    Raises:
        ValueError: If parsing fails, with structured error information.
    """
    try:
        result = normalize_json_string(df_json)
        if not isinstance(result, dict):
            raise ValueError(f"Expected dict after parsing, got {type(result)}")
        return result
    except ValueError as e:
        preview = df_json[:200] if len(df_json) > 200 else df_json
        error_detail = {
            "error": f"Invalid JSON in df_json parameter: {str(e)}",
            "error_type": "json_parse_error",
            "df_json_preview": preview,
            "df_json_length": len(df_json),
            "suggestion": "Ensure df_json is passed as a valid JSON string without extra escaping"
        }
        logger.error(f"Failed to parse df_json: {error_detail}")
        raise ValueError(json.dumps(error_detail)) from e

