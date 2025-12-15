"""JSON parsing utilities for handling escaped and normalized JSON strings that may be double-encoded or contain escape sequences from LLM tool calls."""

import json
from typing import Any, Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


def normalize_json_string(json_str: str, max_depth: int = 5) -> Any:
    """Normalizes and parses a JSON string that may be escaped or double-encoded using multiple parsing strategies. Args: json_str (str): JSON string that may be escaped or encoded. max_depth (int): Maximum recursion depth to prevent infinite loops. Returns: Any: Parsed JSON object (dict, list, or primitive type). Raises: ValueError: If all parsing attempts fail."""
    if not json_str or not isinstance(json_str, str):
        raise ValueError(f"Invalid input: expected non-empty string, got {type(json_str)}")

    original_str = json_str

    def _parse_once(s: str) -> Optional[Any]:
        """Tries all parsing strategies once. Args: s (str): JSON string to parse. Returns: Optional[Any]: Parsed JSON object or None if all strategies fail."""
        # Strategy 1: Direct parse
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass

        stripped = s.strip()

        # Strategy 2: Remove outer quotes if present
        if (stripped.startswith('"') and stripped.endswith('"')) or (
            stripped.startswith("'") and stripped.endswith("'")
        ):
            try:
                unquoted = stripped[1:-1]
                return json.loads(unquoted)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Handle escaped quotes
        try:
            manual_unescaped = s.replace('\\"', '"').replace("\\\\", "\\")
            return json.loads(manual_unescaped)
        except json.JSONDecodeError:
            pass

        # Strategy 4: Remove quotes and unescape
        if stripped.startswith('"') and stripped.endswith('"'):
            try:
                unquoted = stripped[1:-1]
                unescaped = unquoted.replace('\\"', '"').replace("\\\\", "\\")
                return json.loads(unescaped)
            except json.JSONDecodeError:
                pass

        # Strategy 5: Try unicode escape decoding
        try:
            unescaped = s.encode().decode("unicode_escape")
            return json.loads(unescaped)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

        # Strategy 6: Try to find and extract JSON object if embedded in text
        if "{" in s and "}" in s:
            try:
                start_idx = s.find("{")
                end_idx = s.rfind("}") + 1
                if start_idx < end_idx:
                    json_candidate = s[start_idx:end_idx]
                    return json.loads(json_candidate)
            except json.JSONDecodeError:
                pass

        return None

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

    depth = 0
    while isinstance(result, str) and depth < max_depth:
        try:
            parsed = json.loads(result)
            result = parsed
            depth += 1
            logger.debug(f"Recursively parsed JSON string (depth: {depth})")
        except (json.JSONDecodeError, TypeError):
            break

    if isinstance(result, str) and depth >= max_depth:
        logger.warning(f"Reached max recursion depth ({max_depth}), returning string result")

    return result


def parse_df_json_safely(df_json: str) -> Dict[str, Any]:
    """Safely parses df_json parameter with comprehensive error handling. Args: df_json (str): JSON string representation of DataFrame. Returns: Dict[str, Any]: Parsed JSON object (dict). Raises: ValueError: If parsing fails, with structured error information."""
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
            "suggestion": "Ensure df_json is passed as a valid JSON string without extra escaping",
        }
        logger.error(f"Failed to parse df_json: {error_detail}")
        raise ValueError(json.dumps(error_detail)) from e
