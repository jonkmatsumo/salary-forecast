"""
Observability utilities for logging LLM interactions and workflow state.

This module provides structured logging functions for tracking:
- LLM tool calls and results
- Agent interactions and message history
- Workflow state transitions
"""

import json
from typing import Any, Dict, List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _truncate_sensitive_data(data: Any, max_length: int = 500) -> str:
    """Truncate sensitive data for logging. Args: data (Any): Data to truncate. max_length (int): Max length. Returns: str: Truncated string."""
    data_str = str(data)
    if len(data_str) <= max_length:
        return data_str
    return data_str[:max_length] + "... [truncated]"


def _sanitize_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize tool arguments for logging. Args: args (Dict[str, Any]): Tool arguments. Returns: Dict[str, Any]: Sanitized arguments."""
    sanitized = {}
    for key, value in args.items():
        if key == "df_json" and isinstance(value, str):
            sanitized[key] = f"[DataFrame JSON, length={len(value)}]"
        else:
            sanitized[key] = value
    return sanitized


def log_llm_tool_call(
    agent_name: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    iteration: int
) -> None:
    """Log when LLM requests a tool. Args: agent_name (str): Agent name. tool_name (str): Tool name. tool_args (Dict[str, Any]): Tool arguments. iteration (int): Iteration number. Returns: None."""
    sanitized_args = _sanitize_args(tool_args)
    logger.info(
        f"[OBSERVABILITY] agent={agent_name} iteration={iteration} "
        f"tool_call={tool_name} args={json.dumps(sanitized_args)}"
    )


def log_tool_result(
    agent_name: str,
    tool_name: str,
    result: Any,
    iteration: int
) -> None:
    """Log tool execution results. Args: agent_name (str): Agent name. tool_name (str): Tool name. result (Any): Tool result. iteration (int): Iteration number. Returns: None."""
    result_str = str(result)
    result_length = len(result_str)
    result_preview = _truncate_sensitive_data(result_str, max_length=200)
    
    logger.info(
        f"[OBSERVABILITY] agent={agent_name} iteration={iteration} "
        f"tool_result={tool_name} result_length={result_length} "
        f"result_preview={result_preview}"
    )


def log_llm_follow_up(
    agent_name: str,
    messages: List[Any],
    iteration: int
) -> None:
    """Log messages sent to LLM after tool execution. Args: agent_name (str): Agent name. messages (List[Any]): Message list. iteration (int): Iteration number. Returns: None."""
    message_count = len(messages)
    last_message_type = type(messages[-1]).__name__ if messages else "None"
    
    logger.info(
        f"[OBSERVABILITY] agent={agent_name} iteration={iteration} "
        f"follow_up message_count={message_count} last_message_type={last_message_type}"
    )


def log_agent_interaction(
    agent_name: str,
    system_prompt: str,
    user_prompt: str,
    final_response: str
) -> None:
    """Log complete agent interaction. Args: agent_name (str): Agent name. system_prompt (str): System prompt. user_prompt (str): User prompt. final_response (str): Final response. Returns: None."""
    system_preview = _truncate_sensitive_data(system_prompt, max_length=200)
    user_preview = _truncate_sensitive_data(user_prompt, max_length=200)
    response_preview = _truncate_sensitive_data(final_response, max_length=200)
    
    logger.info(
        f"[OBSERVABILITY] agent={agent_name} interaction_complete "
        f"system_prompt_length={len(system_prompt)} "
        f"user_prompt_length={len(user_prompt)} "
        f"response_length={len(final_response)} "
        f"system_preview={system_preview} "
        f"user_preview={user_preview} "
        f"response_preview={response_preview}"
    )


def log_workflow_state_transition(
    node_name: str,
    state_snapshot: Dict[str, Any]
) -> None:
    """Log workflow state changes. Args: node_name (str): Node name. state_snapshot (Dict[str, Any]): State snapshot. Returns: None."""
    phase = state_snapshot.get("current_phase", "unknown")
    has_error = state_snapshot.get("error") is not None
    state_keys = list(state_snapshot.keys())
    
    logger.info(
        f"[OBSERVABILITY] workflow node={node_name} "
        f"state_transition phase={phase} has_error={has_error} "
        f"state_keys={state_keys}"
    )

