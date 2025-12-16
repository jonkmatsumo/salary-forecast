"""Column classification agent that analyzes datasets and classifies columns as targets, features, or ignore."""

import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

import time

from src.agents.tools import compute_correlation_matrix, detect_column_dtype, get_column_statistics
from src.utils.logger import get_logger
from src.utils.observability import (
    log_agent_interaction,
    log_llm_follow_up,
    log_llm_tool_call,
    log_tool_result,
)
from src.utils.performance import LLMCallTracker, extract_tokens_from_langchain_response
from src.utils.prompt_loader import load_prompt

logger = get_logger(__name__)


def get_column_classifier_tools() -> List["BaseTool"]:
    """Return tools available to the column classifier agent.

    Returns:
        List[BaseTool]: List of tool functions.
    """
    return [
        compute_correlation_matrix,
        get_column_statistics,
        detect_column_dtype,
    ]


def create_column_classifier_agent(llm: BaseChatModel) -> Any:
    """Create a column classifier agent with tool-calling capabilities.

    Args:
        llm (BaseChatModel): LangChain chat model with tool calling.

    Returns:
        Any: Runnable with bound tools (exact type depends on LLM implementation).
    """
    tools = get_column_classifier_tools()
    return llm.bind_tools(tools)


def build_classification_prompt(df_json: str, columns: List[str], dtypes: Dict[str, str]) -> str:
    """Build the user prompt for column classification.

    Args:
        df_json (str): JSON representation of DataFrame sample.
        columns (List[str]): Column names.
        dtypes (Dict[str, str]): Column name to dtype mapping.

    Returns:
        str: Formatted prompt.
    """
    column_info = "\n".join([f"- {col}: {dtypes.get(col, 'unknown')}" for col in columns])

    return f"""Please analyze this dataset and classify each column.

## Columns and Data Types
{column_info}

## Data Sample (JSON format for tool use)
```json
{df_json}
```

**IMPORTANT**: When calling tools that require the `df_json` parameter, pass the JSON string exactly as shown above without any additional escaping or quoting. The `df_json` parameter should be a valid JSON string that can be parsed directly. Do not wrap it in extra quotes or escape the quotes within it.

Use the available tools to analyze the columns before making your classification. Focus on:
1. Use `detect_column_dtype` on ambiguous columns to understand their semantic type (especially for string columns that might be locations)
2. Use `compute_correlation_matrix` to see relationships between numeric columns
3. Use `get_column_statistics` on potential target columns to verify they're suitable

After your analysis, provide your final classification as JSON with keys: targets, features, ignore, column_types, reasoning. Note: location columns should be assigned to targets or features based on their role, and their type should be recorded in column_types."""


def parse_classification_response(response_content: str) -> Dict[str, Any]:
    """Parse the agent's response to extract classification.

    Args:
        response_content (str): Raw response text.

    Returns:
        Dict[str, Any]: Parsed classification dictionary.
    """
    logger.debug(
        f"Parsing classification response (length: {len(response_content) if response_content else 0})"
    )

    try:
        json_str = None

        if "```json" in response_content:
            logger.debug("Found ```json block")
            json_str = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            logger.debug("Found ``` block (non-json)")
            json_str = response_content.split("```")[1].split("```")[0].strip()
        else:
            logger.debug("No code blocks found, attempting to parse entire response as JSON")
            json_str = response_content.strip()

        logger.debug(f"Extracted JSON string (length: {len(json_str) if json_str else 0})")
        logger.debug(f"JSON string preview: {json_str[:200] if json_str else 'None'}")

        if not json_str:
            raise ValueError("Empty JSON string extracted from response")

        if json_str.startswith("{") and json_str.endswith("}"):
            pass
        elif "{" in json_str and "}" in json_str:
            start_idx = json_str.find("{")
            end_idx = json_str.rfind("}") + 1
            json_str = json_str[start_idx:end_idx]
            logger.debug("Extracted JSON object from text")

        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected dict, got {type(parsed)}")
        result: Dict[str, Any] = parsed
        logger.debug(
            f"Successfully parsed JSON, keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}"
        )

        if "targets" not in result:
            logger.warning(
                "'targets' key missing from classification response, defaulting to empty list"
            )
            result["targets"] = []
        if "features" not in result:
            logger.warning(
                "'features' key missing from classification response, defaulting to empty list"
            )
            result["features"] = []
        if "ignore" not in result:
            logger.warning(
                "'ignore' key missing from classification response, defaulting to empty list"
            )
            result["ignore"] = []
        if "column_types" not in result:
            result["column_types"] = {}
        if "reasoning" not in result:
            result["reasoning"] = "No reasoning provided"

        if result["targets"] and not all(isinstance(t, str) for t in result["targets"]):
            logger.warning(f"Targets contains non-string values: {result['targets']}")
            result["targets"] = [str(t).strip() for t in result["targets"]]
        else:
            result["targets"] = [str(t).strip() for t in result["targets"]]

        if result["features"] and not all(isinstance(f, str) for f in result["features"]):
            logger.warning(f"Features contains non-string values: {result['features']}")
            result["features"] = [str(f).strip() for f in result["features"]]
        else:
            result["features"] = [str(f).strip() for f in result["features"]]

        if result["ignore"] and not all(isinstance(i, str) for i in result["ignore"]):
            logger.warning(f"Ignore contains non-string values: {result['ignore']}")
            result["ignore"] = [str(i).strip() for i in result["ignore"]]
        else:
            result["ignore"] = [str(i).strip() for i in result["ignore"]]

        result["targets"] = [t for t in result["targets"] if t]
        result["features"] = [f for f in result["features"] if f]
        result["ignore"] = [i for i in result["ignore"] if i]

        logger.debug(
            f"Normalized classification - targets: {result['targets']}, features: {result['features']}, ignore: {result['ignore']}"
        )

        if "locations" in result and result["locations"]:
            for loc_col in result["locations"]:
                if loc_col not in result["column_types"]:
                    result["column_types"][loc_col] = "location"
            if "locations" in result:
                del result["locations"]

        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse classification JSON: {e}")
        logger.debug(f"JSON parse error details: {str(e)}")
        logger.debug(f"Attempted to parse: {json_str[:500] if json_str else 'None'}")

        extracted_targets = []
        extracted_features = []
        extracted_ignore = []
        targets_pattern = r"(?:targets?|target\s*:)\s*\[([^\]]+)\]|(?:targets?|target\s*:)\s*([A-Za-z0-9\s,/\-]+?)(?:\n|$)"
        features_pattern = r"(?:features?|feature\s*:)\s*\[([^\]]+)\]|(?:features?|feature\s*:)\s*([A-Za-z0-9\s,/\-]+?)(?:\n|$)"
        ignore_pattern = r"(?:ignore|ignored)\s*\[([^\]]+)\]|(?:ignore|ignored)\s*:\s*([A-Za-z0-9\s,/\-]+?)(?:\n|$)"

        targets_match = re.search(targets_pattern, response_content, re.IGNORECASE)
        if targets_match:
            targets_str = targets_match.group(1) or targets_match.group(2) or ""
            extracted_targets = [
                t.strip().strip("\"'") for t in re.split(r"[,;]", targets_str) if t.strip()
            ]

        features_match = re.search(features_pattern, response_content, re.IGNORECASE)
        if features_match:
            features_str = features_match.group(1) or features_match.group(2) or ""
            extracted_features = [
                f.strip().strip("\"'") for f in re.split(r"[,;]", features_str) if f.strip()
            ]

        ignore_match = re.search(ignore_pattern, response_content, re.IGNORECASE)
        if ignore_match:
            ignore_str = ignore_match.group(1) or ignore_match.group(2) or ""
            extracted_ignore = [
                i.strip().strip("\"'") for i in re.split(r"[,;]", ignore_str) if i.strip()
            ]

        if extracted_targets or extracted_features or extracted_ignore:
            logger.info(
                f"Extracted classifications from text: {len(extracted_targets)} targets, "
                f"{len(extracted_features)} features, {len(extracted_ignore)} ignore"
            )
            return {
                "targets": extracted_targets,
                "features": extracted_features,
                "ignore": extracted_ignore,
                "column_types": {},
                "reasoning": (
                    response_content[:1000] if len(response_content) > 1000 else response_content
                ),
                "raw_response": response_content,
            }

        # If we couldn't extract anything, check if this is an error message
        error_indicators = ["error", "failed", "issue", "problem", "unable", "cannot", "could not"]
        is_error_response = any(
            indicator in response_content.lower() for indicator in error_indicators
        )

        if is_error_response:
            logger.error(
                f"LLM returned an error message instead of classification. "
                f"Response preview: {response_content[:500]}"
            )
            return {
                "targets": [],
                "features": [],
                "ignore": [],
                "column_types": {},
                "reasoning": f"Error from LLM: {response_content[:500]}",
                "raw_response": response_content,
                "error": "LLM encountered an error during classification",
            }

        return {
            "targets": [],
            "features": [],
            "ignore": [],
            "column_types": {},
            "reasoning": f"Failed to parse response: {response_content[:200]}",
            "raw_response": response_content,
        }


async def run_column_classifier(
    llm: BaseChatModel,
    df_json: str,
    columns: List[str],
    dtypes: Dict[str, str],
    max_iterations: int = 10,
) -> Dict[str, Any]:
    """Run the column classification agent with tool-calling loop.

    Args:
        llm (BaseChatModel): LangChain chat model.
        df_json (str): JSON DataFrame sample.
        columns (List[str]): Column names.
        dtypes (Dict[str, str]): Column to dtype mapping.
        max_iterations (int): Max iterations.

    Returns:
        Dict[str, Any]: Classification result.
    """
    system_prompt = load_prompt("agents/column_classifier_system")
    user_prompt = build_classification_prompt(df_json, columns, dtypes)

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    agent = create_column_classifier_agent(llm)
    tools = {tool.name: tool for tool in get_column_classifier_tools()}

    for iteration in range(max_iterations):
        start_time = time.time()
        response = await agent.ainvoke(messages)
        latency = time.time() - start_time

        prompt_tokens, completion_tokens, total_tokens = extract_tokens_from_langchain_response(
            response
        )
        model_name = getattr(llm, "model_name", "unknown")
        provider = "openai" if "gpt" in model_name.lower() else "gemini"

        from src.utils.performance import get_global_llm_tracker

        tracker = LLMCallTracker(model=model_name, provider=provider, global_tracking=True)
        tracker.record(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency=latency,
        )
        global_tracker = get_global_llm_tracker()
        if global_tracker:
            with global_tracker.lock:
                global_tracker.calls.append(tracker.calls[-1] if tracker.calls else {})

        messages.append(response)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                logger.info(f"Column classifier calling tool: {tool_name}")

                if tool_name in tools:
                    tool_result = tools[tool_name].invoke(tool_args)
                    messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                    )
                else:
                    logger.warning(f"Unknown tool requested: {tool_name}")
        else:
            response_content_raw = response.content if response.content else ""
            if isinstance(response_content_raw, str):
                response_content = response_content_raw
            elif isinstance(response_content_raw, list):
                response_content = str(response_content_raw[0]) if response_content_raw else ""
            else:
                response_content = str(response_content_raw)
            return parse_classification_response(response_content)

    logger.warning("Max iterations reached in column classifier")
    final_content_raw = messages[-1].content if messages else ""
    if isinstance(final_content_raw, str):
        final_content = final_content_raw
    elif isinstance(final_content_raw, list):
        final_content = str(final_content_raw[0]) if final_content_raw else ""
    else:
        final_content = str(final_content_raw) if final_content_raw else ""
    return parse_classification_response(final_content)


def run_column_classifier_sync(
    llm: BaseChatModel,
    df_json: str,
    columns: List[str],
    dtypes: Dict[str, str],
    max_iterations: int = 10,
    preset: Optional[str] = None,
) -> Dict[str, Any]:
    """Synchronous column classifier.

    Args:
        llm (BaseChatModel): LangChain chat model.
        df_json (str): JSON DataFrame sample.
        columns (List[str]): Column names.
        dtypes (Dict[str, str]): Column to dtype mapping.
        max_iterations (int): Max tool-calling iterations.
        preset (Optional[str]): Optional preset prompt name.

    Returns:
        Dict[str, Any]: Classification result.
    """
    system_prompt = load_prompt("agents/column_classifier_system")

    if preset and preset.lower() != "none":
        try:
            preset_content = load_prompt(f"presets/{preset}")
            system_prompt += f"\n\n{preset_content}"
        except Exception as e:
            logger.warning(f"Failed to load preset '{preset}': {e}")

    user_prompt = build_classification_prompt(df_json, columns, dtypes)

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    agent = create_column_classifier_agent(llm)
    tools = {tool.name: tool for tool in get_column_classifier_tools()}

    for iteration in range(max_iterations):
        start_time = time.time()
        response = agent.invoke(messages)
        latency = time.time() - start_time

        prompt_tokens, completion_tokens, total_tokens = extract_tokens_from_langchain_response(
            response
        )
        model_name = getattr(llm, "model_name", "unknown")
        provider = "openai" if "gpt" in model_name.lower() else "gemini"

        from src.utils.performance import get_global_llm_tracker

        tracker = LLMCallTracker(model=model_name, provider=provider, global_tracking=True)
        tracker.record(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency=latency,
        )
        global_tracker = get_global_llm_tracker()
        if global_tracker:
            with global_tracker.lock:
                global_tracker.calls.append(tracker.calls[-1] if tracker.calls else {})

        messages.append(response)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                logger.info(f"Column classifier calling tool: {tool_name}")
                log_llm_tool_call("column_classifier", tool_name, tool_args, iteration + 1)

                if tool_name in tools:
                    try:
                        logger.debug(f"Invoking tool {tool_name} with args: {tool_args}")
                        tool_result = tools[tool_name].invoke(tool_args)
                        logger.debug(
                            f"Tool {tool_name} returned result (type: {type(tool_result)}, length: {len(str(tool_result)) if tool_result else 0})"
                        )
                        logger.debug(
                            f"Tool result preview: {str(tool_result)[:200] if tool_result else 'None'}"
                        )

                        log_tool_result("column_classifier", tool_name, tool_result, iteration + 1)

                        if not isinstance(tool_result, str):
                            logger.warning(
                                f"Tool {tool_name} returned non-string result: {type(tool_result)}, converting to string"
                            )
                            tool_result = str(tool_result)

                        messages.append(
                            ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
                        )
                        logger.debug(f"Added ToolMessage for {tool_name}")
                        log_llm_follow_up("column_classifier", messages, iteration + 1)
                    except Exception as tool_err:
                        logger.error(f"Error invoking tool {tool_name}: {tool_err}", exc_info=True)
                        logger.error(f"Tool args that caused error: {tool_args}")
                        messages.append(
                            ToolMessage(
                                content=f"Error executing tool {tool_name}: {str(tool_err)}",
                                tool_call_id=tool_call["id"],
                            )
                        )
                else:
                    logger.warning(f"Unknown tool requested: {tool_name}")
        else:
            logger.info("No more tool calls, parsing final classification response")
            logger.debug(f"Response content type: {type(response.content)}")
            logger.debug(
                f"Response content length: {len(response.content) if response.content else 0}"
            )
            logger.debug(
                f"Response content preview: {response.content[:500] if response.content else 'None'}"
            )

            try:
                result = parse_classification_response(response.content)
                logger.info("Successfully parsed classification response")
                logger.debug(f"Parsed result keys: {list(result.keys())}")
                logger.info(
                    f"Classification summary: {len(result.get('targets', []))} targets, "
                    f"{len(result.get('features', []))} features, "
                    f"{len(result.get('ignore', []))} ignore"
                )
                logger.debug(f"Targets: {result.get('targets', [])}")
                logger.debug(f"Features: {result.get('features', [])}")
                logger.debug(f"Ignore: {result.get('ignore', [])}")
                log_agent_interaction(
                    "column_classifier",
                    system_prompt,
                    user_prompt,
                    response.content if response.content else "",
                )
                return result
            except Exception as parse_err:
                logger.error(f"Failed to parse classification response: {parse_err}", exc_info=True)
                logger.error(
                    f"Response content that failed to parse: {response.content[:1000] if response.content else 'None'}"
                )
                raise

    logger.warning("Max iterations reached in column classifier")
    final_content_raw = messages[-1].content if messages else ""
    final_content = (
        final_content_raw
        if isinstance(final_content_raw, str)
        else str(final_content_raw) if final_content_raw else ""
    )
    result = parse_classification_response(final_content)
    log_agent_interaction("column_classifier", system_prompt, user_prompt, final_content)
    return result
