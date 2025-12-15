"""Model configuration agent that proposes monotonic constraints, quantiles, and XGBoost hyperparameters."""

import json
import time
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.logger import get_logger
from src.utils.observability import log_agent_interaction
from src.utils.performance import LLMCallTracker, extract_tokens_from_langchain_response
from src.utils.prompt_loader import load_prompt

logger = get_logger(__name__)


def build_configuration_prompt(
    targets: List[str],
    encodings: Dict[str, Any],
    correlation_data: Optional[str] = None,
    column_stats: Optional[Dict[str, Any]] = None,
    dataset_size: int = 0,
) -> str:
    """Build the user prompt for model configuration. Args: targets (List[str]): Target column names. encodings (Dict[str, Any]): Feature encoding recommendations. correlation_data (Optional[str]): Correlation matrix JSON. column_stats (Optional[Dict[str, Any]]): Column statistics. dataset_size (int): Number of rows. Returns: str: Formatted prompt."""
    encoding_lines = []
    for col, config in encodings.get("encodings", {}).items():
        enc_type = config.get("type", "unknown")
        encoding_lines.append(f"- {col}: {enc_type}")
        if enc_type == "ordinal" and "mapping" in config:
            mapping_str = ", ".join([f"{k}={v}" for k, v in list(config["mapping"].items())[:5]])
            encoding_lines.append(f"  Mapping: {mapping_str}...")

    encodings_formatted = (
        "\n".join(encoding_lines) if encoding_lines else "No features to configure"
    )

    prompt = f"""Please configure the XGBoost model based on the following information.

## Target Columns
{', '.join(targets) if targets else 'None specified'}

## Feature Encodings
{encodings_formatted}

## Dataset Size
{dataset_size} rows
"""

    if correlation_data:
        prompt += f"""
## Correlation Data
```json
{correlation_data}
```
"""

    if column_stats:
        stats_formatted = json.dumps(column_stats, indent=2)
        prompt += f"""
## Column Statistics
```json
{stats_formatted}
```
"""

    prompt += """
Based on this information:
1. Determine monotonic constraints for each encoded feature
2. Suggest appropriate quantiles for the prediction task
3. Recommend hyperparameters suitable for the dataset size

Provide your configuration as JSON with keys: features, quantiles, hyperparameters, reasoning."""

    return prompt


def parse_configuration_response(response_content: str) -> Dict[str, Any]:
    """Parses the agent's response to extract model configuration. Args: response_content (str): Raw response text from the agent. Returns: Dict[str, Any]: Parsed configuration dictionary."""
    try:
        if "```json" in response_content:
            json_str = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            json_str = response_content.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_content.strip()

        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected dict, got {type(parsed)}")
        result: Dict[str, Any] = parsed

        # Ensure required keys with defaults
        if "features" not in result:
            result["features"] = []
        if "quantiles" not in result:
            result["quantiles"] = [0.1, 0.25, 0.5, 0.75, 0.9]
        if "hyperparameters" not in result:
            result["hyperparameters"] = get_default_hyperparameters()
        if "reasoning" not in result:
            result["reasoning"] = "No reasoning provided"

        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse configuration JSON: {e}")
        return {
            "features": [],
            "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
            "hyperparameters": get_default_hyperparameters(),
            "reasoning": f"Failed to parse response: {response_content[:200]}",
            "raw_response": response_content,
        }


def get_default_hyperparameters() -> Dict[str, Any]:
    """Return default hyperparameters for XGBoost. Returns: Dict[str, Any]: Default hyperparameters."""
    return {
        "training": {
            "objective": "reg:quantileerror",
            "tree_method": "hist",
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": 0,
        },
        "cv": {
            "num_boost_round": 200,
            "nfold": 5,
            "early_stopping_rounds": 20,
            "verbose_eval": False,
        },
    }


async def run_model_configurator(
    llm: BaseChatModel,
    targets: List[str],
    encodings: Dict[str, Any],
    correlation_data: Optional[str] = None,
    column_stats: Optional[Dict[str, Any]] = None,
    dataset_size: int = 0,
) -> Dict[str, Any]:
    """Runs the model configuration agent which synthesizes information from previous agents to make configuration recommendations. Args: llm (BaseChatModel): LangChain chat model. targets (List[str]): List of target column names. encodings (Dict[str, Any]): Feature encoding recommendations. correlation_data (Optional[str]): Optional correlation JSON string. column_stats (Optional[Dict[str, Any]]): Optional column statistics. dataset_size (int): Number of rows in dataset. Returns: Dict[str, Any]: Model configuration with features, quantiles, hyperparameters."""
    system_prompt = load_prompt("agents/model_configurator_system")
    user_prompt = build_configuration_prompt(
        targets, encodings, correlation_data, column_stats, dataset_size
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    response = await llm.ainvoke(messages)
    content = (
        response.content
        if isinstance(response.content, str)
        else str(response.content) if response.content else ""
    )
    return parse_configuration_response(content)


def run_model_configurator_sync(
    llm: BaseChatModel,
    targets: List[str],
    encodings: Dict[str, Any],
    correlation_data: Optional[str] = None,
    column_stats: Optional[Dict[str, Any]] = None,
    dataset_size: int = 0,
    preset: Optional[str] = None,
) -> Dict[str, Any]:
    """Synchronous model configurator. Args: llm (BaseChatModel): LangChain chat model. targets (List[str]): Target column names. encodings (Dict[str, Any]): Feature encoding recommendations. correlation_data (Optional[str]): Correlation JSON. column_stats (Optional[Dict[str, Any]]): Column statistics. dataset_size (int): Number of rows. preset (Optional[str]): Optional preset prompt name. Returns: Dict[str, Any]: Model configuration with features, quantiles, hyperparameters."""
    system_prompt = load_prompt("agents/model_configurator_system")

    if preset and preset.lower() != "none":
        try:
            preset_content = load_prompt(f"presets/{preset}")
            system_prompt += f"\n\n{preset_content}"
        except Exception as e:
            logger.warning(f"Failed to load preset '{preset}': {e}")

    user_prompt = build_configuration_prompt(
        targets, encodings, correlation_data, column_stats, dataset_size
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    start_time = time.time()
    response = llm.invoke(messages)
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

    response_content_raw = response.content if response.content else ""
    if isinstance(response_content_raw, str):
        response_content = response_content_raw
    elif isinstance(response_content_raw, list):
        response_content = str(response_content_raw[0]) if response_content_raw else ""
    else:
        response_content = str(response_content_raw)
    result: Dict[str, Any] = parse_configuration_response(response_content)
    log_agent_interaction(
        "model_configurator",
        system_prompt,
        user_prompt,
        response_content,
    )
    return result
