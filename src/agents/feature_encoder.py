"""Feature encoding agent that analyzes feature columns and determines the best encoding strategy (numeric, ordinal, onehot, proximity, or label)."""

import json
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

from src.agents.tools import (
    get_unique_value_counts,
    detect_ordinal_patterns,
    get_column_statistics,
)
from src.utils.prompt_loader import load_prompt
from src.utils.logger import get_logger
from src.utils.observability import (
    log_llm_tool_call,
    log_tool_result,
    log_llm_follow_up,
    log_agent_interaction,
)

logger = get_logger(__name__)


def get_feature_encoder_tools() -> List["BaseTool"]:
    """Return tools available to the feature encoder agent. Returns: List[BaseTool]: List of tool functions."""
    return [
        get_unique_value_counts,
        detect_ordinal_patterns,
        get_column_statistics,
    ]


def create_feature_encoder_agent(llm: BaseChatModel) -> Any:
    """Create a feature encoder agent with tool-calling capabilities. Args: llm (BaseChatModel): LangChain chat model. Returns: Any: Runnable with bound tools (exact type depends on LLM implementation)."""
    tools = get_feature_encoder_tools()
    return llm.bind_tools(tools)


def build_encoding_prompt(df_json: str, features: List[str], dtypes: Dict[str, str]) -> str:
    """Build the user prompt for feature encoding analysis. Args: df_json (str): JSON DataFrame sample. features (List[str]): Feature column names. dtypes (Dict[str, str]): Column to dtype mapping. Returns: str: Formatted prompt."""
    feature_info = "\n".join([f"- {col}: {dtypes.get(col, 'unknown')}" for col in features])
    
    return f"""Please analyze these feature columns and determine the best encoding strategy for each.

## Feature Columns to Encode
{feature_info}

## Data Sample (JSON format for tool use)
```json
{df_json}
```

For each feature column:
1. Use `get_column_statistics` to understand the data distribution
2. For string/categorical columns, use `detect_ordinal_patterns` to check for ordinal structure
3. Use `get_unique_value_counts` to see the cardinality and value distribution

Based on your analysis, recommend an encoding type for each feature:
- `numeric`: Already numeric, no encoding needed
- `ordinal`: Has natural order, provide mapping
- `onehot`: Nominal category with low cardinality
- `proximity`: Geographic/location data
- `label`: Moderate cardinality categorical

Provide your final recommendations as JSON with key "encodings" mapping column names to their encoding config."""


def parse_encoding_response(response_content: str) -> Dict[str, Any]:
    """Parse the agent's response to extract encoding recommendations. Args: response_content (str): Raw response text. Returns: Dict[str, Any]: Parsed encoding dictionary."""
    try:
        if "```json" in response_content:
            json_str = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            json_str = response_content.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_content.strip()
        
        result = json.loads(json_str)
        
        if "encodings" not in result:
            result = {"encodings": result, "summary": "Extracted from response"}
        if "summary" not in result:
            result["summary"] = "No summary provided"
            
        return result
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse encoding JSON: {e}")
        return {
            "encodings": {},
            "summary": f"Failed to parse response: {response_content[:200]}",
            "raw_response": response_content
        }


async def run_feature_encoder(
    llm: BaseChatModel,
    df_json: str,
    features: List[str],
    dtypes: Dict[str, str],
    max_iterations: int = 15
) -> Dict[str, Any]:
    """Runs the feature encoding agent. Args: llm (BaseChatModel): LangChain chat model with tool-calling support. df_json (str): JSON representation of DataFrame sample. features (List[str]): List of feature column names. dtypes (Dict[str, str]): Dict mapping column names to dtypes. max_iterations (int): Maximum tool-calling iterations. Returns: Dict[str, Any]: Encoding recommendations with encodings dict and summary."""
    if not features:
        return {
            "encodings": {},
            "summary": "No features to encode"
        }
    
    system_prompt = load_prompt("agents/feature_encoder_system")
    user_prompt = build_encoding_prompt(df_json, features, dtypes)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    agent = create_feature_encoder_agent(llm)
    tools = {tool.name: tool for tool in get_feature_encoder_tools()}
    
    for iteration in range(max_iterations):
        response = await agent.ainvoke(messages)
        messages.append(response)
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                logger.info(f"Feature encoder calling tool: {tool_name}")
                
                if tool_name in tools:
                    tool_result = tools[tool_name].invoke(tool_args)
                    messages.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    ))
                else:
                    logger.warning(f"Unknown tool requested: {tool_name}")
        else:
            return parse_encoding_response(response.content)
    
    logger.warning("Max iterations reached in feature encoder")
    return parse_encoding_response(messages[-1].content if messages else "")


def run_feature_encoder_sync(
    llm: BaseChatModel,
    df_json: str,
    features: List[str],
    dtypes: Dict[str, str],
    max_iterations: int = 15,
    preset: Optional[str] = None
) -> Dict[str, Any]:
    """Synchronous feature encoder. Args: llm (BaseChatModel): LangChain chat model. df_json (str): JSON DataFrame sample. features (List[str]): Feature column names. dtypes (Dict[str, str]): Column to dtype mapping. max_iterations (int): Max iterations. preset (Optional[str]): Optional preset prompt name. Returns: Dict[str, Any]: Encoding recommendations."""
    if not features:
        return {
            "encodings": {},
            "summary": "No features to encode"
        }
    
    system_prompt = load_prompt("agents/feature_encoder_system")
    
    if preset and preset.lower() != "none":
        try:
            preset_content = load_prompt(f"presets/{preset}")
            system_prompt += f"\n\n{preset_content}"
        except Exception as e:
            logger.warning(f"Failed to load preset '{preset}': {e}")
    
    user_prompt = build_encoding_prompt(df_json, features, dtypes)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    agent = create_feature_encoder_agent(llm)
    tools = {tool.name: tool for tool in get_feature_encoder_tools()}
    
    for iteration in range(max_iterations):
        response = agent.invoke(messages)
        messages.append(response)
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                logger.info(f"Feature encoder calling tool: {tool_name}")
                log_llm_tool_call("feature_encoder", tool_name, tool_args, iteration + 1)
                
                if tool_name in tools:
                    tool_result = tools[tool_name].invoke(tool_args)
                    log_tool_result("feature_encoder", tool_name, tool_result, iteration + 1)
                    messages.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    ))
                    log_llm_follow_up("feature_encoder", messages, iteration + 1)
                else:
                    logger.warning(f"Unknown tool requested: {tool_name}")
        else:
            result = parse_encoding_response(response.content)
            log_agent_interaction(
                "feature_encoder",
                system_prompt,
                user_prompt,
                response.content if response.content else ""
            )
            return result
    
    logger.warning("Max iterations reached in feature encoder")
    final_content = messages[-1].content if messages else ""
    result = parse_encoding_response(final_content)
    log_agent_interaction(
        "feature_encoder",
        system_prompt,
        user_prompt,
        final_content
    )
    return result

