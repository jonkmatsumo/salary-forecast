"""
Feature Encoding Agent.

This agent analyzes feature columns and determines the best encoding strategy:
- numeric: No encoding needed
- ordinal: Ordered categorical encoding with mapping
- onehot: One-hot encoding for nominal categories
- proximity: Location-based encoding
- label: Simple label encoding for moderate cardinality
"""

import json
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel

from src.agents.tools import (
    get_unique_value_counts,
    detect_ordinal_patterns,
    get_column_statistics,
)
from src.utils.prompt_loader import load_prompt
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_feature_encoder_tools():
    """Return tools available to the feature encoder agent."""
    return [
        get_unique_value_counts,
        detect_ordinal_patterns,
        get_column_statistics,
    ]


def create_feature_encoder_agent(llm: BaseChatModel):
    """
    Create a feature encoder agent with tool-calling capabilities.
    
    Args:
        llm: A LangChain chat model that supports tool calling.
        
    Returns:
        An agent that can determine encoding strategies using tools.
    """
    tools = get_feature_encoder_tools()
    return llm.bind_tools(tools)


def build_encoding_prompt(df_json: str, features: List[str], dtypes: Dict[str, str]) -> str:
    """
    Build the user prompt for feature encoding analysis.
    
    Args:
        df_json: JSON representation of the DataFrame sample.
        features: List of feature column names to encode.
        dtypes: Dictionary mapping column names to their dtypes.
        
    Returns:
        Formatted user prompt string.
    """
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
    """
    Parse the agent's response to extract encoding recommendations.
    
    Args:
        response_content: Raw response text from the agent.
        
    Returns:
        Parsed encoding dictionary.
    """
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
    """
    Run the feature encoding agent.
    
    Args:
        llm: LangChain chat model with tool-calling support.
        df_json: JSON representation of DataFrame sample.
        features: List of feature column names.
        dtypes: Dict mapping column names to dtypes.
        max_iterations: Maximum tool-calling iterations.
        
    Returns:
        Encoding recommendations with encodings dict and summary.
    """
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
    max_iterations: int = 15
) -> Dict[str, Any]:
    """
    Synchronous version of run_feature_encoder.
    
    Args:
        llm: LangChain chat model with tool-calling support.
        df_json: JSON representation of DataFrame sample.
        features: List of feature column names.
        dtypes: Dict mapping column names to dtypes.
        max_iterations: Maximum tool-calling iterations.
        
    Returns:
        Encoding recommendations with encodings dict and summary.
    """
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
        response = agent.invoke(messages)
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

