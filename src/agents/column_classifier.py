"""
Column Classification Agent.

This agent analyzes a dataset and classifies each column as:
- Target: columns to be predicted
- Feature: columns to use as predictors
- Ignore: columns to exclude from modeling
"""

import json
from typing import Any, Dict, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from src.agents.tools import (
    compute_correlation_matrix,
    get_column_statistics,
    detect_column_dtype,
)
from src.utils.prompt_loader import load_prompt
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_column_classifier_tools():
    """Return tools available to the column classifier agent."""
    return [
        compute_correlation_matrix,
        get_column_statistics,
        detect_column_dtype,
    ]


def create_column_classifier_agent(llm: BaseChatModel):
    """
    Create a column classifier agent with tool-calling capabilities.
    
    Args:
        llm: A LangChain chat model that supports tool calling.
        
    Returns:
        An agent that can classify columns using tools.
    """
    tools = get_column_classifier_tools()
    return llm.bind_tools(tools)


def build_classification_prompt(df_json: str, columns: List[str], dtypes: Dict[str, str]) -> str:
    """
    Build the user prompt for column classification.
    
    Args:
        df_json: JSON representation of the DataFrame sample.
        columns: List of column names.
        dtypes: Dictionary mapping column names to their dtypes.
        
    Returns:
        Formatted user prompt string.
    """
    column_info = "\n".join([f"- {col}: {dtypes.get(col, 'unknown')}" for col in columns])
    
    return f"""Please analyze this dataset and classify each column.

## Columns and Data Types
{column_info}

## Data Sample (JSON format for tool use)
```json
{df_json}
```

Use the available tools to analyze the columns before making your classification. Focus on:
1. Use `detect_column_dtype` on ambiguous columns to understand their semantic type
2. Use `compute_correlation_matrix` to see relationships between numeric columns
3. Use `get_column_statistics` on potential target columns to verify they're suitable

After your analysis, provide your final classification as JSON with keys: targets, features, ignore, reasoning."""


def parse_classification_response(response_content: str) -> Dict[str, Any]:
    """
    Parse the agent's response to extract classification.
    
    Args:
        response_content: Raw response text from the agent.
        
    Returns:
        Parsed classification dictionary.
    """
    # Try to find JSON in the response
    try:
        # Look for JSON block
        if "```json" in response_content:
            json_str = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            json_str = response_content.split("```")[1].split("```")[0].strip()
        else:
            # Try to parse the whole response as JSON
            json_str = response_content.strip()
        
        result = json.loads(json_str)
        
        # Validate required keys
        if "targets" not in result:
            result["targets"] = []
        if "features" not in result:
            result["features"] = []
        if "ignore" not in result:
            result["ignore"] = []
        if "reasoning" not in result:
            result["reasoning"] = "No reasoning provided"
            
        return result
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse classification JSON: {e}")
        return {
            "targets": [],
            "features": [],
            "ignore": [],
            "reasoning": f"Failed to parse response: {response_content[:200]}",
            "raw_response": response_content
        }


async def run_column_classifier(
    llm: BaseChatModel,
    df_json: str,
    columns: List[str],
    dtypes: Dict[str, str],
    max_iterations: int = 10
) -> Dict[str, Any]:
    """
    Run the column classification agent.
    
    This function handles the tool-calling loop, allowing the agent to
    use tools iteratively before providing its final classification.
    
    Args:
        llm: LangChain chat model with tool-calling support.
        df_json: JSON representation of DataFrame sample.
        columns: List of column names.
        dtypes: Dict mapping column names to dtypes.
        max_iterations: Maximum tool-calling iterations.
        
    Returns:
        Classification result with targets, features, ignore, and reasoning.
    """
    system_prompt = load_prompt("agents/column_classifier_system")
    user_prompt = build_classification_prompt(df_json, columns, dtypes)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    agent = create_column_classifier_agent(llm)
    tools = {tool.name: tool for tool in get_column_classifier_tools()}
    
    for iteration in range(max_iterations):
        response = await agent.ainvoke(messages)
        messages.append(response)
        
        # Check for tool calls
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                logger.info(f"Column classifier calling tool: {tool_name}")
                
                if tool_name in tools:
                    # Execute tool
                    tool_result = tools[tool_name].invoke(tool_args)
                    
                    # Add tool result to messages
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    ))
                else:
                    logger.warning(f"Unknown tool requested: {tool_name}")
        else:
            # No more tool calls, parse the final response
            return parse_classification_response(response.content)
    
    logger.warning("Max iterations reached in column classifier")
    return parse_classification_response(messages[-1].content if messages else "")


def run_column_classifier_sync(
    llm: BaseChatModel,
    df_json: str,
    columns: List[str],
    dtypes: Dict[str, str],
    max_iterations: int = 10
) -> Dict[str, Any]:
    """
    Synchronous version of run_column_classifier.
    
    Args:
        llm: LangChain chat model with tool-calling support.
        df_json: JSON representation of DataFrame sample.
        columns: List of column names.
        dtypes: Dict mapping column names to dtypes.
        max_iterations: Maximum tool-calling iterations.
        
    Returns:
        Classification result with targets, features, ignore, and reasoning.
    """
    system_prompt = load_prompt("agents/column_classifier_system")
    user_prompt = build_classification_prompt(df_json, columns, dtypes)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    agent = create_column_classifier_agent(llm)
    tools = {tool.name: tool for tool in get_column_classifier_tools()}
    
    for iteration in range(max_iterations):
        response = agent.invoke(messages)
        messages.append(response)
        
        # Check for tool calls
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                logger.info(f"Column classifier calling tool: {tool_name}")
                
                if tool_name in tools:
                    tool_result = tools[tool_name].invoke(tool_args)
                    
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    ))
                else:
                    logger.warning(f"Unknown tool requested: {tool_name}")
        else:
            return parse_classification_response(response.content)
    
    logger.warning("Max iterations reached in column classifier")
    return parse_classification_response(messages[-1].content if messages else "")

