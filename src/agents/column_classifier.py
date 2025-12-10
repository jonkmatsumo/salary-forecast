"""
Column Classification Agent.

This agent analyzes a dataset and classifies each column as:
- Target: columns to be predicted
- Feature: columns to use as predictors
- Ignore: columns to exclude from modeling
"""

import json
from typing import Any, Dict, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.language_models import BaseChatModel

from src.agents.tools import (
    compute_correlation_matrix,
    get_column_statistics,
    detect_column_dtype,
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


def get_column_classifier_tools() -> List[Any]:
    """Return tools available to the column classifier agent. Returns: List[Any]: List of tool functions."""
    return [
        compute_correlation_matrix,
        get_column_statistics,
        detect_column_dtype,
    ]


def create_column_classifier_agent(llm: BaseChatModel) -> Any:
    """Create a column classifier agent with tool-calling capabilities. Args: llm (BaseChatModel): LangChain chat model with tool calling. Returns: Any: Agent with bound tools."""
    tools = get_column_classifier_tools()
    return llm.bind_tools(tools)


def build_classification_prompt(df_json: str, columns: List[str], dtypes: Dict[str, str]) -> str:
    """Build the user prompt for column classification. Args: df_json (str): JSON representation of DataFrame sample. columns (List[str]): Column names. dtypes (Dict[str, str]): Column name to dtype mapping. Returns: str: Formatted prompt."""
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
    """Parse the agent's response to extract classification. Args: response_content (str): Raw response text. Returns: Dict[str, Any]: Parsed classification dictionary."""
    logger.debug(f"Parsing classification response (length: {len(response_content) if response_content else 0})")
    
    try:
        json_str = None
        
        if "```json" in response_content:
            logger.debug("Found ```json block")
            json_str = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            logger.debug("Found ``` block (non-json)")
            json_str = response_content.split("```")[1].split("```")[0].strip()
        else:
            # Try to parse the whole response as JSON
            logger.debug("No code blocks found, attempting to parse entire response as JSON")
            json_str = response_content.strip()
        
        logger.debug(f"Extracted JSON string (length: {len(json_str) if json_str else 0})")
        logger.debug(f"JSON string preview: {json_str[:200] if json_str else 'None'}")
        
        if not json_str:
            raise ValueError("Empty JSON string extracted from response")
        
        result = json.loads(json_str)
        logger.debug(f"Successfully parsed JSON, keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
        
        # Validate required keys
        if "targets" not in result:
            result["targets"] = []
        if "features" not in result:
            result["features"] = []
        if "ignore" not in result:
            result["ignore"] = []
        if "column_types" not in result:
            result["column_types"] = {}
        if "reasoning" not in result:
            result["reasoning"] = "No reasoning provided"
        
        # Backward compatibility: if "locations" exists, migrate to column_types
        if "locations" in result and result["locations"]:
            for loc_col in result["locations"]:
                if loc_col not in result["column_types"]:
                    result["column_types"][loc_col] = "location"
            # Remove locations key after migration
            if "locations" in result:
                del result["locations"]
            
        return result
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse classification JSON: {e}")
        return {
            "targets": [],
            "features": [],
            "ignore": [],
            "column_types": {},
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
    """Run the column classification agent with tool-calling loop. Args: llm (BaseChatModel): LangChain chat model. df_json (str): JSON DataFrame sample. columns (List[str]): Column names. dtypes (Dict[str, str]): Column to dtype mapping. max_iterations (int): Max iterations. Returns: Dict[str, Any]: Classification result."""
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
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                logger.info(f"Column classifier calling tool: {tool_name}")
                
                if tool_name in tools:
                    tool_result = tools[tool_name].invoke(tool_args)
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


def run_column_classifier_sync(
    llm: BaseChatModel,
    df_json: str,
    columns: List[str],
    dtypes: Dict[str, str],
    max_iterations: int = 10,
    preset: Optional[str] = None
) -> Dict[str, Any]:
    """Synchronous column classifier. Args: llm (BaseChatModel): LangChain chat model. df_json (str): JSON DataFrame sample. columns (List[str]): Column names. dtypes (Dict[str, str]): Column to dtype mapping. max_iterations (int): Max tool-calling iterations. preset (Optional[str]): Optional preset prompt name. Returns: Dict[str, Any]: Classification result."""
    system_prompt = load_prompt("agents/column_classifier_system")
    
    if preset and preset.lower() != "none":
        try:
            preset_content = load_prompt(f"presets/{preset}")
            system_prompt += f"\n\n{preset_content}"
        except Exception as e:
            logger.warning(f"Failed to load preset '{preset}': {e}")
    
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
                        logger.debug(f"Tool {tool_name} returned result (type: {type(tool_result)}, length: {len(str(tool_result)) if tool_result else 0})")
                        logger.debug(f"Tool result preview: {str(tool_result)[:200] if tool_result else 'None'}")
                        
                        log_tool_result("column_classifier", tool_name, tool_result, iteration + 1)
                        
                        if not isinstance(tool_result, str):
                            logger.warning(f"Tool {tool_name} returned non-string result: {type(tool_result)}, converting to string")
                            tool_result = str(tool_result)
                        
                        messages.append(ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call["id"]
                        ))
                        logger.debug(f"Added ToolMessage for {tool_name}")
                        log_llm_follow_up("column_classifier", messages, iteration + 1)
                    except Exception as tool_err:
                        logger.error(f"Error invoking tool {tool_name}: {tool_err}", exc_info=True)
                        logger.error(f"Tool args that caused error: {tool_args}")
                        messages.append(ToolMessage(
                            content=f"Error executing tool {tool_name}: {str(tool_err)}",
                            tool_call_id=tool_call["id"]
                        ))
                else:
                    logger.warning(f"Unknown tool requested: {tool_name}")
        else:
            logger.info("No more tool calls, parsing final classification response")
            logger.debug(f"Response content type: {type(response.content)}")
            logger.debug(f"Response content length: {len(response.content) if response.content else 0}")
            logger.debug(f"Response content preview: {response.content[:500] if response.content else 'None'}")
            
            try:
                result = parse_classification_response(response.content)
                logger.info("Successfully parsed classification response")
                logger.debug(f"Parsed result keys: {list(result.keys())}")
                log_agent_interaction(
                    "column_classifier",
                    system_prompt,
                    user_prompt,
                    response.content if response.content else ""
                )
                return result
            except Exception as parse_err:
                logger.error(f"Failed to parse classification response: {parse_err}", exc_info=True)
                logger.error(f"Response content that failed to parse: {response.content[:1000] if response.content else 'None'}")
                raise
    
    logger.warning("Max iterations reached in column classifier")
    final_content = messages[-1].content if messages else ""
    result = parse_classification_response(final_content)
    log_agent_interaction(
        "column_classifier",
        system_prompt,
        user_prompt,
        final_content
    )
    return result

