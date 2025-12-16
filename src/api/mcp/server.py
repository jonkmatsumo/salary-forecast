"""MCP server implementation using JSON-RPC 2.0."""

from typing import Any, Dict

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from src.api.dependencies import get_current_user
from src.api.mcp.handlers import MCPToolHandler
from src.api.mcp.tools import get_mcp_tools
from src.utils.logger import get_logger

logger = get_logger(__name__)

mcp_router = APIRouter(prefix="/mcp", tags=["mcp"])


def create_mcp_app() -> APIRouter:
    """Create MCP router.

    Returns:
        APIRouter: Configured MCP router.
    """
    return mcp_router


@mcp_router.post("/rpc")
async def mcp_rpc(request: Request, user: str = Depends(get_current_user)):
    """Handle JSON-RPC 2.0 requests.

    Args:
        request (Request): Request object.
        user (str): Current user.

    Returns:
        JSONResponse: JSON-RPC response.
    """
    try:
        body = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse JSON-RPC request: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32700,
                    "message": "Parse error",
                    "data": str(e),
                },
                "id": None,
            },
        )

    jsonrpc_version = body.get("jsonrpc")
    method = body.get("method")
    params = body.get("params", {})
    request_id = body.get("id")

    if jsonrpc_version != "2.0":
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32600,
                    "message": "Invalid Request",
                    "data": "jsonrpc must be '2.0'",
                },
                "id": request_id,
            },
        )

    if not method:
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32600,
                    "message": "Invalid Request",
                    "data": "method is required",
                },
                "id": request_id,
            },
        )

    handler = MCPToolHandler()

    try:
        if method == "tools/list":
            result = await handle_tools_list()
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            if not tool_name:
                raise ValueError("tool name is required")
            result = await handler.handle_tool_call(tool_name, tool_args)
        else:
            raise ValueError(f"Unknown method: {method}")

        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id,
            }
        )
    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32602,
                    "message": "Invalid params",
                    "data": str(e),
                },
                "id": request_id,
            },
        )
    except Exception as e:
        logger.error(f"Error handling RPC request: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": "Internal error",
                    "data": str(e),
                },
                "id": request_id,
            },
        )


async def handle_tools_list() -> Dict[str, Any]:
    """Handle tools/list request.

    Returns:
        Dict[str, Any]: List of available tools.
    """
    tools = get_mcp_tools()
    return {
        "tools": [tool.to_dict() for tool in tools],
    }


def register_mcp_tools(app: Any) -> None:
    """Register MCP router with FastAPI app.

    Args:
        app (Any): FastAPI application.
    """
    app.include_router(mcp_router)
