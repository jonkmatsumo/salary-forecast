"""Unit tests for MCP server."""

from fastapi import APIRouter
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.mcp.server import create_mcp_app, handle_tools_list


def test_create_mcp_app():
    """Test create_mcp_app returns router."""
    router = create_mcp_app()
    assert isinstance(router, APIRouter)
    assert router.prefix == "/mcp"


def test_handle_tools_list():
    """Test handle_tools_list returns tools."""
    import asyncio

    result = asyncio.run(handle_tools_list())

    assert "tools" in result
    assert isinstance(result["tools"], list)
    assert len(result["tools"]) > 0

    for tool in result["tools"]:
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool


def test_mcp_rpc_unknown_method():
    """Test mcp_rpc with unknown method to cover line 89."""
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/mcp/rpc",
        json={
            "jsonrpc": "2.0",
            "method": "unknown_method",
            "id": 1,
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32602
    assert "Unknown method" in data["error"]["data"]
