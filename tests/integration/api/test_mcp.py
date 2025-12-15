"""Integration tests for MCP (Model Context Protocol) endpoints.

Tests cover:
- JSON-RPC 2.0 protocol compliance
- tools/list endpoint (listing all available tools)
- tools/call endpoint (invoking individual tools)
- All 11 MCP tools (list_models, get_model_details, predict_salary, etc.)
- Error handling (invalid JSON, missing params, unknown tools)
- Tool schema validation (required fields, proper structure)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app


@pytest.fixture
def client():
    """Create test client. Returns: TestClient: FastAPI test client."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create mock model for testing. Returns: MagicMock: Mock model."""
    model = MagicMock()
    model.targets = ["BaseSalary", "TotalComp"]
    model.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    model.feature_names = ["Level", "Location", "YearsOfExperience"]
    model.ranked_encoders = {"Level": MagicMock(mapping={"L3": 0, "L4": 1, "L5": 2})}
    model.proximity_encoders = {"Location": MagicMock()}
    model.predict.return_value = {"BaseSalary": {"p10": 150000.0, "p50": 180000.0, "p90": 220000.0}}
    return model


def test_mcp_tools_list(client):
    """Test tools/list endpoint returns available tools."""
    with patch("src.api.mcp.server.get_mcp_tools") as mock_get_tools:
        mock_tool = MagicMock()
        mock_tool.to_dict.return_value = {
            "name": "test_tool",
            "description": "Test tool",
            "inputSchema": {"type": "object"},
        }
        mock_get_tools.return_value = [mock_tool]

        response = client.post(
            "/mcp/rpc",
            json={
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 1,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert "result" in data
        assert "tools" in data["result"]
        assert len(data["result"]["tools"]) > 0


def test_mcp_tools_list_invalid_jsonrpc(client):
    """Test tools/list with invalid jsonrpc version."""
    response = client.post(
        "/mcp/rpc",
        json={
            "jsonrpc": "1.0",
            "method": "tools/list",
            "id": 1,
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert data["jsonrpc"] == "2.0"
    assert "error" in data
    assert data["error"]["code"] == -32600


def test_mcp_tools_list_missing_method(client):
    """Test RPC request without method."""
    response = client.post(
        "/mcp/rpc",
        json={
            "jsonrpc": "2.0",
            "id": 1,
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32600


def test_mcp_list_models_tool(client):
    """Test list_models tool call."""
    with patch("src.api.mcp.server.MCPToolHandler") as MockHandler:
        mock_handler = MagicMock()
        mock_handler.handle_tool_call = AsyncMock(
            return_value={
                "models": [
                    {
                        "run_id": "test123",
                        "start_time": "2024-01-01T00:00:00",
                        "model_type": "XGBoost",
                        "cv_mean_score": 0.85,
                        "dataset_name": "test_data",
                        "additional_tag": None,
                    }
                ],
                "total": 1,
                "limit": 10,
                "offset": 0,
            }
        )
        MockHandler.return_value = mock_handler

        response = client.post(
            "/mcp/rpc",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "list_models",
                    "arguments": {"limit": 10},
                },
                "id": 1,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert "result" in data
        assert "models" in data["result"]
        assert len(data["result"]["models"]) == 1


def test_mcp_get_model_details_tool(client, mock_model):
    """Test get_model_details tool call."""
    with patch("src.api.mcp.server.MCPToolHandler") as MockHandler:
        mock_handler = MagicMock()
        mock_handler.handle_tool_call = AsyncMock(
            return_value={
                "run_id": "test123",
                "metadata": {
                    "run_id": "test123",
                    "start_time": "2024-01-01T00:00:00",
                    "model_type": "XGBoost",
                    "cv_mean_score": 0.85,
                    "dataset_name": "test_data",
                },
                "model_schema": {
                    "ranked_features": [],
                    "proximity_features": [],
                    "numerical_features": [],
                },
                "feature_names": ["Level", "Location"],
                "targets": ["BaseSalary"],
                "quantiles": [0.1, 0.5, 0.9],
            }
        )
        MockHandler.return_value = mock_handler

        response = client.post(
            "/mcp/rpc",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "get_model_details",
                    "arguments": {"run_id": "test123"},
                },
                "id": 1,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert "result" in data
        assert data["result"]["run_id"] == "test123"
        assert "metadata" in data["result"]
        assert "model_schema" in data["result"]


def test_mcp_predict_salary_tool(client, mock_model):
    """Test predict_salary tool call."""
    with patch("src.api.mcp.server.MCPToolHandler") as MockHandler:
        mock_handler = MagicMock()
        mock_handler.handle_tool_call = AsyncMock(
            return_value={
                "status": "success",
                "data": {
                    "predictions": {
                        "BaseSalary": {"p10": 150000.0, "p50": 180000.0, "p90": 220000.0}
                    },
                    "metadata": {},
                },
            }
        )
        MockHandler.return_value = mock_handler

        response = client.post(
            "/mcp/rpc",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "predict_salary",
                    "arguments": {
                        "run_id": "test123",
                        "features": {"Level": "L5", "Location": "San Francisco"},
                    },
                },
                "id": 1,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert "result" in data
        assert "data" in data["result"]
        assert "predictions" in data["result"]["data"]


def test_mcp_predict_salary_tool_model_not_found(client):
    """Test predict_salary with non-existent model."""
    with patch("src.api.mcp.server.MCPToolHandler") as MockHandler:
        from src.api.exceptions import ModelNotFoundError as APIModelNotFoundError

        mock_handler = MagicMock()
        mock_handler.handle_tool_call = AsyncMock(side_effect=APIModelNotFoundError("nonexistent"))
        MockHandler.return_value = mock_handler

        response = client.post(
            "/mcp/rpc",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "predict_salary",
                    "arguments": {
                        "run_id": "nonexistent",
                        "features": {"Level": "L5"},
                    },
                },
                "id": 1,
            },
        )

        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32603


def test_mcp_unknown_tool(client):
    """Test calling unknown tool."""
    response = client.post(
        "/mcp/rpc",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "unknown_tool",
                "arguments": {},
            },
            "id": 1,
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32602


def test_mcp_tools_call_missing_name(client):
    """Test tools/call without tool name."""
    response = client.post(
        "/mcp/rpc",
        json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "arguments": {},
            },
            "id": 1,
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32602


def test_mcp_invalid_json(client):
    """Test MCP endpoint with invalid JSON."""
    response = client.post(
        "/mcp/rpc",
        data="invalid json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32700


def test_mcp_get_training_status_tool(client):
    """Test get_training_status tool call."""
    with patch("src.api.mcp.server.MCPToolHandler") as MockHandler:
        mock_handler = MagicMock()
        mock_handler.handle_tool_call = AsyncMock(
            return_value={
                "job_id": "job123",
                "status": "COMPLETED",
                "progress": 1.0,
                "logs": ["Training started", "Training completed"],
                "submitted_at": "2024-01-01T00:00:00",
                "completed_at": "2024-01-01T01:00:00",
                "result": {"run_id": "model123"},
                "error": None,
                "run_id": "model123",
            }
        )
        MockHandler.return_value = mock_handler

        response = client.post(
            "/mcp/rpc",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "get_training_status",
                    "arguments": {"job_id": "job123"},
                },
                "id": 1,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert "result" in data
        assert data["result"]["status"] == "COMPLETED"
        assert "logs" in data["result"]


def test_mcp_get_feature_importance_tool(client, mock_model):
    """Test get_feature_importance tool call."""
    with patch("src.api.mcp.server.MCPToolHandler") as MockHandler:
        mock_handler = MagicMock()
        mock_handler.handle_tool_call = AsyncMock(
            return_value={
                "features": [
                    {"name": "Level", "gain": 0.5},
                    {"name": "Location", "gain": 0.3},
                ]
            }
        )
        MockHandler.return_value = mock_handler

        response = client.post(
            "/mcp/rpc",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "get_feature_importance",
                    "arguments": {
                        "run_id": "test123",
                        "target": "BaseSalary",
                        "quantile": 0.5,
                    },
                },
                "id": 1,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert "result" in data
        assert "features" in data["result"]


def test_mcp_start_configuration_workflow_tool(client):
    """Test start_configuration_workflow tool call."""
    with patch("src.api.mcp.server.MCPToolHandler") as MockHandler:
        mock_handler = MagicMock()
        mock_handler.handle_tool_call = AsyncMock(
            return_value={
                "workflow_id": "workflow123",
                "phase": "classification",
                "state": {
                    "phase": "classification",
                    "status": "success",
                    "current_result": {
                        "targets": ["Salary"],
                        "features": ["Level"],
                        "ignore": [],
                        "reasoning": "Test reasoning",
                    },
                },
            }
        )
        MockHandler.return_value = mock_handler

        import pandas as pd

        test_df = pd.DataFrame({"Salary": [100000], "Level": ["L3"]})
        df_json = test_df.to_json(orient="records", date_format="iso")

        response = client.post(
            "/mcp/rpc",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "start_configuration_workflow",
                    "arguments": {
                        "data": df_json,
                        "columns": ["Salary", "Level"],
                        "dtypes": {"Salary": "int64", "Level": "object"},
                        "dataset_size": 1,
                        "provider": "openai",
                    },
                },
                "id": 1,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert "result" in data
        assert "workflow_id" in data["result"] or "phase" in data["result"]


def test_mcp_tools_list_returns_all_tools(client):
    """Test that tools/list returns all expected tools."""
    response = client.post(
        "/mcp/rpc",
        json={
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 1,
        },
    )

    assert response.status_code == 200
    data = response.json()
    tools = data["result"]["tools"]
    tool_names = [tool["name"] for tool in tools]

    expected_tools = [
        "list_models",
        "get_model_details",
        "get_model_schema",
        "predict_salary",
        "start_training",
        "get_training_status",
        "start_configuration_workflow",
        "confirm_classification",
        "confirm_encoding",
        "finalize_configuration",
        "get_feature_importance",
    ]

    for expected_tool in expected_tools:
        assert expected_tool in tool_names, f"Expected tool {expected_tool} not found"


def test_mcp_tool_has_required_fields(client):
    """Test that all tools have required fields (name, description, inputSchema)."""
    response = client.post(
        "/mcp/rpc",
        json={
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 1,
        },
    )

    assert response.status_code == 200
    data = response.json()
    tools = data["result"]["tools"]

    for tool in tools:
        assert "name" in tool, f"Tool missing 'name': {tool}"
        assert "description" in tool, f"Tool {tool.get('name')} missing 'description'"
        assert "inputSchema" in tool, f"Tool {tool.get('name')} missing 'inputSchema'"
        assert (
            tool["inputSchema"]["type"] == "object"
        ), f"Tool {tool.get('name')} inputSchema must be object type"
