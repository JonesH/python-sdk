"""Tests for the agent server."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from openserv_sdk.server import AgentServer
from openserv_sdk.config import ServerConfig
from openserv_sdk.exceptions import ToolError

@pytest.fixture
def server_config():
    """Create test server configuration."""
    return ServerConfig(
        host="localhost",
        port=7378
    )

@pytest.fixture
def mock_agent():
    """Create mock agent."""
    mock = AsyncMock()
    mock.handle_tool_route = AsyncMock(return_value="success")
    return mock

@pytest.fixture
def server(server_config, mock_agent):
    """Create test server instance."""
    server = AgentServer(server_config)
    server.set_agent(mock_agent)
    return server

@pytest.fixture
def test_client(server):
    """Create test client."""
    return TestClient(server.app)

def test_health_check(test_client):
    """Test health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_handle_tool_success(test_client, mock_agent):
    """Test successful tool route handling."""
    test_body = {"args": {"test": "value"}, "messages": []}
    
    response = test_client.post(
        "/tools/test_tool",
        json=test_body
    )
    
    assert response.status_code == 200
    assert response.json() == {"result": "success"}
    mock_agent.handle_tool_route.assert_called_once_with("test_tool", test_body)

@pytest.mark.asyncio
async def test_handle_tool_error(test_client, mock_agent):
    """Test error handling in tool route."""
    mock_agent.handle_tool_route.side_effect = ToolError(
        tool_name="test_tool",
        message="Test error"
    )
    
    response = test_client.post(
        "/tools/test_tool",
        json={"args": {}, "messages": []}
    )
    
    assert response.status_code == 400
    assert "Test error" in response.json()["detail"]

@pytest.mark.asyncio
async def test_server_lifecycle(server):
    """Test server start and stop."""
    with patch('uvicorn.Server') as mock_server:
        mock_server_instance = AsyncMock()
        mock_server.return_value = mock_server_instance
        
        # Test start
        await server.start()
        mock_server_instance.serve.assert_called_once()
        
        # Test stop
        await server.stop()

def test_agent_not_initialized(test_client, server):
    """Test behavior when agent is not initialized."""
    server.set_agent(None)
    response = test_client.post("/tools/test", json={"args": {}, "messages": []})
    assert response.status_code == 500
    assert "Agent not initialized" in response.json()["detail"]

def test_invalid_json(test_client):
    """Test handling of invalid JSON."""
    response = test_client.post("/", data="invalid json")
    assert response.status_code == 422  # FastAPI's default validation error 