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
    mock.handle_root_route = AsyncMock()
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
async def test_handle_root_success(server, mock_agent):
    """Test successful root route handling."""
    test_body = {"type": "do-task", "data": "test"}
    
    # Test the route handler directly
    await server.app.dependency_overrides[server.handle_root]({
        "request": {"body": test_body}
    })
    
    mock_agent.handle_root_route.assert_called_once_with(test_body)

@pytest.mark.asyncio
async def test_handle_root_error(server, mock_agent):
    """Test error handling in root route."""
    mock_agent.handle_root_route.side_effect = Exception("Test error")
    
    with pytest.raises(Exception) as exc_info:
        await server.app.dependency_overrides[server.handle_root]({
            "request": {"body": {}}
        })
    
    assert str(exc_info.value) == "Test error"

@pytest.mark.asyncio
async def test_handle_tool_success(server, mock_agent):
    """Test successful tool route handling."""
    test_body = {"args": {"test": "value"}}
    
    result = await server.app.dependency_overrides[server.handle_tool]({
        "tool_name": "test_tool",
        "request": {"body": test_body}
    })
    
    assert result == {"result": "success"}
    mock_agent.handle_tool_route.assert_called_once_with(
        tool_name="test_tool",
        body=test_body
    )

@pytest.mark.asyncio
async def test_handle_tool_error(server, mock_agent):
    """Test error handling in tool route."""
    mock_agent.handle_tool_route.side_effect = ToolError(
        tool_name="test_tool",
        message="Test error"
    )
    
    with pytest.raises(Exception) as exc_info:
        await server.app.dependency_overrides[server.handle_tool]({
            "tool_name": "test_tool",
            "request": {"body": {}}
        })
    
    assert "Test error" in str(exc_info.value)

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
        # No assertions needed as stop is a no-op in our implementation

def test_agent_not_initialized(server):
    """Test behavior when agent is not initialized."""
    server.set_agent(None)
    
    with TestClient(server.app) as client:
        response = client.post("/", json={})
        assert response.status_code == 500
        assert "Agent not initialized" in response.json()["detail"]

def test_invalid_json(server):
    """Test handling of invalid JSON."""
    with TestClient(server.app) as client:
        response = client.post("/", data="invalid json")
        assert response.status_code == 422  # FastAPI's default validation error

def test_concurrent_requests(server, mock_agent):
    """Test handling of concurrent requests."""
    import asyncio
    
    async def make_request():
        with TestClient(server.app) as client:
            return client.post("/tools/test", json={"args": {}})
    
    # Run multiple requests concurrently
    responses = asyncio.gather(*[make_request() for _ in range(5)])
    
    # Verify all requests were successful
    for response in responses:
        assert response.status_code == 200 