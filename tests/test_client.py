"""Tests for API clients."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import json
from datetime import datetime
import httpx

from openserv_sdk.client import OpenServClient, RuntimeClient, BaseClient
from openserv_sdk.config import APIConfig
from openserv_sdk.exceptions import APIError, AuthenticationError

@pytest.fixture
def api_config():
    """Create test API configuration."""
    return APIConfig(
        api_key="test-key",
        platform_url="https://api.test.com",
        runtime_url="https://runtime.test.com"
    )

@pytest.fixture
def mock_httpx():
    """Mock httpx client."""
    with patch('httpx.AsyncClient') as mock:
        mock_client = AsyncMock()
        mock_client.request = AsyncMock()
        mock.return_value = mock_client
        yield mock_client

@pytest.mark.asyncio
async def test_base_client_request_success(api_config, mock_httpx):
    """Test successful API request."""
    client = BaseClient(api_config)
    
    # Mock successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {'content-type': 'application/json'}
    mock_response.json.return_value = {'data': 'test'}
    mock_httpx.request.return_value = mock_response
    
    result = await client._request('GET', '/test')
    assert result == {'data': 'test'}
    mock_httpx.request.assert_called_once()

@pytest.mark.asyncio
async def test_base_client_auth_error(api_config, mock_httpx):
    """Test authentication error handling."""
    client = BaseClient(api_config)
    
    # Mock 401 response
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "401 Unauthorized",
        request=MagicMock(),
        response=mock_response
    )
    mock_httpx.request.return_value = mock_response
    
    with pytest.raises(AuthenticationError):
        await client._request('GET', '/test')

@pytest.mark.asyncio
async def test_base_client_api_error(api_config, mock_httpx):
    """Test API error handling."""
    client = BaseClient(api_config)
    
    # Mock error response
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500 Server Error",
        request=MagicMock(),
        response=mock_response
    )
    mock_httpx.request.return_value = mock_response
    
    with pytest.raises(APIError) as exc_info:
        await client._request('GET', '/test')
    assert exc_info.value.status_code == 500

@pytest.mark.asyncio
async def test_openserv_client_methods(api_config, mock_httpx):
    """Test OpenServ client methods."""
    client = OpenServClient(api_config)
    
    # Mock successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {'content-type': 'application/json'}
    mock_response.json.return_value = {'data': 'test'}
    mock_httpx.request.return_value = mock_response
    
    # Test GET
    result = await client.get('/test')
    assert result == {'data': 'test'}
    
    # Test POST
    result = await client.post('/test', {'key': 'value'})
    assert result == {'data': 'test'}
    
    # Test PUT
    result = await client.put('/test', {'key': 'value'})
    assert result == {'data': 'test'}
    
    # Test DELETE
    result = await client.delete('/test')
    assert result == {'data': 'test'}

@pytest.mark.asyncio
async def test_runtime_client_methods(api_config, mock_httpx):
    """Test Runtime client methods."""
    client = RuntimeClient(api_config)
    
    # Mock successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {'content-type': 'application/json'}
    mock_response.json.return_value = {'data': 'test'}
    mock_httpx.request.return_value = mock_response
    
    # Test execute_task
    result = await client.execute_task(
        workspace_id=1,
        task_id=2,
        tools=[],
        messages=[],
        action={}
    )
    assert result == {'data': 'test'}
    
    # Test handle_chat
    result = await client.handle_chat(
        tools=[],
        messages=[],
        action={}
    )
    assert result == {'data': 'test'}

@pytest.mark.asyncio
async def test_datetime_serialization(api_config, mock_httpx):
    """Test datetime serialization in requests."""
    client = BaseClient(api_config)
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": "test"}
    mock_httpx.request.return_value = mock_response
    
    test_date = datetime.now()
    await client._request('POST', '/test', json_data={"date": test_date})
    
    # Verify datetime was serialized to ISO format
    mock_httpx.request.assert_called_once()
    call_args = mock_httpx.request.call_args
    assert call_args is not None
    assert 'content' in call_args[1]
    content = json.loads(call_args[1]['content'])
    assert isinstance(content['date'], str)
    datetime.fromisoformat(content['date'])  # Should not raise error 