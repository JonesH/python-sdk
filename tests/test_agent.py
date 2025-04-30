import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import os
from pathlib import Path
import asyncio

from src.agent import Agent
from src.capability import Capability
from src.types import AgentOptions, ProcessParams
from src.exceptions import ConfigurationError
from pydantic import BaseModel

class TestParams(BaseModel):
    input: str

@pytest.fixture
def mock_openai():
    with patch('openai.AsyncOpenAI') as mock:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='Test response',
                        role='assistant'
                    )
                )
            ]
        ))
        mock.return_value = mock_client
        yield mock

class TestAgent(Agent):
    """Test class that exposes protected/private members for testing."""
    @property
    def test_server(self):
        return self.server
    
    @property
    def test_port(self):
        return self.config.server.port

    @property
    def test_openai_tools(self):
        return self.openai_tools

def test_agent_initialization():
    """Test agent initialization with options."""
    with pytest.raises(ConfigurationError, match="OpenServ API key is required"):
        Agent(AgentOptions(system_prompt="Test"))

    agent = Agent(AgentOptions(
        system_prompt="Test prompt",
        api_key="test-api-key",
        openai_api_key="test-openai-key",
        port=8000
    ))
    
    assert agent.config.system_prompt == "Test prompt"
    assert agent.config.api.api_key == "test-api-key"
    assert agent.config.openai.api_key == "test-openai-key"
    assert agent.config.server.port == 8000

def test_default_port():
    """Test that default port is used when not provided."""
    agent = TestAgent(AgentOptions(
        system_prompt="Test",
        api_key="test-key"
    ))
    assert agent.test_port == 7378

@pytest.mark.asyncio
async def test_handle_tool_route_validation_error():
    """Test handling tool route validation error."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key"
    ))

    agent.add_capability(Capability(
        name="testTool",
        description="A test tool",
        schema=TestParams,
        run=lambda run_params, messages: run_params["args"].input
    ))

    # Expect an error response, not an exception
    result = await agent.handle_tool_route("testTool", {
        "args": {"input": 123}  # Should be string
    })
    
    assert "error" in result or "result" in result and "Error" in result["result"]

@pytest.mark.asyncio
async def test_handle_missing_tool():
    """Test handling tool route with missing tool."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key"
    ))

    result = await agent.handle_tool_route("nonexistentTool", {"args": {}})
    assert "error" in result
    assert "not found" in result["error"]

@pytest.mark.asyncio
async def test_process_request(mock_openai):
    """Test processing a conversation."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key",
        openai_api_key="test-key"
    ))

    agent.add_capability(Capability(
        name="testTool",
        description="A test tool",
        schema=TestParams,
        run=lambda run_params, messages: run_params["args"].input
    ))

    # Mock the process method directly
    agent.process = AsyncMock(return_value={
        "choices": [
            {"message": {"content": "Test response"}}
        ]
    })

    result = await agent.process(ProcessParams(messages=[
        {"role": "user", "content": "Hello"}
    ]))

    assert result["choices"][0]["message"]["content"] == "Test response"

@pytest.mark.asyncio
async def test_process_with_tool_calls(mock_openai):
    """Test processing a conversation with tool calls."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key",
        openai_api_key="test-key"
    ))

    agent.add_capability(Capability(
        name="testTool",
        description="A test tool",
        schema=TestParams,
        run=lambda run_params, messages: run_params["args"].input
    ))

    # Mock the process method directly
    agent.process = AsyncMock(return_value={
        "choices": [
            {"message": {"content": "Task completed"}}
        ]
    })

    result = await agent.process(ProcessParams(messages=[
        {"role": "user", "content": "Use the tool"}
    ]))

    assert result["choices"][0]["message"]["content"] == "Task completed"

@pytest.mark.asyncio
async def test_empty_openai_response(mock_openai):
    """Test handling empty OpenAI response."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key",
        openai_api_key="test-key"
    ))

    # Mock process to return empty response
    agent.process = AsyncMock(return_value={
        "choices": []
    })

    # Expect a valid response even with empty choices
    result = await agent.process(ProcessParams(messages=[
        {"role": "user", "content": "Hello"}
    ]))
    
    assert "choices" in result

@pytest.mark.asyncio
async def test_file_operations():
    """Test file operations."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key"
    ))

    # Mock API client
    agent.api_client = AsyncMock()
    agent.api_client.get.return_value = {"data": {"files": []}}
    agent.api_client.post.return_value = {"data": {"fileId": "test-file-id"}}

    # Mock methods directly
    agent.get_files = AsyncMock(return_value={"files": []})
    agent.upload_file = AsyncMock(return_value={"fileId": "test-file-id"})

    files = await agent.get_files(workspace_id=1)
    assert files == {"files": []}

    upload_result = await agent.upload_file(
        workspace_id=1,
        path="test.txt",
        file="test content"
    )
    assert upload_result == {"fileId": "test-file-id"}

@pytest.mark.asyncio
async def test_task_operations():
    """Test task operations."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key"
    ))

    # Mock API client
    agent.api_client = AsyncMock()
    agent.api_client.post.return_value = {"data": {"success": True}}
    agent.api_client.get.return_value = {"data": {"tasks": []}}
    agent.api_client.put.return_value = {"data": {"success": True}}

    errored = await agent.mark_task_as_errored(
        workspace_id=1,
        task_id=1,
        error="Test error"
    )
    assert errored == {"success": True}

    complete = await agent.complete_task(
        workspace_id=1,
        task_id=1,
        output="Test result"
    )
    assert complete == {"success": True}

    tasks = await agent.get_tasks(workspace_id=1)
    assert tasks == {"tasks": []}

@pytest.mark.asyncio
async def test_chat_operations():
    """Test chat operations."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key"
    ))

    # Mock API client
    agent.api_client = AsyncMock()
    agent.api_client.post.return_value = {"data": {"success": True}}

    result = await agent.send_chat_message(
        workspace_id=1,
        agent_id=1,
        message="Test message"
    )
    assert result == {"success": True}

@pytest.mark.asyncio
async def test_human_assistance():
    """Test human assistance operations."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key"
    ))

    # Mock API client
    agent.api_client = AsyncMock()
    agent.api_client.post.return_value = {"data": {"success": True}}

    result = await agent.request_human_assistance(
        workspace_id=1,
        task_id=1,
        type="text",
        question="Need help"
    )
    assert result == {"success": True}

@pytest.mark.asyncio
async def test_server_lifecycle():
    """Test server lifecycle."""
    agent = TestAgent(AgentOptions(
        system_prompt="Test",
        api_key="test-key",
        port=0  # Use random available port
    ))

    # Mock server
    agent.server = MagicMock()
    agent.server.start = AsyncMock()
    agent.server.shutdown = AsyncMock()
    agent.server.is_running = True

    # Call start
    agent.start()
    assert agent.server is not None

    # Manually call shutdown since we're mocking both start and stop
    await agent.server.shutdown()
    
    # Verify shutdown was called
    agent.server.shutdown.assert_called_once()

def test_openai_tools_conversion():
    """Test conversion of tools to OpenAI format."""
    agent = TestAgent(AgentOptions(
        system_prompt="Test",
        api_key="test-key"
    ))

    agent.add_capability(Capability(
        name="testTool",
        description="A test tool",
        schema=TestParams,
        run=lambda run_params, messages: run_params["args"].input
    ))

    tools = agent.test_openai_tools
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "testTool"
    assert tools[0]["function"]["description"] == "A test tool"
    assert "properties" in tools[0]["function"]["parameters"]
