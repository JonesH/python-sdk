import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import os
from typing import Dict, Any
import aiohttp

from openserv_sdk.agent import Agent
from openserv_sdk.capability import Capability
from openserv_sdk.types import (
    AgentOptions, ProcessParams, GetTasksParams,
    RequestHumanAssistanceParams, TaskStatus, UploadFileParams,
    UpdateTaskStatusParams, ListFilesParams
)
from openserv_sdk.exceptions import RuntimeError, ToolError
from pydantic import BaseModel

class TestParams(BaseModel):
    input: str

@pytest.fixture
def mock_openai():
    """Mock AsyncOpenAI client for testing."""
    with patch('openai.AsyncOpenAI') as mock:
        # Create mock response
        mock_response = MagicMock()
        mock_response.model_dump = MagicMock(return_value={
            "choices": [{
                "message": {
                    "content": "Test response",
                    "role": "assistant",
                    "tool_calls": None
                }
            }]
        })
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='Test response',
                    role='assistant',
                    tool_calls=None,
                    model_dump=MagicMock(return_value={
                        "content": "Test response",
                        "role": "assistant",
                        "tool_calls": None
                    })
                )
            )
        ]
        
        # Create mock client
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock.return_value = mock_client
        yield mock

class TestAgent(Agent):
    """Test class that exposes protected/private members for testing."""
    @property
    def test_server(self):
        return self.server
    
    @property
    def test_port(self):
        return self.config.port

    @property
    def test_openai_tools(self):
        return self.openai_tools
        
    def get_test_config(self) -> Dict[str, Any]:
        """Get test configuration."""
        return {
            "port": self.test_port,
            "server": self.test_server
        }

def test_agent_initialization():
    """Test agent initialization with options."""
    agent = Agent(AgentOptions(
        system_prompt="Test prompt",
        api_key=os.getenv('OPENSERV_API_KEY'),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        port=int(os.getenv('PORT', '7378'))
    ))
    
    assert agent.config.system_prompt == "Test prompt"
    assert agent.config.api.api_key == os.getenv('OPENSERV_API_KEY')
    assert agent.config.openai.api_key == os.getenv('OPENAI_API_KEY')
    assert agent.config.port == int(os.getenv('PORT', '7378'))

@pytest.mark.asyncio
async def test_handle_tool_route():
    """Test handling a tool route."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key",
        openai_api_key="test-openai-key"
    ))
    
    # Add a test tool
    async def test_run(params, messages):
        return "success"
    
    capability = Capability(
        name="test_tool",
        description="A test tool",
        schema=TestParams,
        run=test_run
    )
    
    agent.add_capability(capability)
    
    result = await agent.handle_tool_route(
        tool_name="test_tool",
        body={"args": {"input": "test"}, "messages": []}
    )
    
    assert result == "success"

@pytest.mark.asyncio
async def test_handle_missing_tool():
    """Test handling a missing tool."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key",
        openai_api_key="test-openai-key"
    ))
    
    with pytest.raises(ToolError) as exc_info:
        await agent.handle_tool_route(
            tool_name="nonexistentTool",
            body={"args": {}, "messages": []}
        )
    
    assert "Tool not found" in str(exc_info.value)

@pytest.mark.asyncio
async def test_process_request(mock_openai):
    """Test processing a conversation."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key",
        openai_api_key="test-openai-key"
    ))

    # Mock OpenAI response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock(
        content="Test response",
        tool_calls=None
    )
    mock_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_response)

    # Set the mocked client
    agent._openai_client = mock_openai.return_value

    result = await agent.process(ProcessParams(messages=[
        {"role": "user", "content": "Hello"}
    ]))

    assert result == {"result": "Test response"}

@pytest.mark.asyncio
async def test_empty_openai_response(mock_openai):
    """Test handling empty OpenAI response."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key",
        openai_api_key="test-openai-key"
    ))
    
    # Ensure OpenAI client is initialized
    agent.openai_client = mock_openai.return_value
    
    # Mock empty response
    mock_response = MagicMock()
    mock_response.choices = []
    mock_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
    
    with pytest.raises(RuntimeError, match="No response from OpenAI"):
        await agent.process(ProcessParams(messages=[
            {"role": "user", "content": "Hello"}
        ]))

@pytest.mark.asyncio
async def test_file_operations():
    """Test file operations in agent."""
    agent = Agent(
        AgentOptions(
            api_key="test-key",
            platform_url="http://test-platform",
            runtime_url="http://test-runtime",
            system_prompt="Test system prompt"
        )
    )

    # Mock API client responses
    agent._api_client.get = AsyncMock(return_value={"data": []})
    agent._api_client.post = AsyncMock(return_value={"data": {"id": "test-file-id"}})

    # Test file listing
    files = await agent.get_files(workspace_id=1)
    assert files == []

    # Test text file upload
    text_content = "Hello World"
    upload_response = await agent.upload_file(
        UploadFileParams(
            workspace_id=1,
            path="test.txt",
            file=text_content,
            task_ids=[1, 2],
            skip_summarizer=True
        )
    )
    assert upload_response == {"id": "test-file-id"}

    # Test binary file upload
    binary_content = b"Binary Content"
    upload_response = await agent.upload_file(
        UploadFileParams(
            workspace_id=1,
            path="test.bin",
            file=binary_content
        )
    )
    assert upload_response == {"id": "test-file-id"}

    # Verify API calls
    agent._api_client.get.assert_called_once_with("/workspaces/1/file")
    
    # Verify post calls for both uploads
    assert agent._api_client.post.call_count == 2
    
    # Get the FormData from the first call (text file)
    text_call_args = agent._api_client.post.call_args_list[0][1]
    assert isinstance(text_call_args['data'], aiohttp.FormData)
    
    # Get the FormData from the second call (binary file)
    binary_call_args = agent._api_client.post.call_args_list[1][1]
    assert isinstance(binary_call_args['data'], aiohttp.FormData)

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

    tasks = await agent.get_tasks(GetTasksParams(workspace_id=1))
    assert tasks == {"tasks": []}

@pytest.mark.asyncio
async def test_chat_operations(mock_openai):
    """Test chat operations."""
    agent = Agent(AgentOptions(
        system_prompt="Test Agent",
        api_key="test-key",
        openai_api_key="test-openai-key"
    ))

    # Mock OpenAI response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock(
        content="Test response",
        tool_calls=None
    )
    mock_openai.return_value.chat.completions.create = AsyncMock(return_value=mock_response)

    # Set the mocked client
    agent._openai_client = mock_openai.return_value

    response = await agent.process(ProcessParams(
        messages=[{"role": "user", "content": "Hello"}]
    ))

    assert response == {"result": "Test response"}

@pytest.mark.asyncio
async def test_human_assistance():
    """Test requesting human assistance."""
    agent = Agent(AgentOptions(
        system_prompt="Test Agent",
        api_key=os.getenv('OPENSERV_API_KEY'),
        openai_api_key=os.getenv('OPENAI_API_KEY')
    ))
    
    params = RequestHumanAssistanceParams(
        workspace_id=1,
        task_id=1,
        type="text",
        question="test question",
        agent_dump={"key": "value"}
    )
    
    # Mock the API client
    agent.api_client = AsyncMock()
    agent.api_client.post.return_value = {"data": {"status": "success"}}
    
    response = await agent.request_human_assistance(params)
    assert response["status"] == "success"

@pytest.mark.asyncio
async def test_server_lifecycle(mock_openai):
    """Test server lifecycle operations."""
    agent = Agent(AgentOptions(
        system_prompt="Test Agent",
        api_key="test-key",
        openai_api_key="test-openai-key"
    ))

    # Mock server start/stop methods
    mock_server = MagicMock()
    mock_server.start = AsyncMock()
    mock_server.stop = AsyncMock()
    agent.server = mock_server

    # Test server start
    await agent.start()
    mock_server.start.assert_called_once()

    # Test server stop
    await agent.stop()
    mock_server.stop.assert_called_once()

    # Verify server is cleaned up
    assert agent.server is None

@pytest.mark.asyncio
async def test_openai_tools_conversion(mock_openai):
    """Test converting tools to OpenAI format."""
    agent = Agent(AgentOptions(
        system_prompt="Test Agent",
        api_key="test-key",
        openai_api_key="test-openai-key"
    ))
    
    tools = [
        {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "param": {"type": "string"}
                }
            }
        }
    ]
    
    openai_tools = agent.convert_to_openai_tools(tools)
    assert len(openai_tools) == 1
    assert openai_tools[0]["type"] == "function"
    assert openai_tools[0]["function"]["name"] == "test_tool"

@pytest.mark.asyncio
async def test_update_task_status():
    """Test updating task status."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key"
    ))
    
    # Mock API client
    agent.api_client = AsyncMock()
    agent.api_client.post.return_value = {"data": {"success": True}}
    
    params = UpdateTaskStatusParams(
        workspace_id=1,
        task_id=1,
        status=TaskStatus.IN_PROGRESS
    )
    
    result = await agent.update_task_status(params)
    assert result == {"success": True}
