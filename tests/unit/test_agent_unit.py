import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import os
from typing import Dict, Any
from pydantic import BaseModel

from openserv_sdk.agent import Agent
from openserv_sdk.capability import Capability
from openserv_sdk.types import (
    AgentOptions, ProcessParams
)
from openserv_sdk.exceptions import RuntimeError, ToolError

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
    def test_port(self):
        return self.config.port

    @property
    def test_openai_tools(self):
        return self.openai_tools
        
    def get_test_config(self) -> Dict[str, Any]:
        """Get test configuration."""
        return {
            "port": self.test_port,
            "server": self.server
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