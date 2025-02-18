import pytest
from unittest.mock import AsyncMock
from pydantic import BaseModel

from openserv_sdk import Agent, AgentOptions, Capability

class TestInput(BaseModel):
    input: str

@pytest.mark.asyncio
async def test_execute_capability():
    """Test executing a capability function."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key",
        openai_api_key="test-openai-key"
    ))

    async def test_run(params: dict, messages):
        return params["args"]["input"]

    capability = Capability(
        name="test_capability",
        description="A test capability",
        schema=TestInput,
        run=test_run
    )

    agent.add_capability(capability)

    result = await agent.handle_tool_route(
        "test_capability",
        {"args": {"input": "test"}, "messages": [], "action": None}
    )

    assert result == "test"

def test_validate_capability_schema():
    """Test capability schema validation."""
    capability = Capability(
        name="testCapability",
        description="A test capability",
        schema=TestInput,
        run=lambda params, messages: str(params["args"]["input"])
    )

    with pytest.raises(ValueError):
        capability.schema.model_validate({"invalid": "value"})

@pytest.mark.asyncio
async def test_handle_multiple_capabilities():
    """Test handling multiple capabilities."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key",
        openai_api_key="test-openai-key"
    ))

    async def test_run(params: dict, messages):
        return params["args"]["input"]

    capabilities = [
        Capability(
            name="tool1",
            description="Tool 1",
            schema=TestInput,
            run=test_run
        ),
        Capability(
            name="tool2",
            description="Tool 2",
            schema=TestInput,
            run=test_run
        )
    ]

    agent.add_capabilities(capabilities)

    # Test both tools
    result1 = await agent.handle_tool_route(
        "tool1",
        {"args": {"input": "test1"}, "messages": [], "action": None}
    )
    assert result1 == "test1"

    result2 = await agent.handle_tool_route(
        "tool2",
        {"args": {"input": "test2"}, "messages": [], "action": None}
    )
    assert result2 == "test2"

def test_duplicate_capability():
    """Test adding duplicate capability."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key",
        openai_api_key="test-openai-key"
    ))

    async def test_run(params: dict, messages):
        return params["args"]["input"]

    capability = Capability(
        name="test_tool",
        description="Test tool",
        schema=TestInput,
        run=test_run
    )

    agent.add_capability(capability)
    with pytest.raises(ValueError, match='Capability with name "test_tool" already exists'):
        agent.add_capability(capability)

def test_duplicate_capabilities():
    """Test adding capabilities with duplicate names."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key",
        openai_api_key="test-openai-key"
    ))

    async def test_run(params: dict, messages):
        return params["args"]["input"]

    capability = Capability(
        name="test_tool",
        description="Test tool",
        schema=TestInput,
        run=test_run
    )

    capabilities = [capability, capability]
    with pytest.raises(ValueError, match="Duplicate capability names found"):
        agent.add_capabilities(capabilities) 
