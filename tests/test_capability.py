import pytest
from pydantic import BaseModel
from src.capability import Capability
from src.agent import Agent
from src.types import AgentOptions

class TestParams(BaseModel):
    input: str

@pytest.mark.asyncio
async def test_execute_capability():
    """Test executing a capability function."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key",
        openai_api_key="test-key"
    ))

    agent.add_capability(Capability(
        name="testCapability",
        description="A test capability",
        schema=TestParams,
        run=lambda run_params, messages: run_params["args"].input
    ))

    result = await agent.handle_tool_route("testCapability", {
        "args": {"input": "test"}
    })
    assert result == {"result": "test"}

def test_validate_capability_schema():
    """Test capability schema validation."""
    capability = Capability(
        name="testCapability",
        description="A test capability",
        schema=TestParams,
        run=lambda run_params, messages: run_params["args"].input
    )

    with pytest.raises(ValueError):
        capability.schema.model_validate({"input": 123})  # Should be string

@pytest.mark.asyncio
async def test_handle_multiple_capabilities():
    """Test handling multiple capabilities."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key",
        openai_api_key="test-key"
    ))

    capabilities = [
        Capability(
            name="tool1",
            description="Tool 1",
            schema=TestParams,
            run=lambda run_params, messages: run_params["args"].input
        ),
        Capability(
            name="tool2",
            description="Tool 2",
            schema=TestParams,
            run=lambda run_params, messages: run_params["args"].input
        )
    ]

    agent.add_capabilities(capabilities)

    # Test both tools
    result1 = await agent.handle_tool_route("tool1", {
        "args": {"input": "test1"}
    })
    assert result1 == {"result": "test1"}

    result2 = await agent.handle_tool_route("tool2", {
        "args": {"input": "test2"}
    })
    assert result2 == {"result": "test2"}

def test_duplicate_capability():
    """Test adding duplicate capability."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key",
        openai_api_key="test-key"
    ))

    agent.add_capability(Capability(
        name="test",
        description="Tool 1",
        schema=TestParams,
        run=lambda run_params, messages: run_params["args"].input
    ))

    with pytest.raises(ValueError, match='Tool with name "test" already exists'):
        agent.add_capability(Capability(
            name="test",
            description="Tool 1 duplicate",
            schema=TestParams,
            run=lambda run_params, messages: run_params["args"].input
        ))

def test_duplicate_capabilities():
    """Test adding capabilities with duplicate names."""
    agent = Agent(AgentOptions(
        system_prompt="Test",
        api_key="test-key",
        openai_api_key="test-key"
    ))

    capabilities = [
        Capability(
            name="tool1",
            description="Tool 1",
            schema=TestParams,
            run=lambda run_params, messages: run_params["args"].input
        ),
        Capability(
            name="tool1",  # Duplicate name
            description="Tool 1 duplicate",
            schema=TestParams,
            run=lambda run_params, messages: run_params["args"].input
        )
    ]

    with pytest.raises(ValueError, match='Tool with name "tool1" already exists'):
        agent.add_capabilities(capabilities) 
