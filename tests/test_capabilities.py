import pytest
from pydantic import BaseModel
from src.agent import Agent
from src.capability import Capability
from src.types import AgentOptions

@pytest.fixture
def mock_api_key():
    return "test-openserv-key"

class TestInput(BaseModel):
    input: str

@pytest.mark.asyncio
async def test_execute_capability(mock_api_key):
    agent = Agent(AgentOptions(
        api_key=mock_api_key,
        system_prompt="You are a test agent",
        openai_api_key="test-openai-key"
    ))

    async def test_run(run_params, messages):
        return run_params["args"].input

    agent.add_capability(Capability(
        name="testCapability",
        description="A test capability",
        schema=TestInput,
        run=test_run
    ))

    result = await agent.handle_tool_route("testCapability", {
        "args": {"input": "test"}
    })

    assert result == {"result": "test"}

def test_validate_capability_schema(mock_api_key):
    class TestNumberInput(BaseModel):
        input: int

    async def test_run(run_params, messages):
        return str(run_params["args"].input)

    capability = Capability(
        name="testCapability",
        description="A test capability",
        schema=TestNumberInput,
        run=test_run
    )

    with pytest.raises(ValueError):
        capability.schema(input="not a number")

@pytest.mark.asyncio
async def test_handle_multiple_capabilities(mock_api_key):
    agent = Agent(AgentOptions(
        api_key=mock_api_key,
        system_prompt="You are a test agent",
        openai_api_key="test-openai-key"
    ))

    async def test_run(run_params, messages):
        return run_params["args"].input

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

    for capability in capabilities:
        agent.add_capability(capability)

    # Test both tools
    result1 = await agent.handle_tool_route("tool1", {
        "args": {"input": "test1"}
    })
    assert result1 == {"result": "test1"}

    result2 = await agent.handle_tool_route("tool2", {
        "args": {"input": "test2"}
    })
    assert result2 == {"result": "test2"}

def test_duplicate_capability(mock_api_key):
    agent = Agent(AgentOptions(
        api_key=mock_api_key,
        system_prompt="You are a test agent",
        openai_api_key="test-openai-key"
    ))

    async def test_run(run_params, messages):
        return run_params["args"].input

    agent.add_capability(Capability(
        name="test",
        description="Tool 1",
        schema=TestInput,
        run=test_run
    ))

    with pytest.raises(ValueError) as exc_info:
        agent.add_capability(Capability(
            name="test",
            description="Tool 1 duplicate",
            schema=TestInput,
            run=test_run
        ))
    
    assert str(exc_info.value) == 'Tool with name "test" already exists'

def test_duplicate_capabilities_in_list(mock_api_key):
    agent = Agent(AgentOptions(
        api_key=mock_api_key,
        system_prompt="You are a test agent",
        openai_api_key="test-openai-key"
    ))

    async def test_run(run_params, messages):
        return run_params["args"].input

    capabilities = [
        Capability(
            name="tool1",
            description="Tool 1",
            schema=TestInput,
            run=test_run
        ),
        Capability(
            name="tool1",
            description="Tool 1 duplicate",
            schema=TestInput,
            run=test_run
        )
    ]

    with pytest.raises(ValueError) as exc_info:
        for capability in capabilities:
            agent.add_capability(capability)
    
    assert str(exc_info.value) == 'Tool with name "tool1" already exists' 
