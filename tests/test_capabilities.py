import pytest
from pydantic import BaseModel
from openserv_sdk import Agent, AgentOptions, Capability

@pytest.fixture
def mock_api_key():
    return "test-openserv-key"

class TestInput(BaseModel):
    input: str

@pytest.mark.asyncio
async def test_execute_capability(mock_api_key):
    agent = Agent(AgentOptions(
        api_key=mock_api_key,
        system_prompt="You are a test agent"
    ))

    async def test_run(params, messages):
        return params["args"]["input"]

    capability = Capability(
        name="testCapability",
        description="A test capability",
        schema=TestInput,
        run=test_run
    )

    agent.add_capability(capability)

    result = await agent.handle_tool_route(
        "testCapability",
        {"args": {"input": "test"}, "messages": [], "action": None}
    )

    assert result == "test"

def test_validate_capability_schema(mock_api_key):
    class TestNumberInput(BaseModel):
        input: int

    async def test_run(params, messages):
        return str(params["args"]["input"])

    capability = Capability(
        name="testCapability",
        description="A test capability",
        schema=TestNumberInput,
        run=test_run
    )

    with pytest.raises(ValueError):
        capability.schema.model_validate({"input": "not a number"})

@pytest.mark.asyncio
async def test_handle_multiple_capabilities(mock_api_key):
    agent = Agent(AgentOptions(
        api_key=mock_api_key,
        system_prompt="You are a test agent"
    ))

    async def test_run(params, messages):
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

def test_duplicate_capability(mock_api_key):
    agent = Agent(AgentOptions(
        api_key=mock_api_key,
        system_prompt="You are a test agent"
    ))

    async def test_run(params, messages):
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

def test_duplicate_capabilities_in_list(mock_api_key):
    agent = Agent(AgentOptions(
        api_key=mock_api_key,
        system_prompt="You are a test agent"
    ))

    async def test_run(params, messages):
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
