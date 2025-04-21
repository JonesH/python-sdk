import pytest
from unittest.mock import MagicMock, patch
from src.agent import Agent
from src.types import AgentOptions, DoTaskAction, RespondChatMessageAction, AgentKind, TaskStatus, Workspace, AgentBase, Task

# Create a test class that exposes protected methods for testing
class TestAgent(Agent):
    async def test_do_task(self, action: DoTaskAction):
        return await self.do_task(action)

    async def test_respond_to_chat(self, action: RespondChatMessageAction):
        return await self.respond_to_chat(action)

    @property
    def test_openai(self):
        return self._openai

    @test_openai.setter
    def test_openai(self, client):
        self._openai = client

@pytest.fixture
def mock_api_key():
    return "test-openserv-key"

def test_required_api_methods(mock_api_key):
    agent = Agent(AgentOptions(
        api_key=mock_api_key,
        system_prompt="You are a test agent"
    ))

    required_methods = [
        'upload_file',
        'update_task_status',
        'complete_task',
        'mark_task_as_errored',
        'add_log_to_task',
        'request_human_assistance',
        'send_chat_message',
        'create_task',
        'get_task_detail',
        'get_agents',
        'get_tasks',
        'get_files',
        'process',
        'start',
        'add_capability'
    ]

    for method in required_methods:
        assert hasattr(agent, method), f"{method} should be a method"
        assert callable(getattr(agent, method)), f"{method} should be callable"

@pytest.mark.asyncio
async def test_process_without_openai_key(mock_api_key):
    with patch.dict('os.environ', {}, clear=True):  # Clear OPENAI_API_KEY from env
        agent = Agent(AgentOptions(
            api_key=mock_api_key,
            system_prompt="You are a test agent"
        ))

        with pytest.raises(Exception) as exc_info:
            await agent.process({"messages": [{"role": "user", "content": "test message"}]})
        
        assert str(exc_info.value) == "OpenAI API key is required"

def test_start_method_available(mock_api_key):
    agent = Agent(AgentOptions(
        api_key=mock_api_key,
        system_prompt="You are a test agent"
    ))
    assert hasattr(agent, "start")
    assert callable(agent.start)

@pytest.mark.asyncio
async def test_custom_error_handler(mock_api_key):
    handled_error = None
    handled_context = None

    def error_handler(error, context):
        nonlocal handled_error, handled_context
        handled_error = error
        handled_context = context

    agent = Agent(AgentOptions(
        api_key=mock_api_key,
        system_prompt="You are a test agent",
        on_error=error_handler
    ))

    try:
        await agent.handle_tool_route({
            "params": {"toolName": "nonexistent"},
            "body": {}
        })
        pytest.fail("Expected error to be thrown")
    except Exception as error:
        assert isinstance(error, ValueError)
        assert isinstance(handled_error, Exception)
        assert handled_context["context"] == "handle_tool_route"

@pytest.mark.asyncio
async def test_process_method_error_handling(mock_api_key):
    handled_error = None
    handled_context = None

    def error_handler(error, context):
        nonlocal handled_error, handled_context
        handled_error = error
        handled_context = context

    agent = TestAgent(AgentOptions(
        api_key=mock_api_key,
        system_prompt="You are a test agent",
        openai_api_key="test-key",
        on_error=error_handler
    ))

    # Mock OpenAI to throw an error
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.side_effect = Exception("OpenAI error")
    agent.test_openai = mock_openai

    try:
        await agent.process({"messages": [{"role": "user", "content": "test"}]})
    except Exception:
        pass

    assert isinstance(handled_error, Exception)
    assert str(handled_error) == "OpenAI error"
    assert handled_context["context"] == "process"

@pytest.mark.asyncio
async def test_do_task_error_handling(mock_api_key):
    handled_error = None
    handled_context = None

    def error_handler(error, context):
        nonlocal handled_error, handled_context
        handled_error = error
        handled_context = context

    agent = TestAgent(AgentOptions(
        api_key=mock_api_key,
        system_prompt="You are a test agent",
        on_error=error_handler
    ))

    test_action = DoTaskAction(
        type="do-task",
        workspace=Workspace(
            id=1,
            goal="Test workspace",
            bucket_folder="test",
            agents=[]
        ),
        me=AgentBase(
            id=1,
            name="Test Agent",
            kind=AgentKind.EXTERNAL,
            isBuiltByAgentBuilder=False
        ),
        task=Task(
            id=1,
            description="test task",
            dependencies=[],
            humanAssistanceRequests=[]
        ),
        integrations=[],
        memories=[]
    )

    await agent.test_do_task(test_action)

    assert isinstance(handled_error, Exception)
    assert handled_context["context"] == "do_task"
    assert handled_context["action"] == test_action

@pytest.mark.asyncio
async def test_respond_to_chat_error_handling(mock_api_key):
    handled_error = None
    handled_context = None

    def error_handler(error, context):
        nonlocal handled_error, handled_context
        handled_error = error
        handled_context = context

    agent = TestAgent(AgentOptions(
        api_key=mock_api_key,
        system_prompt="You are a test agent",
        on_error=error_handler
    ))
    
    # Create a mock runtime client
    mock_runtime_client = MagicMock()
    mock_runtime_client.handle_chat = MagicMock()
    agent.runtime_client = mock_runtime_client

    test_action = RespondChatMessageAction(
        type="respond-chat-message",
        workspace=Workspace(
            id=1,
            goal="Test workspace",
            bucket_folder="test",
            agents=[]
        ),
        me=AgentBase(
            id=1,
            name="Test Agent",
            kind=AgentKind.EXTERNAL,
            isBuiltByAgentBuilder=False
        ),
        integrations=[],
        memories=[],
        messages=[]
    )

    await agent.test_respond_to_chat(test_action)
    
    # Assert that the runtime client's handle_chat method was called with the right parameters
    mock_runtime_client.handle_chat.assert_called_once()
    call_args = mock_runtime_client.handle_chat.call_args[1]
    assert 'single_use' in call_args
    assert call_args['single_use'] is True
    assert 'action' in call_args
    assert 'messages' in call_args
    assert 'tools' in call_args 
