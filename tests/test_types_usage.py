import pytest
from openserv_sdk.types import (
    AgentOptions, ProcessParams, RespondChatMessageAction,
    DoTaskAction, IntegrationCallRequest,
    GetTasksParams, GetTaskDetailParams, GetAgentsParams,
    UploadFileParams, CreateTaskParams, AddLogToTaskParams,
    RequestHumanAssistanceParams, UpdateTaskStatusParams,
    SendChatMessageParams, TaskStatus
)

@pytest.mark.asyncio
async def test_agent_options():
    """Test AgentOptions type."""
    options = AgentOptions(
        system_prompt="Test prompt",
        api_key="test-key",
        openai_api_key="test-openai-key",
        port=7378
    )
    assert options.system_prompt == "Test prompt"
    assert options.api_key == "test-key"
    assert options.openai_api_key == "test-openai-key"
    assert options.port == 7378

@pytest.mark.asyncio
async def test_process_params():
    """Test ProcessParams type."""
    params = ProcessParams(messages=[
        {"role": "user", "content": "Hello"}
    ])
    assert len(params.messages) == 1
    assert params.messages[0]["role"] == "user"
    assert params.messages[0]["content"] == "Hello"

@pytest.mark.asyncio
async def test_respond_chat_message_action():
    """Test RespondChatMessageAction type."""
    action = RespondChatMessageAction(
        type="respond-chat-message",
        me={"id": 1, "name": "test-agent", "kind": "external", "isBuiltByAgentBuilder": False},
        messages=[{"id": 1, "author": "user", "message": "Hello", "createdAt": "2024-01-01T00:00:00Z"}],
        workspace={"id": 1, "goal": "test", "bucket_folder": "test", "agents": []},
        integrations=[],
        memories=[]
    )
    assert action.type == "respond-chat-message"
    assert action.me.id == 1
    assert len(action.messages) == 1

@pytest.mark.asyncio
async def test_do_task_action():
    """Test DoTaskAction type."""
    action = DoTaskAction(
        type="do-task",
        me={"id": 1, "name": "test-agent", "kind": "external", "isBuiltByAgentBuilder": False},
        task={"id": 1, "description": "test", "body": "test", "expectedOutput": "test", "input": "test", "dependencies": [], "humanAssistanceRequests": []},
        workspace={"id": 1, "goal": "test", "bucket_folder": "test", "agents": []},
        integrations=[],
        memories=[]
    )
    assert action.type == "do-task"
    assert action.task.id == 1
    assert action.workspace.id == 1

@pytest.mark.asyncio
async def test_get_tasks_params():
    """Test GetTasksParams type."""
    params = GetTasksParams(workspace_id=1)
    assert params.workspace_id == 1

@pytest.mark.asyncio
async def test_get_task_detail_params():
    """Test GetTaskDetailParams type."""
    params = GetTaskDetailParams(workspace_id=1, task_id=2)
    assert params.workspace_id == 1
    assert params.task_id == 2

@pytest.mark.asyncio
async def test_get_agents_params():
    """Test GetAgentsParams type."""
    params = GetAgentsParams(workspace_id=1)
    assert params.workspace_id == 1

@pytest.mark.asyncio
async def test_upload_file_params():
    """Test UploadFileParams type."""
    params = UploadFileParams(
        workspace_id=1,
        path="test.txt",
        file="test content",
        task_ids=[1, 2],
        skip_summarizer=True
    )
    assert params.workspace_id == 1
    assert params.path == "test.txt"
    assert params.file == "test content"
    assert params.task_ids == [1, 2]
    assert params.skip_summarizer is True

@pytest.mark.asyncio
async def test_create_task_params():
    """Test CreateTaskParams type."""
    params = CreateTaskParams(
        workspace_id=1,
        assignee=2,
        description="Test task",
        body="Test body",
        input="Test input",
        expected_output="Test output",
        dependencies=[3, 4]
    )
    assert params.workspace_id == 1
    assert params.assignee == 2
    assert params.description == "Test task"
    assert params.dependencies == [3, 4]

@pytest.mark.asyncio
async def test_add_log_to_task_params():
    """Test AddLogToTaskParams type."""
    params = AddLogToTaskParams(
        workspace_id=1,
        task_id=2,
        severity="info",
        type="text",
        body="Test log message"
    )
    assert params.workspace_id == 1
    assert params.task_id == 2
    assert params.severity == "info"
    assert params.type == "text"

@pytest.mark.asyncio
async def test_request_human_assistance_params():
    """Test RequestHumanAssistanceParams type."""
    params = RequestHumanAssistanceParams(
        workspace_id=1,
        task_id=2,
        type="text",
        question="test question",
        agent_dump={"key": "value"}
    )
    assert params.workspace_id == 1
    assert params.task_id == 2
    assert params.type == "text"
    assert params.question == "test question"

@pytest.mark.asyncio
async def test_update_task_status_params():
    """Test UpdateTaskStatusParams type."""
    params = UpdateTaskStatusParams(
        workspace_id=1,
        task_id=2,
        status=TaskStatus.IN_PROGRESS
    )
    assert params.workspace_id == 1
    assert params.task_id == 2
    assert params.status == TaskStatus.IN_PROGRESS

@pytest.mark.asyncio
async def test_send_chat_message_params():
    """Test SendChatMessageParams type."""
    params = SendChatMessageParams(
        workspace_id=1,
        agent_id=2,
        message="Test message"
    )
    assert params.workspace_id == 1
    assert params.agent_id == 2
    assert params.message == "Test message"

@pytest.mark.asyncio
async def test_integration_call_request():
    """Test IntegrationCallRequest type."""
    request = IntegrationCallRequest(
        workspace_id=1,
        integration_id="test-integration",
        details={"method": "GET", "endpoint": "/api/test"}
    )
    assert request.workspace_id == 1
    assert request.integration_id == "test-integration"
    assert request.details.method == "GET"
    assert request.details.endpoint == "/api/test" 