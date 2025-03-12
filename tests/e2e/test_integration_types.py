import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp
from datetime import datetime
from openserv_sdk.agent import Agent
from openserv_sdk.types import (
    AgentOptions, UploadFileParams, GetTasksParams,
    CreateTaskParams, UpdateTaskStatusParams, TaskStatus,
    SendChatMessageParams, GetTaskDetailParams, GetAgentsParams,
    AddLogToTaskParams, RequestHumanAssistanceParams,
    ProcessParams, RespondChatMessageAction, DoTaskAction,
    IntegrationCallRequest, AgentBase, Task, Workspace,
    AgentKind, ChatMessage
)

@pytest.fixture
def mock_openai():
    """Mock OpenAI client."""
    with patch('openai.AsyncOpenAI') as mock:
        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices = [
            AsyncMock(
                message=AsyncMock(
                    content="Test response",
                    tool_calls=None,
                    model_dump=lambda: {
                        "content": "Test response",
                        "role": "assistant",
                        "tool_calls": None
                    }
                )
            )
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        mock.return_value = mock_client
        yield mock_client

@pytest.fixture
def test_agent(mock_openai):
    """Create a test agent with mocked clients."""
    agent = Agent(AgentOptions(
        system_prompt="Test Agent",
        api_key="test-key",
        openai_api_key="test-openai-key",
        platform_url="https://api.test.com",
        runtime_url="https://runtime.test.com",
        port=7378,
        host="localhost",
        log_level="debug",
        reload=True
    ))
    # Mock the API client
    agent.api_client = AsyncMock()
    agent.runtime_client = AsyncMock()
    agent._openai_client = mock_openai
    return agent

@pytest.mark.asyncio
async def test_upload_file_workflow(test_agent):
    """Test the complete file upload workflow."""
    # Mock the API response
    test_agent.api_client.post.return_value = {
        "data": {"file_id": "test-file-123"}
    }
    
    # Test with string content
    text_params = UploadFileParams(
        workspace_id=1,
        path="test.txt",
        file="Test file content",
        task_ids=[1, 2],
        skip_summarizer=True
    )
    result = await test_agent.upload_file(text_params)
    assert result == {"file_id": "test-file-123"}
    
    # Test with binary content
    binary_params = UploadFileParams(
        workspace_id=1,
        path="test.bin",
        file=b"Binary content",
        task_ids=1,  # Test single task ID
        skip_summarizer=False
    )
    result = await test_agent.upload_file(binary_params)
    assert result == {"file_id": "test-file-123"}

@pytest.mark.asyncio
async def test_task_operations_workflow(test_agent):
    """Test all task-related operations."""
    # Mock responses for all API calls
    test_agent.api_client.post.side_effect = [
        {"data": {"task_id": "task-123"}},  # Create task
        {"data": {"status": "success"}},   # First status update
        {"data": {"status": "success"}},   # Second status update
        {"data": {"status": "success"}},   # Third status update
        {"data": {"status": "success"}},   # First log
        {"data": {"status": "success"}}    # Second log
    ]
    test_agent.api_client.get.return_value = {
        "data": {"task": {"id": 1, "status": TaskStatus.IN_PROGRESS}}
    }
    
    # Create task with all fields
    create_params = CreateTaskParams(
        workspace_id=1,
        assignee=2,
        description="Test task",
        body="Test body",
        input="Test input",
        expected_output="Expected output",
        dependencies=[3, 4]
    )
    create_result = await test_agent.create_task(create_params)
    assert create_result == {"task_id": "task-123"}
    
    # Get task details
    detail_params = GetTaskDetailParams(workspace_id=1, task_id=2)
    detail_result = await test_agent.get_task_detail(detail_params)
    assert detail_result["task"]["status"] == TaskStatus.IN_PROGRESS
    
    # Verify get_task_detail was called with correct URL format
    test_agent.api_client.get.assert_called_with("/workspaces/1/task/2/detail")
    
    # Update task status (test different statuses)
    for status in [TaskStatus.IN_PROGRESS, TaskStatus.DONE, TaskStatus.ERROR]:
        update_params = UpdateTaskStatusParams(
            workspace_id=1,
            task_id=2,
            status=status
        )
        await test_agent.update_task_status(update_params)
    
    # Add log with different types
    log_params = AddLogToTaskParams(
        workspace_id=1,
        task_id=2,
        severity="info",
        type="text",
        body="Test log message"
    )
    await test_agent.add_log_to_task(log_params)

    openai_log_params = AddLogToTaskParams(
        workspace_id=1,
        task_id=2,
        severity="warning",
        type="openai-message",
        body={"role": "assistant", "content": "Test"}
    )
    await test_agent.add_log_to_task(openai_log_params)
    
    # Verify correct URL formats were used
    test_agent.api_client.post.assert_any_call(
        "/workspaces/1/task/2/status",
        {"status": TaskStatus.ERROR.value}
    )
    
    test_agent.api_client.post.assert_any_call(
        "/workspaces/1/task/2/log",
        {
            "severity": "info",
            "type": "text",
            "body": "Test log message"
        }
    )

@pytest.mark.asyncio
async def test_process_and_chat_workflow(test_agent):
    """Test processing and chat operations."""
    # Mock responses
    test_agent.api_client.post.return_value = {"data": {"status": "success"}}
    
    # Test process with different message types
    process_params = ProcessParams(messages=[
        {"role": "system", "content": "You are a test agent"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "tool", "content": "Tool response"}
    ])
    result = await test_agent.process(process_params)
    assert result == {"result": "Test response"}
    
    # Test respond to chat message
    chat_action = RespondChatMessageAction(
        type="respond-chat-message",
        me=AgentBase(
            id=1,
            name="test-agent",
            kind=AgentKind.EXTERNAL,
            is_built_by_agent_builder=False
        ),
        messages=[
            ChatMessage(
                id=1,
                author="user",
                message="Hello",
                created_at=datetime.now()
            )
        ],
        workspace=Workspace(
            id=1,
            goal="test goal",
            bucket_folder="test",
            agents=[]
        ),
        integrations=[],
        memories=[]
    )
    await test_agent.respond_to_chat(chat_action)

@pytest.mark.asyncio
async def test_task_execution_workflow(test_agent):
    """Test task execution workflow."""
    # Mock runtime client response
    test_agent.runtime_client.execute_task.return_value = {"data": {"status": "success"}}
    test_agent.api_client.post.return_value = {"data": {"status": "success"}}
    
    # Create task action
    task_action = DoTaskAction(
        type="do-task",
        me=AgentBase(
            id=1,
            name="test-agent",
            kind=AgentKind.EXTERNAL,
            is_built_by_agent_builder=False
        ),
        task=Task(
            id=1,
            description="test task",
            body="test body",
            expected_output="test output",
            input="test input",
            dependencies=[],
            human_assistance_requests=[]
        ),
        workspace=Workspace(
            id=1,
            goal="test goal",
            bucket_folder="test",
            agents=[]
        ),
        integrations=[],
        memories=[]
    )

    # Execute task and await all responses
    await test_agent.do_task(task_action)
    
    # Verify runtime client call
    test_agent.runtime_client.execute_task.assert_called_once()
    call_args = test_agent.runtime_client.execute_task.call_args
    assert call_args[1]["workspace_id"] == 1
    assert call_args[1]["task_id"] == 1
    assert "tools" in call_args[1]
    assert "messages" in call_args[1]
    assert "action" in call_args[1]
    
    # Verify status update was called with correct URL format
    test_agent.api_client.post.assert_called_with(
        "/workspaces/1/task/1/status",
        {"status": "in-progress"}
    )
    
    # Test error handling
    test_agent.runtime_client.execute_task.side_effect = Exception("Test error")
    
    # Reset the post mock to track new calls
    test_agent.api_client.post.reset_mock()
    
    # Execute task again, this time it should fail
    try:
        await test_agent.do_task(task_action)
    except:
        pass  # We expect an exception
    
    # Verify mark_task_as_errored was called with correct URL format
    test_agent.api_client.post.assert_any_call(
        "/workspaces/1/task/1/error",
        {"error": "Test error"}
    )

@pytest.mark.asyncio
async def test_human_assistance_workflow(test_agent):
    """Test human assistance workflow."""
    test_agent.api_client.post.return_value = {"data": {"status": "success"}}
    
    # Test text question
    text_params = RequestHumanAssistanceParams(
        workspace_id=1,
        task_id=2,
        type="text",
        question="Need help with this task",
        agent_dump={"state": "current state"}
    )
    await test_agent.request_human_assistance(text_params)
    
    # Test plan review
    plan_params = RequestHumanAssistanceParams(
        workspace_id=1,
        task_id=2,
        type="project-manager-plan-review",
        question={"plan": "detailed plan"},
        agent_dump={"state": "planning state"}
    )
    await test_agent.request_human_assistance(plan_params)
    
    # Verify correct URL format was used
    test_agent.api_client.post.assert_any_call(
        "/workspaces/1/task/2/human-assistance",
        {
            "type": "text",
            "question": "Need help with this task",
            "agentDump": {"state": "current state"}
        }
    )

@pytest.mark.asyncio
async def test_integration_workflow(test_agent):
    """Test integration workflow."""
    test_agent.api_client.post.return_value = {"data": {"result": "success"}}
    
    # Test integration call with all fields
    request = IntegrationCallRequest(
        workspace_id=1,
        integration_id="test-integration",
        details={
            "method": "POST",
            "endpoint": "/api/test",
            "headers": {"Authorization": "Bearer token"},
            "params": {"key": "value"},
            "data": {"body": "content"},
            "retries": 3,
            "base_url_override": "https://api.example.com",
            "decompress": True,
            "response_type": "json",
            "retry_on": [500, 503]
        }
    )
    await test_agent.call_integration(request)

@pytest.mark.asyncio
async def test_task_listing_workflow(test_agent):
    """Test the task listing workflow."""
    # Mock the API response
    test_agent.api_client.get.return_value = {
        "data": {
            "tasks": [
                {
                    "id": 1,
                    "description": "Task 1",
                    "status": TaskStatus.TODO
                },
                {
                    "id": 2,
                    "description": "Task 2",
                    "status": TaskStatus.IN_PROGRESS
                }
            ]
        }
    }
    
    # Get tasks
    params = GetTasksParams(workspace_id=1)
    result = await test_agent.get_tasks(params)
    
    # Verify the API call
    test_agent.api_client.get.assert_called_once_with("/workspaces/1/tasks")
    
    # Verify the response structure
    assert "tasks" in result
    assert len(result["tasks"]) == 2
    assert result["tasks"][0]["id"] == 1
    assert result["tasks"][1]["status"] == TaskStatus.IN_PROGRESS