import pytest
from unittest.mock import AsyncMock
import aiohttp

from openserv_sdk.agent import Agent
from openserv_sdk.types import (
    AgentOptions, UploadFileParams, GetFilesParams, CreateTaskParams, UpdateTaskStatusParams,
    GetTasksParams
)

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
    files = await agent.get_files(GetFilesParams(workspace_id=1))
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
    agent._api_client.get.assert_called_once_with("/workspaces/1/files")
    
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

    # Mock API client responses with side effects
    agent._api_client.get = AsyncMock(return_value={"data": []})
    
    # Create different responses for different POST calls
    post_responses = [
        {"data": {"id": 1}},  # For task creation
        {"data": {"status": "success"}}  # For status update
    ]
    post_mock = AsyncMock(side_effect=post_responses)
    agent._api_client.post = post_mock
    
    agent._api_client.put = AsyncMock(return_value={"data": {"status": "success"}})

    # Test task listing
    tasks = await agent.get_tasks(GetTasksParams(workspace_id=1))
    assert tasks == []

    # Test task creation
    task = await agent.create_task(CreateTaskParams(
        workspace_id=1,
        assignee=2,
        description="Test task",
        body="Test body",
        input="Test input",
        expected_output="Test output",
        dependencies=[]
    ))
    assert task == {"id": 1}

    # Test task status update
    status_update = await agent.update_task_status(UpdateTaskStatusParams(
        workspace_id=1,
        task_id=1,
        status="in-progress"
    ))
    assert status_update == {"status": "success"} 