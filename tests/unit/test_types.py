import pytest
from datetime import datetime
from pydantic import ValidationError
from openserv_sdk.types import (
    AgentKind,
    TaskStatus,
    DoTaskAction,
    RespondChatMessageAction,
    AgentBase,
    Task,
    Workspace,
    TaskDependency,
    TaskAttachment,
    HumanAssistanceRequest,
    ChatMessage,
    GetFilesParams,
    UploadFileParams,
    MarkTaskAsErroredParams,
    CompleteTaskParams,
    SendChatMessageParams,
    GetTaskDetailParams,
    GetAgentsParams,
    GetTasksParams,
    CreateTaskParams,
    AddLogToTaskParams,
    RequestHumanAssistanceParams,
    UpdateTaskStatusParams,
    ProcessParams,
    Integration,
    Memory,
    Agent
)

def test_validate_do_task_action():
    action = DoTaskAction(
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
            bucket_folder="test-folder",
            agents=[]
        ),
        integrations=[],
        memories=[]
    )

    assert action.type == "do-task"
    assert action.me.name == "test-agent"
    assert action.task.description == "test task"

def test_validate_respond_chat_message_action():
    action = RespondChatMessageAction(
        type="respond-chat-message",
        me=AgentBase(
            id=1,
            name="test-agent",
            kind=AgentKind.EXTERNAL,
            is_built_by_agent_builder=False
        ),
        messages=[
            ChatMessage(
                author="user",
                created_at=datetime.now(),
                id=1,
                message="test message"
            )
        ],
        workspace=Workspace(
            id=1,
            goal="test goal",
            bucket_folder="test-folder",
            agents=[]
        ),
        integrations=[],
        memories=[]
    )

    assert action.type == "respond-chat-message"
    assert action.me.name == "test-agent"
    assert len(action.messages) == 1
    assert action.messages[0].message == "test message"

def test_reject_invalid_action_type():
    with pytest.raises(ValidationError):
        DoTaskAction(
            type="invalid-type",
            me=AgentBase(
                id=1,
                name="test-agent",
                kind=AgentKind.EXTERNAL,
                is_built_by_agent_builder=False
            )
        )

def test_reject_invalid_do_task_action():
    with pytest.raises(ValidationError):
        DoTaskAction(
            type="do-task",
            me=AgentBase(
                id=1,
                name="test-agent",
                kind="invalid-kind",  # type: ignore
                is_built_by_agent_builder=False
            )
        )

def test_reject_invalid_respond_chat_message_action():
    with pytest.raises(ValidationError):
        RespondChatMessageAction(
            type="respond-chat-message",
            me=AgentBase(
                id=1,
                name="test-agent",
                kind=AgentKind.EXTERNAL,
                is_built_by_agent_builder=False
            ),
            messages=[
                ChatMessage(
                    author="invalid-author",  # type: ignore
                    created_at=datetime.now(),
                    id=1,
                    message="test message"
                )
            ]
        )

def test_validate_upload_file_params():
    # Test with array of taskIds
    params1 = UploadFileParams(
        workspace_id=1,
        path="test.txt",
        file="test content",
        task_ids=[1, 2, 3],
        skip_summarizer=True
    )
    assert params1.workspace_id == 1

    # Test with single taskId
    params2 = UploadFileParams(
        workspace_id=1,
        path="test.txt",
        file="test content",
        task_ids=1
    )
    assert params2.workspace_id == 1

    # Test with null taskIds
    params3 = UploadFileParams(
        workspace_id=1,
        path="test.txt",
        file="test content",
        task_ids=None
    )
    assert params3.workspace_id == 1

    # Test with skipSummarizer false
    params4 = UploadFileParams(
        workspace_id=1,
        path="test.txt",
        file="test content",
        skip_summarizer=False
    )
    assert params4.workspace_id == 1

    # Test with bytes file
    params5 = UploadFileParams(
        workspace_id=1,
        path="test.txt",
        file=b"test content"
    )
    assert params5.workspace_id == 1

    # Test with minimum required fields
    params6 = UploadFileParams(
        workspace_id=1,
        path="test.txt",
        file="test content"
    )
    assert params6.workspace_id == 1

def test_validate_get_files_params():
    """Test validation of GetFilesParams."""
    # Test valid case
    params = GetFilesParams(workspace_id=1)
    assert params.workspace_id == 1

    # Test invalid workspaceId types
    with pytest.raises(ValidationError):
        GetFilesParams(workspace_id="invalid")  # String instead of int

    with pytest.raises(ValidationError):
        GetFilesParams(workspace_id=-1)  # Negative integer not allowed

    with pytest.raises(ValidationError):
        GetFilesParams(workspace_id=0)  # Zero not allowed

    with pytest.raises(ValidationError):
        GetFilesParams()  # Missing required field

    with pytest.raises(ValidationError):
        GetFilesParams(workspace_id=None)  # None not allowed

def test_validate_mark_task_as_errored_params():
    params = MarkTaskAsErroredParams(
        workspace_id=1,
        task_id=2,
        error="Test error message"
    )
    assert params.workspace_id == 1
    assert params.task_id == 2
    assert params.error == "Test error message"

def test_validate_complete_task_params():
    params = CompleteTaskParams(
        workspace_id=1,
        task_id=2,
        output="Test task output"
    )
    assert params.workspace_id == 1
    assert params.task_id == 2
    assert params.output == "Test task output"

def test_validate_send_chat_message_params():
    params = SendChatMessageParams(
        workspace_id=1,
        agent_id=2,
        message="Test chat message"
    )
    assert params.workspace_id == 1
    assert params.agent_id == 2
    assert params.message == "Test chat message"

def test_validate_get_task_detail_params():
    params = GetTaskDetailParams(
        workspace_id=1,
        task_id=2
    )
    assert params.workspace_id == 1
    assert params.task_id == 2

def test_validate_agent_kind():
    assert AgentKind.EXTERNAL == "external"
    assert AgentKind.ELIZA == "eliza"
    assert AgentKind.OPENSERV == "openserv"
    with pytest.raises(ValueError):
        AgentKind("invalid")  # type: ignore

def test_validate_task_status():
    assert TaskStatus.TODO == "to-do"
    assert TaskStatus.IN_PROGRESS == "in-progress"
    assert TaskStatus.HUMAN_ASSISTANCE_REQUIRED == "human-assistance-required"
    assert TaskStatus.ERROR == "error"
    assert TaskStatus.DONE == "done"
    assert TaskStatus.CANCELLED == "cancelled"
    with pytest.raises(ValueError):
        TaskStatus("invalid")  # type: ignore

def test_validate_do_task_action_with_agent_builder():
    action = DoTaskAction(
        type="do-task",
        me=AgentBase(
            id=1,
            name="test-agent",
            kind=AgentKind.EXTERNAL,
            is_built_by_agent_builder=True,
            system_prompt="You are a test agent"
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
            bucket_folder="test-folder",
            agents=[]
        ),
        integrations=[],
        memories=[]
    )

    assert action.me.is_built_by_agent_builder is True
    assert action.me.system_prompt == "You are a test agent"

def test_validate_get_agents_params():
    params = GetAgentsParams(workspace_id=1)
    assert params.workspace_id == 1

def test_validate_get_tasks_params():
    params = GetTasksParams(workspace_id=1)
    assert params.workspace_id == 1

def test_validate_create_task_params():
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
    assert params.body == "Test body"
    assert params.input == "Test input"
    assert params.expected_output == "Test output"
    assert params.dependencies == [3, 4]

def test_validate_add_log_to_task_params():
    text_params = AddLogToTaskParams(
        workspace_id=1,
        task_id=2,
        severity="info",
        type="text",
        body="Test log message"
    )
    assert text_params.workspace_id == 1

    openai_params = AddLogToTaskParams(
        workspace_id=1,
        task_id=2,
        severity="warning",
        type="openai-message",
        body={"role": "assistant", "content": "Test message"}
    )
    assert openai_params.workspace_id == 1

def test_validate_request_human_assistance_params():
    text_params = RequestHumanAssistanceParams(
        workspace_id=1,
        task_id=2,
        type="text",
        question="Test question"
    )
    assert text_params.workspace_id == 1

    review_params = RequestHumanAssistanceParams(
        workspace_id=1,
        task_id=2,
        type="project-manager-plan-review",
        question={"plan": "Test plan"},
        agent_dump={"data": "Test data"}
    )
    assert review_params.workspace_id == 1

def test_validate_update_task_status_params():
    params = UpdateTaskStatusParams(
        workspace_id=1,
        task_id=2,
        status=TaskStatus.IN_PROGRESS
    )
    assert params.workspace_id == 1
    assert params.task_id == 2
    assert params.status == TaskStatus.IN_PROGRESS

def test_validate_process_params():
    params = ProcessParams(
        messages=[
            {"role": "user", "content": "Test message"},
            {"role": "assistant", "content": "Test response"}
        ]
    )
    assert len(params.messages) == 2
    assert params.messages[0]["role"] == "user"
    assert params.messages[1]["role"] == "assistant"

def test_validate_task_dependencies():
    action = DoTaskAction(
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
            dependencies=[
                TaskDependency(
                    id=2,
                    description="dependency task",
                    output="dependency output",
                    status=TaskStatus.DONE,
                    attachments=[
                        TaskAttachment(
                            id=3,
                            path="test.txt",
                            fullUrl="http://example.com/test.txt",
                            summary="test summary"
                        )
                    ]
                )
            ],
            human_assistance_requests=[]
        ),
        workspace=Workspace(
            id=1,
            goal="test goal",
            bucket_folder="test-folder",
            agents=[]
        ),
        integrations=[],
        memories=[]
    )

    assert action.task.dependencies[0].id == 2
    assert action.task.dependencies[0].attachments[0].path == "test.txt"

    # Test with nullish fields
    action_with_nullish = DoTaskAction(
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
            dependencies=[
                TaskDependency(
                    id=2,
                    description="dependency task",
                    output=None,
                    status=TaskStatus.DONE,
                    attachments=[
                        TaskAttachment(
                            id=3,
                            path="test.txt",
                            fullUrl="http://example.com/test.txt",
                            summary=None
                        )
                    ]
                )
            ],
            human_assistance_requests=[]
        ),
        workspace=Workspace(
            id=1,
            goal="test goal",
            bucket_folder="test-folder",
            agents=[]
        ),
        integrations=[],
        memories=[]
    )

    assert action_with_nullish.task.dependencies[0].output is None
    assert action_with_nullish.task.dependencies[0].attachments[0].summary is None

def test_validate_human_assistance_requests():
    action = DoTaskAction(
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
            human_assistance_requests=[
                HumanAssistanceRequest(
                    id=2,
                    type="text",
                    question="test question",
                    status="pending",
                    agent_dump={"data": "test data"},
                    human_response=None
                ),
                HumanAssistanceRequest(
                    id=3,
                    type="project-manager-plan-review",
                    question="test plan review",
                    status="responded",
                    agent_dump={"plan": "test plan"},
                    human_response="approved"
                )
            ]
        ),
        workspace=Workspace(
            id=1,
            goal="test goal",
            bucket_folder="test-folder",
            agents=[]
        ),
        integrations=[],
        memories=[]
    )

    assert len(action.task.human_assistance_requests) == 2
    assert action.task.human_assistance_requests[0].status == "pending"
    assert action.task.human_assistance_requests[1].human_response == "approved"

def test_validate_integrations():
    action = DoTaskAction(
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
            bucket_folder="test-folder",
            agents=[
                Agent(
                    id=2,
                    name="test agent",
                    kind=AgentKind.EXTERNAL,
                    capabilities_description="test capabilities"
                )
            ]
        ),
        integrations=[
            Integration(
                id=3,
                connection_id="test-connection",
                provider_config_key="test-provider",
                provider="test",
                created="2024-01-01",
                metadata={"key": "value"},
                scopes=["read", "write"],
                open_api={
                    "title": "Test API",
                    "description": "Test API description"
                }
            ),
            Integration(
                id=4,
                connection_id="test-connection-2",
                provider_config_key="test-provider-2",
                provider="test-2",
                created="2024-01-02",
                metadata=None,
                open_api={
                    "title": "Test API 2",
                    "description": "Test API description 2"
                }
            )
        ],
        memories=[
            Memory(
                id=5,
                memory="test memory",
                created_at=datetime(2024, 1, 1)
            )
        ]
    )

    assert len(action.integrations) == 2
    assert action.integrations[0].connection_id == "test-connection"
    assert action.integrations[1].provider == "test-2"
    assert len(action.memories) == 1

def test_validate_respond_chat_message_action_with_all_fields():
    action = RespondChatMessageAction(
        type="respond-chat-message",
        me=AgentBase(
            id=1,
            name="test-agent",
            kind=AgentKind.EXTERNAL,
            is_built_by_agent_builder=True,
            system_prompt="You are a test agent"
        ),
        messages=[
            ChatMessage(
                author="user",
                created_at=datetime.now(),
                id=1,
                message="test message"
            ),
            ChatMessage(
                author="agent",
                created_at=datetime.now(),
                id=2,
                message="test response"
            )
        ],
        workspace=Workspace(
            id=1,
            goal="test goal",
            bucket_folder="test-folder",
            agents=[
                Agent(
                    id=2,
                    name="test agent",
                    kind=AgentKind.EXTERNAL,
                    capabilities_description="test capabilities"
                )
            ]
        ),
        integrations=[
            Integration(
                id=3,
                connection_id="test-connection",
                provider_config_key="test-provider",
                provider="test",
                created="2024-01-01",
                metadata={"key": "value"},
                scopes=["read", "write"],
                open_api={
                    "title": "Test API",
                    "description": "Test API description"
                }
            )
        ],
        memories=[
            Memory(
                id=4,
                memory="test memory",
                created_at=datetime(2024, 1, 1)
            )
        ]
    )

    assert action.me.is_built_by_agent_builder is True
    assert len(action.messages) == 2
    assert len(action.workspace.agents) == 1
    assert len(action.integrations) == 1
    assert len(action.memories) == 1 
