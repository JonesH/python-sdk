from enum import Enum
from typing import Optional, List, Dict, Any, Union, Literal, Callable
from pydantic import BaseModel, Field, validator
from datetime import datetime
from dataclasses import dataclass

# Common Config class for all models
class CommonConfig:
    """Common configuration for all models to handle camelCase serialization."""
    alias_generator = lambda s: ''.join(word.capitalize() if i else word for i, word in enumerate(s.split('_')))
    populate_by_name = True

class AgentKind(str, Enum):
    EXTERNAL = 'external'
    ELIZA = 'eliza'
    OPENSERV = 'openserv'

class TaskStatus(str, Enum):
    TODO = 'to-do'
    IN_PROGRESS = 'in-progress'
    HUMAN_ASSISTANCE_REQUIRED = 'human-assistance-required'
    ERROR = 'error'
    DONE = 'done'
    CANCELLED = 'cancelled'

class AgentBase(BaseModel):
    id: int
    name: str
    kind: AgentKind
    is_built_by_agent_builder: bool = Field(False, alias="isBuiltByAgentBuilder")
    system_prompt: Optional[str] = Field(None, alias="systemPrompt")
    capabilities_description: Optional[str] = None

    class Config(CommonConfig):
        pass

class Agent(BaseModel):
    id: int
    name: str
    kind: str = "openserv"
    capabilities_description: Optional[str] = None

    class Config(CommonConfig):
        pass

class TaskAttachment(BaseModel):
    id: int
    path: str
    full_url: str = Field(..., alias="fullUrl")
    summary: Optional[str] = None

    class Config(CommonConfig):
        pass

class TaskDependency(BaseModel):
    id: int
    description: str
    output: Optional[str] = None
    status: TaskStatus
    attachments: List[TaskAttachment] = []

    class Config(CommonConfig):
        pass

class HumanAssistanceRequest(BaseModel):
    id: int
    agent_dump: Optional[Any] = Field(None, alias="agentDump")
    human_response: Optional[str] = Field(None, alias="humanResponse")
    question: str
    status: Literal['pending', 'responded']
    type: Literal['text', 'project-manager-plan-review']

    class Config(CommonConfig):
        pass

class Task(BaseModel):
    id: int
    description: str
    body: Optional[str] = None
    expected_output: Optional[str] = Field(None, alias="expectedOutput")
    input: Optional[str] = None
    dependencies: List[TaskDependency] = []
    human_assistance_requests: List[HumanAssistanceRequest] = Field([], alias="humanAssistanceRequests")

    class Config(CommonConfig):
        pass

class Workspace(BaseModel):
    id: int
    goal: str
    bucket_folder: str
    agents: List[Agent]

    class Config(CommonConfig):
        pass

class Integration(BaseModel):
    id: int
    connection_id: str = Field(..., alias="connectionId")
    provider_config_key: str = Field(..., alias="providerConfigKey")
    provider: str
    created: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    scopes: Optional[List[str]] = None
    open_api: Dict[str, str] = Field(..., alias="openAPI")

    class Config(CommonConfig):
        pass

class Memory(BaseModel):
    id: int
    memory: str
    created_at: datetime = Field(..., alias="createdAt")

    class Config(CommonConfig):
        pass

class AgentAction(BaseModel):
    type: Literal['do-task', 'respond-chat-message']
    me: AgentBase
    task: Optional[Task] = None
    workspace: Workspace
    integrations: List[Integration] = []
    memories: List[Memory] = []

    class Config(CommonConfig):
        pass

class DoTaskAction(AgentAction):
    type: Literal['do-task']
    task: Task

    class Config(CommonConfig):
        pass

class ChatMessage(BaseModel):
    author: Literal['user', 'agent']
    message: str
    id: int
    created_at: datetime = Field(..., alias="createdAt")

    class Config(CommonConfig):
        pass

class RespondChatMessageAction(AgentAction):
    type: Literal['respond-chat-message']
    me: AgentBase
    messages: List[ChatMessage]
    workspace: Workspace
    integrations: List[Integration] = []
    memories: List[Memory] = []

    class Config(CommonConfig):
        pass

class ProcessParams(BaseModel):
    messages: List[Dict[str, str]]

    class Config(CommonConfig):
        pass

class AgentOptions(BaseModel):
    """Configuration options for creating a new Agent instance."""
    system_prompt: str
    api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    model: Optional[str] = "gpt-4"
    port: Optional[int] = 7378
    host: Optional[str] = '0.0.0.0'
    log_level: Optional[str] = 'debug'
    reload: Optional[bool] = False
    platform_url: Optional[str] = 'https://api.openserv.ai'
    runtime_url: Optional[str] = 'https://agents.openserv.ai'
    on_error: Optional[Callable[[Exception, Dict[str, Any]], None]] = None

    class Config(CommonConfig):
        pass

class GetFilesParams(BaseModel):
    workspace_id: int = Field(..., gt=0, alias="workspaceId", description="Workspace ID must be a positive integer")

    class Config(CommonConfig):
        pass

@dataclass
class ListFilesParams:
    """Parameters for listing files in a workspace."""
    workspace_id: int

class FileContent(BaseModel):
    """File content that can be uploaded to the workspace."""
    content: Union[str, bytes]
    file_name: str = Field(..., alias="fileName")
    content_type: Optional[str] = Field(None, alias="contentType")

    class Config(CommonConfig):
        pass

class UploadedFile(BaseModel):
    """Represents a file that has been uploaded to the workspace."""
    id: str
    path: str
    url: str
    content: Optional[Union[str, bytes]] = None
    content_type: str = Field(..., alias="contentType")
    file_name: str = Field(..., alias="fileName")
    size: int
    summary: Optional[str] = None

    class Config(CommonConfig):
        pass

class UploadFileParams(BaseModel):
    """Parameters for uploading a file to a workspace."""
    workspace_id: int = Field(..., gt=0, alias="workspaceId", description="ID of the workspace to upload to")
    path: str = Field(..., description="Path/name for the file in the workspace")
    file: Union[str, bytes] = Field(..., description="File content as string or bytes")
    task_ids: Optional[Union[int, List[int]]] = Field(None, alias="taskIds", description="Task ID(s) to associate with the file")
    skip_summarizer: Optional[bool] = Field(None, alias="skipSummarizer", description="Whether to skip content summarization")
    content_type: Optional[str] = Field(None, alias="contentType", description="Override automatic content type detection")

    class Config(CommonConfig):
        json_schema_extra = {
            "examples": [
                {
                    "workspaceId": 1,
                    "path": "example.txt",
                    "file": "Hello, World!",
                    "taskIds": [1, 2],
                    "skipSummarizer": False
                }
            ]
        }

class MarkTaskAsErroredParams(BaseModel):
    workspace_id: int = Field(..., alias="workspaceId")
    task_id: int = Field(..., alias="taskId")
    error: str

    class Config(CommonConfig):
        pass

class CompleteTaskParams(BaseModel):
    workspace_id: int = Field(..., alias="workspaceId")
    task_id: int = Field(..., alias="taskId")
    output: str

    class Config(CommonConfig):
        pass

class SendChatMessageParams(BaseModel):
    workspace_id: int = Field(..., alias="workspaceId")
    agent_id: int = Field(..., alias="agentId")
    message: str

    class Config(CommonConfig):
        pass

class GetTaskDetailParams(BaseModel):
    workspace_id: int = Field(..., alias="workspaceId")
    task_id: int = Field(..., alias="taskId")

    class Config(CommonConfig):
        pass

class GetAgentsParams(BaseModel):
    workspace_id: int = Field(..., alias="workspaceId")

    class Config(CommonConfig):
        pass

class GetTasksParams(BaseModel):
    workspace_id: int = Field(..., alias="workspaceId")

    class Config(CommonConfig):
        pass

class CreateTaskParams(BaseModel):
    workspace_id: int = Field(..., alias="workspaceId")
    assignee: int
    description: str
    body: str
    input: str
    expected_output: str = Field(..., alias="expectedOutput")
    dependencies: List[int]

    class Config(CommonConfig):
        pass

class AddLogToTaskParams(BaseModel):
    """Parameters for adding a log entry to a task."""
    workspace_id: int = Field(..., gt=0, alias="workspaceId", description="Workspace ID must be a positive integer")
    task_id: int = Field(..., gt=0, alias="taskId", description="Task ID must be a positive integer")
    severity: Literal['info', 'warning', 'error'] = Field(..., description="Severity level of the log")
    type: Literal['text', 'openai-message'] = Field(..., description="Type of log entry")
    body: Union[str, dict] = Field(..., description="Log content (string for text, dict for openai-message)")

    @validator('body')
    def validate_body(cls, v, values):
        """Validate the body matches the specified type."""
        log_type = values.get('type')
        if log_type == 'text' and not isinstance(v, str):
            raise ValueError("Body must be a string for text type logs")
        elif log_type == 'openai-message' and not isinstance(v, dict):
            raise ValueError("Body must be a dictionary for openai-message type logs")
        
        if log_type == 'openai-message' and isinstance(v, dict):
            required_fields = ['role', 'content']
            if not all(field in v for field in required_fields):
                raise ValueError("OpenAI message must contain 'role' and 'content' fields")
            
            valid_roles = ['system', 'user', 'assistant', 'tool']
            if v.get('role') not in valid_roles:
                raise ValueError(f"Invalid role. Must be one of: {', '.join(valid_roles)}")
        
        return v

    class Config(CommonConfig):
        pass

class RequestHumanAssistanceParams(BaseModel):
    workspace_id: int = Field(..., alias="workspaceId")
    task_id: int = Field(..., alias="taskId")
    type: Literal['text', 'project-manager-plan-review']
    question: Union[str, dict]
    agent_dump: Optional[dict] = Field(None, alias="agentDump")

    class Config(CommonConfig):
        pass

class UpdateTaskStatusParams(BaseModel):
    """Parameters for updating a task's status."""
    workspace_id: int = Field(..., gt=0, alias="workspaceId", description="Workspace ID must be a positive integer")
    task_id: int = Field(..., gt=0, alias="taskId", description="Task ID must be a positive integer")
    status: TaskStatus = Field(..., description="New status for the task")

    @validator('status')
    def validate_status(cls, v):
        """Validate the task status is a valid enum value."""
        if not isinstance(v, TaskStatus):
            try:
                return TaskStatus(v)
            except ValueError:
                raise ValueError(f"Invalid status. Must be one of: {', '.join(TaskStatus.__members__.values())}")
        return v

    class Config(CommonConfig):
        pass

class ProxyConfiguration(BaseModel):
    endpoint: str
    provider_config_key: Optional[str] = Field(None, alias="providerConfigKey")
    connection_id: Optional[str] = Field(None, alias="connectionId")
    method: Optional[Literal['GET', 'POST', 'PATCH', 'PUT', 'DELETE', 'get', 'post', 'patch', 'put', 'delete']] = None
    headers: Optional[Dict[str, str]] = None
    params: Optional[Union[str, Dict[str, Union[str, int]]]] = None
    data: Optional[Any] = None
    retries: Optional[int] = None
    base_url_override: Optional[str] = Field(None, alias="baseUrlOverride")
    decompress: Optional[bool] = None
    response_type: Optional[Literal['arraybuffer', 'blob', 'document', 'json', 'text', 'stream']] = Field(None, alias="responseType")
    retry_on: Optional[List[int]] = Field(None, alias="retryOn")

    class Config(CommonConfig):
        pass

class IntegrationCallRequest(BaseModel):
    workspace_id: int = Field(..., alias="workspaceId")
    integration_id: str = Field(..., alias="integrationId")
    details: ProxyConfiguration

    class Config(CommonConfig):
        pass

class EngagementMetrics(BaseModel):
    """Social media engagement metrics."""
    likes: int = Field(ge=0)
    shares: int = Field(ge=0)
    comments: int = Field(ge=0)
    impressions: int = Field(ge=0)

    class Config(CommonConfig):
        pass

class SocialMediaPostParams(BaseModel):
    """Parameters for creating a social media post."""
    platform: str
    topic: str

    class Config(CommonConfig):
        pass

class AnalyzeEngagementParams(BaseModel):
    """Parameters for analyzing engagement metrics."""
    platform: str
    metrics: EngagementMetrics

    class Config(CommonConfig):
        pass 