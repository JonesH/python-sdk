from enum import Enum
from typing import Optional, List, Dict, Any, Union, Literal, Callable
from pydantic import BaseModel, Field
from datetime import datetime

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
    isBuiltByAgentBuilder: bool = False
    systemPrompt: Optional[str] = None
    capabilities_description: Optional[str] = None

class Agent(BaseModel):
    id: int
    name: str
    kind: str = "openserv"
    capabilities_description: Optional[str] = None

class TaskAttachment(BaseModel):
    id: int
    path: str
    fullUrl: str
    summary: Optional[str] = None

class TaskDependency(BaseModel):
    id: int
    description: str
    output: Optional[str] = None
    status: TaskStatus
    attachments: List[TaskAttachment] = []

class HumanAssistanceRequest(BaseModel):
    id: int
    agentDump: Optional[Any] = None
    humanResponse: Optional[str] = None
    question: str
    status: Literal['pending', 'responded']
    type: Literal['text', 'project-manager-plan-review']

class Task(BaseModel):
    id: int
    description: str
    body: Optional[str] = None
    expectedOutput: Optional[str] = None
    input: Optional[str] = None
    dependencies: List[TaskDependency] = []
    humanAssistanceRequests: List[HumanAssistanceRequest] = []

class Workspace(BaseModel):
    id: int
    goal: str
    bucket_folder: str
    agents: List[Agent]

class Integration(BaseModel):
    id: int
    connection_id: str
    provider_config_key: str
    provider: str
    created: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    scopes: Optional[List[str]] = None
    openAPI: Dict[str, str]

class Memory(BaseModel):
    id: int
    memory: str
    createdAt: datetime

class AgentAction(BaseModel):
    type: Literal['do-task', 'respond-chat-message']
    me: AgentBase
    task: Optional[Task] = None
    workspace: Workspace
    integrations: List[Integration] = []
    memories: List[Memory] = []

class DoTaskAction(AgentAction):
    type: Literal['do-task']
    task: Task

class ChatMessage(BaseModel):
    author: Literal['user', 'agent']
    message: str
    id: int
    createdAt: datetime

class RespondChatMessageAction(BaseModel):
    type: str = "respond-chat-message"
    me: Agent
    messages: List[ChatMessage]
    workspace: Workspace
    integrations: List[Dict[str, Any]] = []
    memories: List[Memory] = []

class ProcessParams(BaseModel):
    messages: List[Dict[str, str]]

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

class GetFilesParams(BaseModel):
    workspace_id: int

class UploadFileParams(BaseModel):
    workspace_id: int
    path: str
    task_ids: Optional[Union[List[int], int, None]] = None
    skip_summarizer: Optional[bool] = None
    file: Union[bytes, str]

class MarkTaskAsErroredParams(BaseModel):
    workspace_id: int
    task_id: int
    error: str

class CompleteTaskParams(BaseModel):
    workspace_id: int
    task_id: int
    output: str

class SendChatMessageParams(BaseModel):
    workspace_id: int
    agent_id: int
    message: str

class GetTaskDetailParams(BaseModel):
    workspace_id: int
    task_id: int

class GetAgentsParams(BaseModel):
    workspace_id: int

class GetTasksParams(BaseModel):
    workspace_id: int

class CreateTaskParams(BaseModel):
    workspace_id: int
    assignee: int
    description: str
    body: str
    input: str
    expected_output: str
    dependencies: List[int]

class AddLogToTaskParams(BaseModel):
    workspace_id: int
    task_id: int
    severity: Literal['info', 'warning', 'error']
    type: Literal['text', 'openai-message']
    body: Union[str, dict]

class RequestHumanAssistanceParams(BaseModel):
    workspace_id: int
    task_id: int
    type: Literal['text', 'project-manager-plan-review']
    question: Union[str, dict]
    agent_dump: Optional[dict] = None

class UpdateTaskStatusParams(BaseModel):
    workspace_id: int
    task_id: int
    status: TaskStatus

class ProxyConfiguration(BaseModel):
    endpoint: str
    provider_config_key: Optional[str] = None
    connection_id: Optional[str] = None
    method: Optional[Literal['GET', 'POST', 'PATCH', 'PUT', 'DELETE', 'get', 'post', 'patch', 'put', 'delete']] = None
    headers: Optional[Dict[str, str]] = None
    params: Optional[Union[str, Dict[str, Union[str, int]]]] = None
    data: Optional[Any] = None
    retries: Optional[int] = None
    base_url_override: Optional[str] = None
    decompress: Optional[bool] = None
    response_type: Optional[Literal['arraybuffer', 'blob', 'document', 'json', 'text', 'stream']] = None
    retry_on: Optional[List[int]] = None

class IntegrationCallRequest(BaseModel):
    workspace_id: int
    integration_id: str
    details: ProxyConfiguration 

class EngagementMetrics(BaseModel):
    """Social media engagement metrics."""
    likes: int = Field(ge=0)
    shares: int = Field(ge=0)
    comments: int = Field(ge=0)
    impressions: int = Field(ge=0)

class SocialMediaPostParams(BaseModel):
    """Parameters for creating a social media post."""
    platform: str
    topic: str

class AnalyzeEngagementParams(BaseModel):
    """Parameters for analyzing engagement metrics."""
    platform: str
    metrics: EngagementMetrics 
