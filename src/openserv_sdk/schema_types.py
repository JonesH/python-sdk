from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal, TypeVar, ForwardRef, Annotated, Generic

from pydantic import BaseModel, Field, model_validator, RootModel


# Forward declarations
DoTaskActionSchema = ForwardRef('DoTaskActionSchema')
RespondChatMessageActionSchema = ForwardRef('RespondChatMessageActionSchema')
FunctionDefinition = ForwardRef('FunctionDefinition')
FunctionParameters = ForwardRef('FunctionParameters')
Task = ForwardRef('Task')
Workspace = ForwardRef('Workspace')
Integration = ForwardRef('Integration')
Memory = ForwardRef('Memory')
TextQuestion = ForwardRef('TextQuestion')
ProjectManagerPlanReviewQuestion = ForwardRef('ProjectManagerPlanReviewQuestion')
ProjectManagerPlanTask = ForwardRef('ProjectManagerPlanTask')
InsufficientBalanceQuestion = ForwardRef('InsufficientBalanceQuestion')
OpenAPIInfo = ForwardRef('OpenAPIInfo')
ProxyConfiguration = ForwardRef('ProxyConfiguration')
TaskDependency = ForwardRef('TaskDependency')
TaskStatus = ForwardRef('TaskStatus')
HumanAssistanceRequest = ForwardRef('HumanAssistanceRequest')
Attachment = ForwardRef('Attachment')


class ActionSchema(BaseModel):
    """Base schema for actions."""
    type: Literal["do-task", "respond-chat-message"]


class Action(RootModel):
    """Union type for actions."""
    root: Union[DoTaskActionSchema, RespondChatMessageActionSchema]


class AddLogToTaskParams(BaseModel):
    """Parameters for adding a log entry to a task."""
    workspace_id: int
    task_id: int
    severity: Literal["info", "warning", "error"]
    type: Literal["text", "openai-message"]
    body: Union[str, Dict[str, Any]]


class AgentInfo(BaseModel):
    """Represents basic information about an agent."""
    id: int
    name: str
    capabilities_description: str = Field(..., alias="capabilities_description")


class AgentKind(str, Enum):
    """Types of agents in the system."""
    EXTERNAL = "external"
    ELIZA = "eliza"
    OPENSERV = "openserv"


class AgentWithSystemPrompt(BaseModel):
    """Represents an agent with its system prompt information."""
    id: int
    name: str
    kind: AgentKind
    is_built_by_agent_builder: bool = Field(..., alias="isBuiltByAgentBuilder")
    system_prompt: Optional[str] = Field(None, alias="systemPrompt")
    
    @model_validator(mode='after')
    def validate_system_prompt(self):
        if self.is_built_by_agent_builder and not self.system_prompt:
            raise ValueError("systemPrompt is required when isBuiltByAgentBuilder is true")
        return self


class Attachment(BaseModel):
    """Represents a file attachment."""
    id: int
    path: str
    full_url: str = Field(..., alias="fullUrl")
    summary: Optional[str] = None


class ChatCompletionMessageParam(BaseModel):
    """A message in a chat completion request."""
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ChatCompletionTool(BaseModel):
    """Represents a tool for chat completion."""
    type: Literal["function"]
    function: FunctionDefinition


class ChatCompletion(BaseModel):
    """Represents a chat completion response."""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class ChatMessageInfo(BaseModel):
    """Represents a chat message."""
    author: Literal["agent", "user"]
    created_at: datetime = Field(..., alias="createdAt")
    id: int
    message: str


class CompleteTaskParams(BaseModel):
    """Parameters for completing a task."""
    workspace_id: int
    task_id: int
    output: str


class CreateTaskParams(BaseModel):
    """Parameters for creating a new task."""
    workspace_id: int
    assignee: int
    description: str
    body: str
    input: str
    expected_output: str
    dependencies: List[int]


class DoTaskActionSchema(BaseModel):
    """Schema for a do-task action."""
    type: Literal["do-task"]
    me: AgentWithSystemPrompt
    task: Task
    workspace: Workspace
    integrations: List[Integration]
    memories: List[Memory]


class FunctionDefinition(BaseModel):
    """Represents a function definition."""
    name: str
    description: Optional[str] = None
    parameters: FunctionParameters


class FunctionParameters(BaseModel):
    """Represents the parameters for a function."""
    type: str = "object"
    properties: Dict[str, Any] = {}
    required: Optional[List[str]] = None


class GetAgentsParams(BaseModel):
    """Parameters for retrieving agents in a workspace."""
    workspace_id: int


class GetFilesParams(BaseModel):
    """Parameters for retrieving files from a workspace."""
    workspace_id: int


class GetSecretsParams(BaseModel):
    """Parameters for retrieving secrets from a workspace."""
    workspace_id: int


class GetSecretValueParams(BaseModel):
    """Parameters for retrieving a secret value."""
    workspace_id: int
    secret_id: int


class GetTaskDetailParams(BaseModel):
    """Parameters for retrieving task details."""
    workspace_id: int
    task_id: int


class GetTasksParams(BaseModel):
    """Parameters for retrieving tasks in a workspace."""
    workspace_id: int


class HumanAssistanceRequest(BaseModel):
    """Base model for human assistance requests."""
    type: Literal["text", "project-manager-plan-review", "insufficient-balance", "json"]
    question: Union[TextQuestion, ProjectManagerPlanReviewQuestion, InsufficientBalanceQuestion, Dict[str, Any]]
    agent_dump: Optional[Any] = Field(None, alias="agentDump")
    human_response: Optional[str] = Field(None, alias="humanResponse")
    id: Optional[int] = None
    status: Optional[Literal["pending", "responded"]] = None

    @model_validator(mode='after')
    def validate_question_type(self):
        if self.type == "text" and not isinstance(self.question, TextQuestion):
            raise ValueError("Question must be TextQuestion for type 'text'")
        if self.type == "project-manager-plan-review" and not isinstance(self.question, ProjectManagerPlanReviewQuestion):
            raise ValueError("Question must be ProjectManagerPlanReviewQuestion for type 'project-manager-plan-review'")
        if self.type == "insufficient-balance" and not isinstance(self.question, InsufficientBalanceQuestion):
            raise ValueError("Question must be InsufficientBalanceQuestion for type 'insufficient-balance'")
        if self.type == "json" and not isinstance(self.question, dict):
            raise ValueError("Question must be a dictionary for type 'json'")
        return self


class InsufficientBalanceQuestion(BaseModel):
    """An insufficient balance question."""
    type: Literal["insufficient-balance"]
    question: Dict[str, Literal["insufficient-balance"]] = Field(..., alias="question")


class Integration(BaseModel):
    """Represents an integration."""
    id: int
    connection_id: str = Field(..., alias="connection_id")
    provider_config_key: str = Field(..., alias="provider_config_key")
    provider: str
    created: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    scopes: Optional[List[str]] = None
    open_api: OpenAPIInfo = Field(..., alias="openAPI")


class IntegrationCallRequest(BaseModel):
    """Parameters for calling an integration."""
    workspace_id: int
    integration_id: str
    details: ProxyConfiguration


class MarkTaskAsErroredParams(BaseModel):
    """Parameters for marking a task as errored."""
    workspace_id: int
    task_id: int
    error: str


class Memory(BaseModel):
    """Represents an agent memory."""
    id: int
    memory: str
    created_at: datetime = Field(..., alias="createdAt")


class OpenAPIInfo(BaseModel):
    """Represents OpenAPI information for an integration."""
    title: str
    description: str


class ProcessParams(BaseModel):
    """Parameters for processing a conversation with OpenAI."""
    messages: List[ChatCompletionMessageParam]


class ProjectManagerPlanReviewQuestion(BaseModel):
    """Question for a project manager plan review."""
    type: Literal["project-manager-plan-review"]
    tasks: List[ProjectManagerPlanTask]


class ProjectManagerPlanTask(BaseModel):
    """Represents a task in a project manager's plan."""
    index: int
    assignee_agent_id: int = Field(..., alias="assigneeAgentId")
    assignee_agent_name: str = Field(..., alias="assigneeAgentName")
    task_description: str = Field(..., alias="taskDescription")
    task_body: str = Field(..., alias="taskBody")
    input: str
    expected_output: str = Field(..., alias="expectedOutput")


class ProxyConfiguration(BaseModel):
    """Configuration for API proxy requests."""
    endpoint: str
    provider_config_key: Optional[str] = None
    connection_id: Optional[str] = None
    method: Optional[Literal["GET", "POST", "PATCH", "PUT", "DELETE", "get", "post", "patch", "put", "delete"]] = None
    headers: Optional[Dict[str, str]] = None
    params: Optional[Union[str, Dict[str, Union[str, int]]]] = None
    data: Optional[Any] = None
    retries: Optional[int] = None
    base_url_override: Optional[str] = Field(None, alias="baseUrlOverride")
    decompress: Optional[bool] = None
    response_type: Optional[Literal["arraybuffer", "blob", "document", "json", "text", "stream"]] = None
    retry_on: Optional[List[int]] = Field(None, alias="retryOn")


class RequestHumanAssistanceParams(BaseModel):
    """Parameters for requesting human assistance."""
    workspace_id: int
    task_id: int
    type: Literal["text", "project-manager-plan-review"]
    question: Union[str, Dict[str, Any]]
    agent_dump: Optional[Dict[str, Any]] = None


class RespondChatMessageActionSchema(BaseModel):
    """Schema for a respond-chat-message action."""
    type: Literal["respond-chat-message"]
    me: AgentWithSystemPrompt
    messages: List[ChatMessageInfo]
    workspace: Workspace
    integrations: List[Integration]
    memories: List[Memory]


class SendChatMessageParams(BaseModel):
    """Parameters for sending a chat message."""
    workspace_id: int
    agent_id: int
    message: str


class Task(BaseModel):
    """Represents a task."""
    id: int
    description: str
    body: Optional[str] = None
    expected_output: Optional[str] = Field(None, alias="expectedOutput")
    input: Optional[str] = None
    dependencies: List[TaskDependency]
    human_assistance_requests: List[HumanAssistanceRequest] = Field(
        [], alias="humanAssistanceRequests"
    )
    status: Optional[TaskStatus] = None
    attachments: Optional[List[Attachment]] = None


class TaskDependency(BaseModel):
    """Represents a task dependency."""
    id: int
    description: str
    output: Optional[str] = None
    status: TaskStatus
    attachments: List[Attachment]


class CapabilityFuncParams(BaseModel, Generic[TypeVar('Schema')]):
    """Helper type for capability function parameters."""
    args: Any  # Using Any since we can't directly translate z.infer<Schema>
    action: Optional[Action] = None


class TaskStatus(str, Enum):
    """Possible statuses for a task."""
    TO_DO = "to-do"
    IN_PROGRESS = "in-progress"
    HUMAN_ASSISTANCE_REQUIRED = "human-assistance-required"
    ERROR = "error"
    DONE = "done"
    CANCELLED = "cancelled"


class TextQuestion(BaseModel):
    """A text-based question."""
    type: Literal["text"]
    question: str = Field(..., min_length=1, strip_whitespace=True)


class UpdateTaskStatusParams(BaseModel):
    """Parameters for updating a task's status."""
    workspace_id: int
    task_id: int
    status: TaskStatus


class UploadFileParams(BaseModel):
    """Parameters for uploading a file to a workspace."""
    workspace_id: int
    path: str
    file: Union[bytes, str]
    task_ids: Optional[Union[int, List[int]]] = None
    skip_summarizer: Optional[bool] = None


class Workspace(BaseModel):
    """Represents a workspace."""
    id: int
    goal: str
    bucket_folder: str = Field(..., alias="bucket_folder")
    agents: List[AgentInfo]