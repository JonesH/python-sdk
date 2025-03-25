import os
import json
from typing import Any, Callable, Dict, List, Optional, Type, Union, TypeVar, cast
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
import uvicorn
from openai import AsyncOpenAI

from .capability import Capability
from .logger import logger
from .schema_types import (
    ActionSchema, DoTaskActionSchema, RespondChatMessageActionSchema, 
    GetFilesParams, GetSecretsParams, GetSecretValueParams, UploadFileParams, 
    MarkTaskAsErroredParams, CompleteTaskParams, SendChatMessageParams, 
    GetTaskDetailParams, GetAgentsParams, GetTasksParams, CreateTaskParams, 
    AddLogToTaskParams, RequestHumanAssistanceParams, UpdateTaskStatusParams, 
    ProcessParams, IntegrationCallRequest, TaskStatus, ChatCompletionMessageParam,
    ChatCompletionTool, ChatCompletion
)

# Default configuration
PLATFORM_URL = os.environ.get('OPENSERV_API_URL', 'https://api.openserv.ai')
RUNTIME_URL = os.environ.get('OPENSERV_RUNTIME_URL', 'https://agents.openserv.ai')
DEFAULT_PORT = int(os.environ.get('PORT', '7378'))


@dataclass
class AgentOptions:
    """
    Configuration options for creating a new Agent instance.
    
    Attributes:
        system_prompt: The system prompt that defines the agent's behavior and context.
            Used as the initial system message in OpenAI chat completions.
        
        api_key: The OpenServ API key for authentication. 
            Can also be provided via OPENSERV_API_KEY environment variable.
            Required for all API operations.
        
        port: The port number for the agent's HTTP server. 
            Defaults to 7378 if not specified.
        
        openai_api_key: The OpenAI API key for chat completions. 
            Can also be provided via OPENAI_API_KEY environment variable.
            Required when using the process() method.
        
        on_error: Error handler function for all agent operations. 
            Defaults to logging the error if not provided.
            Takes an error and optional context as parameters.
    """
    
    system_prompt: str
    """The system prompt that defines the agent's behavior and context."""
    
    api_key: Optional[str] = None
    """The OpenServ API key for authentication. Can also be provided via OPENSERV_API_KEY environment variable."""
    
    port: Optional[int] = None
    """The port number for the agent's HTTP server. Defaults to 7378 if not specified."""
    
    openai_api_key: Optional[str] = None
    """The OpenAI API key for chat completions. Can also be provided via OPENAI_API_KEY environment variable."""
    
    on_error: Optional[Callable[[Exception, Optional[Dict[str, Any]]], None]] = None
    """Error handler function for all agent operations. Defaults to logging the error if not provided."""


T = TypeVar('T', bound=BaseModel)


class Agent:
    """
    Main Agent class for the OpenServ Python SDK.
    
    This class provides the core functionality for creating AI agents that can
    execute tasks, chat with users, and integrate with the OpenServ platform.
    """
    
    def __init__(self, options: AgentOptions):
        """
        Creates a new Agent instance.
        Sets up the FastAPI application, middleware, and routes.
        Initializes API clients with appropriate authentication.
        
        Args:
            options: Configuration options for the agent
            
        Raises:
            ValueError: If OpenServ API key is not provided in options or environment
        """
        self.app = FastAPI(title="OpenServ Agent")
        self.port = options.port or DEFAULT_PORT
        self.system_prompt = options.system_prompt
        self.api_key = options.api_key or os.environ.get('OPENSERV_API_KEY', '')
        self.options = options
        self.tools: List[Capability] = []
        self._openai: Optional[AsyncOpenAI] = None
        self.server = None

        if not self.api_key:
            raise ValueError(
                'OpenServ API key is required. Please provide it in options or set OPENSERV_API_KEY environment variable.'
            )

        # Add middleware
        self.app.add_middleware(GZipMiddleware)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Initialize API clients
        self.api_client = httpx.AsyncClient(
            base_url=PLATFORM_URL,
            headers={
                'Content-Type': 'application/json',
                'x-openserv-key': self.api_key
            },
            timeout=60.0
        )

        # Initialize runtime client
        self.runtime_client = httpx.AsyncClient(
            base_url=RUNTIME_URL,
            headers={
                'Content-Type': 'application/json',
                'x-openserv-key': self.api_key
            },
            timeout=60.0
        )

        # Set up routes
        self._setup_routes()
    
    @property
    def openai_tools(self) -> List[ChatCompletionTool]:
        """
        Getter that converts the agent's tools into OpenAI function calling format.
        Used when making chat completion requests to OpenAI.
        
        Returns:
            List of ChatCompletionTool objects
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema.model_json_schema()
                }
            }
            for tool in self.tools
        ]
    
    @property
    def openai(self) -> AsyncOpenAI:
        """
        Getter that provides access to the OpenAI client instance.
        Lazily initializes the client with the API key from options or environment.
        
        Raises:
            ValueError: If no OpenAI API key is available
            
        Returns:
            The OpenAI client instance
        """
        if not self._openai:
            api_key = self.options.openai_api_key or os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError(
                    'OpenAI API key is required for process(). Please provide it in options or set OPENAI_API_KEY environment variable.'
                )
            self._openai = AsyncOpenAI(api_key=api_key)
        return self._openai
    
    async def add_log_to_task(self, params: AddLogToTaskParams) -> Dict[str, Any]:
        """
        Adds a log entry to a task.
        
        Args:
            params: Parameters for adding the log
            params.workspace_id: ID of the workspace containing the task
            params.task_id: ID of the task to add the log to
            params.severity: Severity level of the log ('info', 'warning', 'error')
            params.type: Type of log entry ('text', 'openai-message')
            params.body: Content of the log entry (string or object)
            
        Returns:
            The created log entry details
        """
        response = await self.api_client.post(
            f"/workspaces/{params.workspace_id}/tasks/{params.task_id}/log",
            json={
                "severity": params.severity,
                "type": params.type,
                "body": params.body
            }
        )
        response.raise_for_status()
        try:
            return response.json()
        except Exception as e:
            logger.error(f"Error parsing JSON response in add_log_to_task: {str(e)}")
            return {"success": True, "message": "Log added to task (response could not be parsed)"}
    
    def add_capability(self, *, name: str, description: str, schema: Type[BaseModel], run: Callable) -> 'Agent':
        """
        Adds a single capability (tool) to the agent.
        Each capability must have a unique name and defines a function that can be called via the API.
        
        Example:
            ```python
            from pydantic import BaseModel
            
            class SearchParams(BaseModel):
                query: str
                limit: int = 10
            
            agent.add_capability(
                name="search",
                description="Search for information",
                schema=SearchParams,
                run=lambda params, messages: f"Searching for: {params.args.query}"
            )
            ```
        
        Args:
            name: Unique name for the capability
                Used to identify the tool in API calls and OpenAI function calls
            description: Description of what the capability does
                Used by OpenAI to understand when to use this tool
            schema: Pydantic model defining the capability's parameters
                Must be a subclass of BaseModel
                Defines the expected input structure
            run: Function that implements the capability's behavior
                Takes params (dict with 'args' and optional 'action') and messages list
                Returns a string or dict that will be JSON serialized
        
        Returns:
            The agent instance for method chaining
        
        Raises:
            ValueError: If a capability with the same name already exists
        """
        # Validate tool name uniqueness
        if any(tool.name == name for tool in self.tools):
            raise ValueError(f'Tool with name "{name}" already exists')
        
        # Create and add capability
        self.tools.append(Capability(name=name, description=description, schema=schema, run=run))
        return self
    
    def add_capabilities(self, capabilities: List[Dict]) -> 'Agent':
        """
        Adds multiple capabilities (tools) to the agent at once.
        Each capability must have a unique name and not conflict with existing capabilities.
        
        Args:
            capabilities: List of capability configurations with name, description, schema, and run function
            
        Returns:
            The agent instance for method chaining
        """
        for capability in capabilities:
            self.add_capability(
                name=capability['name'],
                description=capability['description'],
                schema=capability['schema'],
                run=capability['run']
            )
        return self
    
    async def call_integration(self, integration: IntegrationCallRequest) -> Dict[str, Any]:
        """
        Calls an integration endpoint through the OpenServ platform.
        This method allows agents to interact with external services and APIs that are integrated with OpenServ.
        
        Args:
            integration: The integration request parameters
            integration.workspace_id: ID of the workspace where the integration is configured
            integration.integration_id: ID of the integration to call
            integration.details: Details of the integration call
            integration.details.endpoint: The endpoint to call on the integration
            integration.details.method: The HTTP method to use (GET, POST, etc.)
            integration.details.data: Optional data payload for the request
            
        Returns:
            The response from the integration endpoint
        """
        response = await self.api_client.post(
            f"/workspaces/{integration.workspace_id}/integration/{integration.integration_id}/proxy",
            json=integration.details.model_dump()
        )
        response.raise_for_status()
        try:
            return response.json()
        except Exception as e:
            logger.error(f"Error parsing JSON response in call_integration: {str(e)}")
            return {"success": True, "message": "Integration called (response could not be parsed)"}
    
    async def complete_task(self, params: CompleteTaskParams) -> Dict[str, Any]:
        """
        Completes a task with the specified output.
        
        Args:
            params: Parameters for completing the task
            params.workspace_id: ID of the workspace containing the task
            params.task_id: ID of the task to complete
            params.output: The output content to set for the task
            
        Returns:
            The completed task details
        """
        logger.info(f"Completing task {params.task_id} in workspace {params.workspace_id}")
        
        response = await self.api_client.put(
            f"/workspaces/{params.workspace_id}/tasks/{params.task_id}/complete",
            json={"output": params.output}
        )
        response.raise_for_status()
        
        # Log the response status for debugging
        logger.info(f"Task completion API response status: {response.status_code}")
        
        # Check if response has content before trying to parse it
        if response.content and len(response.content.strip()) > 0:
            try:
                return response.json()
            except Exception as e:
                logger.error(f"Error parsing JSON response in complete_task: {str(e)}")
                return {"success": True, "message": "Task completed successfully"}
        else:
            logger.info(f"Task {params.task_id} completed successfully (empty response)")
            return {"success": True, "message": "Task completed successfully"}
    
    async def create_task(self, params: CreateTaskParams) -> Dict[str, Any]:
        """
        Creates a new task in a workspace.
        
        Args:
            params: Parameters for creating the task
            params.workspace_id: ID of the workspace to create the task in
            params.assignee: ID of the agent to assign the task to
            params.description: Short description of the task
            params.body: Detailed body content of the task
            params.input: Input data for the task
            params.expected_output: Expected output format or content
            params.dependencies: List of task IDs that this task depends on
            
        Returns:
            The created task details
        """
        response = await self.api_client.post(f"/workspaces/{params.workspace_id}/task", json={
            "assignee": params.assignee,
            "description": params.description,
            "body": params.body,
            "input": params.input,
            "expectedOutput": params.expected_output,
            "dependencies": params.dependencies
        })
        response.raise_for_status()
        try:
            return response.json()
        except Exception as e:
            logger.error(f"Error parsing JSON response in create_task: {str(e)}")
            return {"success": True, "message": "Task created (response could not be parsed)"}
    
    async def do_task(self, action: DoTaskActionSchema):
        """
        Handle a task execution request.
        This method can be overridden by extending classes to customize task handling.
        
        Args:
            action: The task action to handle
            action.task: The task details including description, body, etc.
            action.workspace: The workspace context where the task is running
            action.me: Information about the current agent
        """
        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

        if action.task and action.task.description:
            messages.append({
                "role": "user",
                "content": action.task.description
            })

        try:
            await self.runtime_client.post("/runtime/execute", json={
                "tools": [self._convert_tool_to_json_schema(tool) for tool in self.tools],
                "messages": messages,
                "action": action.model_dump()
            })
        except Exception as error:
            self._handle_error(error, {
                "action": action.model_dump(),
                "context": "do_task"
            })
    
    async def get_agents(self, params: GetAgentsParams) -> Dict[str, Any]:
        """
        Gets a list of agents in a workspace.
        
        Args:
            params: Parameters for getting agents
            params.workspace_id: ID of the workspace to get agents from
            
        Returns:
            Dictionary containing a list of agents in the workspace and metadata
        """
        response = await self.api_client.get(f"/workspaces/{params.workspace_id}/agents")
        response.raise_for_status()
        try:
            return response.json()
        except Exception as e:
            logger.error(f"Error parsing JSON response in get_agents: {str(e)}")
            return {"agents": [], "message": "Could not parse agents response"}
    
    async def get_files(self, params: GetFilesParams) -> Dict[str, Any]:
        """
        Gets files in a workspace.
        
        Args:
            params: Parameters for the file retrieval
            params.workspace_id: ID of the workspace to get files from
            
        Returns:
            Dictionary containing a list of files in the workspace and metadata
        """
        response = await self.api_client.get(f"/workspaces/{params.workspace_id}/files")
        response.raise_for_status()
        try:
            return response.json()
        except Exception as e:
            logger.error(f"Error parsing JSON response in get_files: {str(e)}")
            return {"files": [], "message": "Could not parse files response"}
    
    async def get_secrets(self, params: GetSecretsParams) -> Dict[str, Any]:
        """
        Get all secrets for an agent in a workspace.
        
        Args:
            params: Parameters for the secrets retrieval
            params.workspace_id: ID of the workspace to get secrets from
            
        Returns:
            Dictionary containing a list of agent secrets and metadata
        """
        response = await self.api_client.get(f"/workspaces/{params.workspace_id}/agent-secrets")
        response.raise_for_status()
        try:
            return response.json()
        except Exception as e:
            logger.error(f"Error parsing JSON response in get_secrets: {str(e)}")
            return {"secrets": [], "message": "Could not parse secrets response"}
    
    async def get_secret_value(self, params: GetSecretValueParams) -> str:
        """
        Get the value of a secret for an agent in a workspace.
        
        Args:
            params: Parameters for the secret value retrieval
            params.workspace_id: ID of the workspace containing the secret
            params.secret_id: ID of the secret to retrieve the value for
            
        Returns:
            The value of the secret as a string
        """
        response = await self.api_client.get(
            f"/workspaces/{params.workspace_id}/agent-secrets/{params.secret_id}/value"
        )
        response.raise_for_status()
        try:
            return response.json()
        except Exception as e:
            logger.error(f"Error parsing JSON response in get_secret_value: {str(e)}")
            return ""
    
    async def get_task_detail(self, params: GetTaskDetailParams) -> Dict[str, Any]:
        """
        Retrieves detailed information about a specific task.
        
        Args:
            params: Parameters for retrieving task details
            params.workspace_id: ID of the workspace containing the task
            params.task_id: ID of the task to retrieve details for
            
        Returns:
            Detailed information about the requested task
        """
        response = await self.api_client.get(f"/workspaces/{params.workspace_id}/tasks/{params.task_id}/detail")
        response.raise_for_status()
        try:
            return response.json()
        except Exception as e:
            logger.error(f"Error parsing JSON response in get_task_detail: {str(e)}")
            return {"success": False, "message": "Failed to parse task details response"}
    
    async def get_tasks(self, params: GetTasksParams) -> Dict[str, Any]:
        """
        Gets a list of tasks in a workspace.
        
        Args:
            params: Parameters for getting tasks
            params.workspace_id: ID of the workspace to retrieve tasks from
            
        Returns:
            Dictionary containing a list of tasks in the workspace and metadata
        """
        response = await self.api_client.get(f"/workspaces/{params.workspace_id}/tasks")
        response.raise_for_status()
        try:
            return response.json()
        except Exception as e:
            logger.error(f"Error parsing JSON response in get_tasks: {str(e)}")
            return {"tasks": [], "message": "Could not parse tasks response"}
    
    async def _handle_root_route(self, request: Request) -> Dict[str, Any]:
        """
        Handles the root route for task execution and chat message responses.
        
        Args:
            request: The HTTP request
            
        Returns:
            Success response
            
        Raises:
            HTTPException: If action type is invalid
        """
        try:
            body = await request.json()
            action = ActionSchema.model_validate(body)
            
            if action.type == "do-task":
                do_task_action = DoTaskActionSchema.model_validate(body)
                await self.do_task(do_task_action)
            elif action.type == "respond-chat-message":
                respond_chat_action = RespondChatMessageActionSchema.model_validate(body)
                await self.respond_to_chat(respond_chat_action)
            else:
                raise HTTPException(status_code=400, detail=f"Invalid action type: {action.type}")
            
            return {"success": True}
        except Exception as error:
            self._handle_error(error, {
                "request": str(request),
                "context": "handle_root_route"
            })
            raise
    
    async def _handle_tool_route(self, request: Request) -> Dict[str, Any]:
        """
        Handles execution of a specific tool/capability.
        
        Args:
            request: The HTTP request
            
        Returns:
            The result of the tool execution
            
        Raises:
            HTTPException: If tool name is missing or tool is not found
        """
        try:
            tool_name = request.path_params.get("tool_name")
            if not tool_name:
                raise HTTPException(status_code=400, detail="Tool name is required")

            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                raise HTTPException(status_code=404, detail=f'Tool "{tool_name}" not found')

            body = await request.json()
            args = body.get("args", {})
            action = body.get("action")
            messages = body.get("messages", [])

            # Validate arguments against the tool's schema
            validated_args = tool.schema.model_validate(args)
            
            # Call the tool with validated arguments
            result = await tool.run({"args": validated_args, "action": action}, messages)
            
            return {"result": result}
        except Exception as error:
            self._handle_error(error, {
                "request": str(request),
                "context": "handle_tool_route"
            })
            raise
    
    def _handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """
        Default error handler that logs the error or calls the custom handler.
        This method is used internally to handle errors consistently across the agent.
        
        The handler:
        1. Uses the custom error handler if provided in options
        2. Falls back to logging the error with context if no custom handler
        3. Preserves the original error for re-raising
        
        Args:
            error: The error that occurred
                Can be any Exception type
            context: Additional context about where the error occurred
                Optional dictionary with debugging information
                Common keys: 'action', 'tool_call', 'context'
        
        Example:
            ```python
            try:
                # Some operation that might fail
                result = await self.process(params)
            except Exception as e:
                self._handle_error(e, {
                    "context": "process",
                    "params": params
                })
                raise
            ```
        """
        handler = self.options.on_error or (lambda err, ctx: logger.error(
            f"Error in agent operation: {str(err)}",
            extra={"error": str(err), **(ctx or {})}
        ))
        handler(error, context)
    
    async def mark_task_as_errored(self, params: MarkTaskAsErroredParams) -> Dict[str, Any]:
        """
        Marks a task as errored.
        
        Args:
            params: Parameters for marking the task as errored
            params.workspace_id: ID of the workspace containing the task
            params.task_id: ID of the task to mark as errored
            params.error: Error message or details to associate with the task
            
        Returns:
            The updated task details
        """
        logger.info(f"Marking task {params.task_id} as errored in workspace {params.workspace_id}")
        
        response = await self.api_client.post(
            f"/workspaces/{params.workspace_id}/tasks/{params.task_id}/error",
            json={"error": params.error}
        )
        response.raise_for_status()
        
        # Log the response status for debugging
        logger.info(f"Task error API response status: {response.status_code}")
        
        # Check if response has content before trying to parse it
        if response.content and len(response.content.strip()) > 0:
            try:
                return response.json()
            except Exception as e:
                logger.error(f"Error parsing JSON response in mark_task_as_errored: {str(e)}")
                return {"success": True, "message": "Task marked as errored successfully"}
        else:
            logger.info(f"Task {params.task_id} marked as errored successfully (empty response)")
            return {"success": True, "message": "Task marked as errored successfully"}
    
    async def process(self, *, messages: List[ChatCompletionMessageParam]) -> ChatCompletion:
        """
        Processes a conversation with OpenAI, handling tool calls iteratively until completion.
        This method is the primary way to test agents locally and process conversations with OpenAI.
        
        The method:
        1. Validates the OpenAI API key
        2. Processes messages iteratively (up to 10 iterations)
        3. Handles tool calls from OpenAI
        4. Manages conversation state
        5. Returns the final completion
        
        Example:
            ```python
            agent = Agent(options)
            result = await agent.process(messages=[
                {"role": "user", "content": "Hello, how can you help me?"}
            ])
            ```
        
        Args:
            messages: The conversation history as a list of message objects
                Each message should have 'role' and 'content' fields
                Supported roles: 'system', 'user', 'assistant', 'tool'
        
        Returns:
            The final response from OpenAI as a ChatCompletion object
            Contains the model's response and any tool calls made
        
        Raises:
            ValueError: If no response is received from OpenAI or max iterations are reached
            ValueError: If OpenAI API key is not provided
            ValueError: If tool call arguments are invalid
        """
        try:
            api_key = self.options.openai_api_key or os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError(
                    'OpenAI API key is required for process(). Please provide it in options or set OPENAI_API_KEY environment variable.'
                )

            current_messages = list(messages)
            completion = None
            iteration_count = 0
            MAX_ITERATIONS = 10

            while iteration_count < MAX_ITERATIONS:
                completion = await self.openai.chat.completions.create(
                    model="gpt-4o",
                    messages=current_messages,
                    tools=self.openai_tools if self.tools else None
                )

                if not completion.choices or not completion.choices[0].message:
                    raise ValueError('No response from OpenAI')

                last_message = completion.choices[0].message

                # If there are no tool calls, we're done
                if not last_message.tool_calls:
                    return completion

                # Process each tool call
                tool_results = []
                for tool_call in last_message.tool_calls:
                    if not tool_call.function:
                        raise ValueError('Tool call function is missing')
                    
                    name = tool_call.function.name
                    args_str = tool_call.function.arguments
                    try:
                        parsed_args = json.loads(args_str)
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid JSON in tool call arguments: {args_str}")

                    try:
                        # Find the tool in our tools array
                        tool = next((t for t in self.tools if t.name == name), None)
                        if not tool:
                            raise ValueError(f'Tool "{name}" not found')

                        # Call the tool's run method with the parsed arguments
                        validated_args = tool.schema.model_validate(parsed_args)
                        result = await tool.run({"args": validated_args}, current_messages)
                        
                        tool_results.append({
                            "role": "tool",
                            "content": json.dumps(result) if not isinstance(result, str) else result,
                            "tool_call_id": tool_call.id
                        })
                    except Exception as error:
                        error_message = str(error)
                        self._handle_error(error, {
                            "tool_call": tool_call.model_dump(),
                            "context": "tool_execution"
                        })
                        tool_results.append({
                            "role": "tool",
                            "content": json.dumps({"error": error_message}),
                            "tool_call_id": tool_call.id
                        })

                # Add the assistant's message and tool results to the conversation
                current_messages.append(last_message.model_dump())
                current_messages.extend(tool_results)
                iteration_count += 1

            raise ValueError('Max iterations reached without completion')
        except Exception as error:
            self._handle_error(error, {"context": "process"})
            raise
    
    async def request_human_assistance(self, params: RequestHumanAssistanceParams) -> Dict[str, Any]:
        """
        Requests human assistance for a task.
        
        Args:
            params: Parameters for requesting assistance
            params.workspace_id: ID of the workspace containing the task
            params.task_id: ID of the task needing assistance
            params.type: Type of assistance needed ('text', 'project-manager-plan-review')
            params.question: Question or request for the human (string or object)
            params.agent_dump: Optional agent state/context information
            
        Returns:
            The created assistance request details
        """
        question = params.question
        
        if isinstance(question, str):
            question = {
                "type": "text",
                "question": question
            }
        else:
            question = {
                "type": "json",
                **question
            }
        
        payload = {
            "type": params.type,
            "question": question
        }
        
        if params.agent_dump:
            payload["agentDump"] = params.agent_dump
        
        response = await self.api_client.post(
            f"/workspaces/{params.workspace_id}/tasks/{params.task_id}/human-assistance",
            json=payload
        )
        response.raise_for_status()
        try:
            return response.json()
        except Exception as e:
            logger.error(f"Error parsing JSON response in request_human_assistance: {str(e)}")
            return {"success": True, "message": "Human assistance requested (response could not be parsed)"}
    
    async def respond_to_chat(self, action: RespondChatMessageActionSchema):
        """
        Handle a chat message response request.
        This method can be overridden by extending classes to customize chat handling.
        
        Args:
            action: The chat action to handle
            action.messages: List of previous chat messages in the conversation
            action.workspace: The workspace context where the chat is happening
            action.me: Information about the current agent
        """
        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

        if action.messages:
            for msg in action.messages:
                messages.append({
                    "role": "user" if msg.author == "user" else "assistant",
                    "content": msg.message
                })

        try:
            await self.runtime_client.post("/runtime/chat", json={
                "tools": [self._convert_tool_to_json_schema(tool) for tool in self.tools],
                "messages": messages,
                "action": action.model_dump()
            })
        except Exception as error:
            self._handle_error(error, {
                "action": action.model_dump(),
                "context": "respond_to_chat"
            })
    
    def run(self):
        """
        Runs the agent's HTTP server in a blocking manner.
        
        Returns:
            None
        """
        logger.info(f"Agent server starting on port {self.port}")
        uvicorn.run(self.app, host="0.0.0.0", port=self.port)
    
    async def send_chat_message(self, params: SendChatMessageParams) -> Dict[str, Any]:
        """
        Sends a chat message from the agent.
        
        Args:
            params: Parameters for sending the chat message
            params.workspace_id: ID of the workspace where the chat is happening
            params.agent_id: ID of the agent sending the message
            params.message: Content of the message to send
            
        Returns:
            The sent message details
        """
        response = await self.api_client.post(
            f"/workspaces/{params.workspace_id}/agent-chat/{params.agent_id}/message",
            json={"message": params.message}
        )
        response.raise_for_status()
        try:
            return response.json()
        except Exception as e:
            logger.error(f"Error parsing JSON response in send_chat_message: {str(e)}")
            return {"success": True, "message": "Chat message sent (response could not be parsed)"}
    
    def _setup_routes(self):
        """
        Sets up the FastAPI routes for the agent's HTTP server.
        Configures health check endpoint and routes for tool execution.
        """
        @self.app.get("/health")
        async def health():
            return {"status": "ok", "uptime": os.getpid()}
        
        @self.app.post("/")
        async def root(request: Request):
            return await self._handle_root_route(request)
        
        @self.app.post("/tools/{tool_name}")
        async def tool_route(request: Request):
            return await self._handle_tool_route(request)
    
    async def start(self):
        """
        Starts the agent's HTTP server.
        
        Returns:
            None
            
        Raises:
            Exception: If server fails to start
        """
        config = uvicorn.Config(
            app=self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )
        self.server = uvicorn.Server(config)
        
        logger.info(f"Agent server starting on port {self.port}")
        await self.server.serve()
    
    async def stop(self):
        """
        Stops the agent's HTTP server.
        
        Returns:
            None
        """
        # Close HTTP clients
        await self.api_client.aclose()
        await self.runtime_client.aclose()
        
        # Stop server if running
        if self.server:
            self.server.should_exit = True
    
    async def update_task_status(self, params: UpdateTaskStatusParams) -> Dict[str, Any]:
        """
        Updates the status of a task.
        
        Args:
            params: Parameters for updating the status
            params.workspace_id: ID of the workspace containing the task
            params.task_id: ID of the task to update
            params.status: New status for the task (e.g., "IN_PROGRESS", "COMPLETED", "ERRORED")
            
        Returns:
            The updated task details
        """
        response = await self.api_client.put(
            f"/workspaces/{params.workspace_id}/tasks/{params.task_id}/status",
            json={"status": params.status}
        )
        response.raise_for_status()
        return response.json()
    
    async def upload_file(self, params: UploadFileParams) -> Dict[str, Any]:
        """
        Uploads a file to a workspace.
        
        Args:
            params: Parameters for the file upload
            params.workspace_id: ID of the workspace to upload to
            params.path: Path where the file should be stored
            params.task_ids: Optional task IDs to associate with the file
            params.skip_summarizer: Whether to skip file summarization
            params.file: The file content to upload (Buffer or string)
            
        Returns:
            The uploaded file details
        """
        form_data = {}
        form_data["path"] = params.path
        
        if params.task_ids is not None:
            if isinstance(params.task_ids, int):
                task_ids = [params.task_ids]
            else:
                task_ids = params.task_ids
            form_data["taskIds"] = json.dumps(task_ids)
        
        if params.skip_summarizer is not None:
            form_data["skipSummarizer"] = str(params.skip_summarizer).lower()
        
        # Prepare file content
        file_content = params.file
        if isinstance(file_content, str):
            file_content = file_content.encode('utf-8')
        
        files = {"file": (params.path, file_content)}
        
        response = await self.api_client.post(
            f"/workspaces/{params.workspace_id}/file",
            data=form_data,
            files=files
        )
        response.raise_for_status()
        try:
            return response.json()
        except Exception as e:
            logger.error(f"Error parsing JSON response in upload_file: {str(e)}")
            return {"success": True, "message": "File uploaded (response could not be parsed)"}
    
    def _convert_tool_to_json_schema(self, tool: Capability) -> Dict[str, Any]:
        """
        Converts a tool/capability to JSON schema format.
        
        Args:
            tool: The capability to convert
            
        Returns:
            The JSON schema representation of the tool
        """
        return {
            "name": tool.name,
            "description": tool.description,
            "schema": tool.schema.model_json_schema()
        }