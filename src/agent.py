"""
Main Agent implementation for the OpenServ Agent library.
"""

import logging
from typing import Optional, List, Dict, Any, TypeVar, Generic, Callable, Awaitable, cast, Union
import openai
import asyncio
import signal
from pydantic import BaseModel
import os
import json
import traceback

# Configure logging to show INFO and above
logging.basicConfig(level=logging.INFO)

from .config import Config, APIConfig, OpenAIConfig
from .client import OpenServClient, RuntimeClient
from .server import AgentServer
from .capability import Capability
from .exceptions import ConfigurationError, RuntimeError
from .types import (
    AgentOptions,
    DoTaskAction,
    RespondChatMessageAction,
    ProcessParams,
    ChatMessage,
    AgentAction,
    TaskStatus,
    GetTaskDetailParams,
    GetAgentsParams,
    GetTasksParams,
    CreateTaskParams,
    AddLogToTaskParams,
    RequestHumanAssistanceParams,
    UpdateTaskStatusParams,
    IntegrationCallRequest,
    ProxyConfiguration
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class Agent:
    """
    Main Agent class that orchestrates the OpenServ Agent functionality.
    
    This class handles:
    - Configuration and initialization
    - Tool/capability management
    - API communication
    - Task and chat message processing
    - Server management
    """
    
    def __init__(self, options: AgentOptions) -> None:
        """Initialize the agent with options."""
        self.config = Config(
            api=APIConfig(
                api_key=options.api_key,
                platform_url=options.platform_url,
                runtime_url=options.runtime_url
            ),
            openai=OpenAIConfig(
                api_key=options.openai_api_key or os.getenv('OPENAI_API_KEY'),
                model=options.model
            ),
            system_prompt=options.system_prompt,
            port=options.port or 7378,
            host=options.host or '0.0.0.0',
            log_level=options.log_level or 'debug',
            reload=options.reload or False
        )
        
        self.config.validate_api_key()
        self.openai_client = openai.AsyncOpenAI(api_key=self.config.openai.api_key)
        self.tools = []
        self.on_error = options.on_error
        
        # Initialize components
        self.server = AgentServer(self.config.server)
        self.server.set_agent(self)
        
    @property
    def api_client(self) -> OpenServClient:
        """Get the API client instance."""
        if not self._api_client:
            self._api_client = OpenServClient(self.config.api)
        return self._api_client

    @api_client.setter
    def api_client(self, client):
        """Set the API client instance (for testing)."""
        self._api_client = client

    @property
    def runtime_client(self) -> RuntimeClient:
        """Get the runtime client instance."""
        if not self._runtime_client:
            self._runtime_client = RuntimeClient(self.config.api)
        return self._runtime_client

    @runtime_client.setter
    def runtime_client(self, client):
        """Set the runtime client instance (for testing)."""
        self._runtime_client = client

    @property
    def openai_tools(self) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI function format."""
        return [{
            'type': 'function',
            'function': {
                'name': tool.name,
                'description': tool.description,
                'parameters': tool.schema.model_json_schema()
            }
        } for tool in self.tools]

    def add_capability(self, capability: Capability[T]) -> 'Agent':
        """Add a single capability to the agent."""
        if any(t.name == capability.name for t in self.tools):
            raise ValueError(f'Capability with name "{capability.name}" already exists')
        
        self.tools.append(capability)
        return self

    def add_capabilities(self, capabilities: List[Capability[T]]) -> 'Agent':
        """Add multiple capabilities to the agent."""
        # Check for duplicates first
        names = [cap.name for cap in capabilities]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate capability names found")
            
        for capability in capabilities:
            self.add_capability(capability)
        return self

    async def process(self, params: ProcessParams) -> Dict[str, Any]:
        """Process a conversation with OpenAI."""
        try:
            if not self.config.openai.api_key:
                raise ConfigurationError(
                    'OpenAI API key is required for process(). Please provide it in options or set OPENAI_API_KEY environment variable.'
                )

            current_messages = params.messages.copy()
            completion = None
            iteration_count = 0
            MAX_ITERATIONS = 10

            while iteration_count < MAX_ITERATIONS:
                completion = await self.openai_client.chat.completions.create(
                    model=self.config.openai.model,
                    messages=current_messages,
                    tools=self.openai_tools if self.tools else None
                )

                if not completion.choices or not completion.choices[0].message:
                    raise RuntimeError('No response from OpenAI')

                last_message = completion.choices[0].message

                # If no tool calls, we're done
                if not getattr(last_message, 'tool_calls', None):
                    return completion.model_dump()

                # Process each tool call
                tool_results = []
                for tool_call in last_message.tool_calls:
                    if not tool_call.function:
                        raise RuntimeError('Tool call function is missing')

                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    try:
                        tool = next((t for t in self.tools if t.name == name), None)
                        if not tool:
                            raise RuntimeError(f'Tool "{name}" not found')

                        result = await tool.run({"args": args}, current_messages)
                        tool_results.append({
                            'role': 'tool',
                            'content': str(result),
                            'tool_call_id': tool_call.id
                        })
                    except Exception as error:
                        error_message = str(error)
                        self.handle_error(error, {
                            'tool_call': tool_call,
                            'context': 'tool_execution'
                        })
                        tool_results.append({
                            'role': 'tool',
                            'content': json.dumps({'error': error_message}),
                            'tool_call_id': tool_call.id
                        })

                # Add messages to conversation
                current_messages.append(last_message)
                current_messages.extend(tool_results)
                iteration_count += 1

            raise RuntimeError('Max iterations reached without completion')
        except Exception as error:
            self.handle_error(error, {'context': 'process'})
            raise

    async def handle_root_route(self, body: Dict[str, Any]) -> None:
        """Handle the root route for task execution and chat message responses."""
        logger.info("Handling root route request with body type: %s", body.get('type'))
        try:
            if body.get('type') == 'do-task':
                logger.info("Processing do-task action")
                action = DoTaskAction.model_validate(body)
                # Fire and forget - don't await
                asyncio.create_task(self.do_task(action))
            elif body.get('type') == 'respond-chat-message':
                logger.info("Processing respond-chat-message action")
                action = RespondChatMessageAction.model_validate(body)
                # Fire and forget - don't await
                asyncio.create_task(self.respond_to_chat(action))
            else:
                raise ValueError('Invalid action type')
        except Exception as error:
            logger.error("Root route handler failed: %s", str(error), exc_info=True)
            raise

    async def handle_tool_route(
        self,
        tool_name: str,
        body: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Handle a tool route request."""
        try:
            logger.info(f"Handling tool route: {tool_name}")
            logger.debug(f"Request body: {body}")
            
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                raise ValueError(f'Tool "{tool_name}" not found')

            # Parse and validate args using the tool's schema
            args = body.get('args', {}) if body else {}
            validated_args = tool.schema.model_validate(args)
            
            logger.debug(f"Validated args: {validated_args}")
            
            result = await tool.run({"args": validated_args.model_dump()}, messages or [])
            logger.info(f"Tool execution result: {result}")
            
            return {"result": result}
        except Exception as error:
            logger.error(f"Tool route handler failed: {str(error)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.handle_error(error, {
                'request': {'tool': tool_name, 'body': body},
                'context': 'handle_tool_route'
            })
            raise

    async def start(self) -> None:
        """Start the agent server."""
        if not self.server:
            self.server = AgentServer(self.config.server)
            self.server.set_agent(self)
        await self.server.start()

    async def stop(self) -> None:
        """Stop the agent server."""
        if self.server:
            await self.server.stop()
            self.server = None

    async def do_task(self, action: DoTaskAction) -> None:
        """Handle a task execution request."""
        messages = [
            {'role': 'system', 'content': self.config.system_prompt}
        ]

        if action.task.description:
            messages.append({
                'role': 'user',
                'content': action.task.description
            })

        try:
            await self.runtime_client.execute_task(
                workspace_id=action.workspace.id,
                task_id=action.task.id,
                tools=[self._convert_tool_to_json_schema(t) for t in self.tools],
                messages=messages,
                action=action.model_dump()
            )
        except Exception as error:
            logger.error("Task execution failed: %s", str(error), exc_info=True)
            raise

    async def respond_to_chat(self, action: RespondChatMessageAction) -> None:
        """Handle a chat message response request."""
        messages = [
            {'role': 'system', 'content': self.config.system_prompt}
        ]

        if action.messages:
            for msg in action.messages:
                messages.append({
                    'role': 'user' if msg.author == 'user' else 'assistant',
                    'content': msg.message,
                    'id': msg.id,
                    'createdAt': msg.createdAt.isoformat()
                })

        try:
            # Fire and forget - don't wait for or process response
            await self.runtime_client.handle_chat(
                tools=[self._convert_tool_to_json_schema(t) for t in self.tools],
                messages=messages,
                action=action.model_dump()
            )
        except Exception as error:
            logger.error("Chat response failed: %s", str(error), exc_info=True)
            # Don't re-raise the error to match TypeScript behavior

    @staticmethod
    def _convert_tool_to_json_schema(tool: Capability[BaseModel]) -> Dict[str, Any]:
        """Convert a tool to JSON schema format."""
        return {
            'name': tool.name,
            'description': tool.description,
            'schema': tool.schema.model_json_schema()
        } 

    async def get_files(self, workspace_id: int) -> Dict[str, Any]:
        """Get files in a workspace."""
        response = await self.api_client.get(f"/workspaces/{workspace_id}/files")
        return response["data"]

    async def upload_file(self, workspace_id: int, path: str, file: Union[str, bytes], task_ids: Optional[List[int]] = None, skip_summarizer: bool = False) -> Dict[str, Any]:
        """Upload a file to a workspace."""
        response = await self.api_client.post(f"/workspaces/{workspace_id}/files", {
            "path": path,
            "file": file,
            "taskIds": task_ids,
            "skipSummarizer": skip_summarizer
        })
        return response["data"]

    async def get_tasks(self, workspace_id: Union[int, GetTasksParams]) -> Dict[str, Any]:
        """Gets a list of tasks in a workspace."""
        if isinstance(workspace_id, GetTasksParams):
            params = workspace_id
        else:
            params = GetTasksParams(workspace_id=workspace_id)

        response = await self.api_client.get(f"/workspaces/{params.workspace_id}/tasks")
        return response["data"]

    async def mark_task_as_errored(self, workspace_id: int, task_id: int, error: str) -> Dict[str, Any]:
        """Mark a task as errored."""
        response = await self.api_client.post(f"/workspaces/{workspace_id}/tasks/{task_id}/error", {
            "error": error
        })
        return response["data"]

    async def complete_task(self, workspace_id: int, task_id: int, output: str) -> Dict[str, Any]:
        """Complete a task."""
        response = await self.api_client.post(f"/workspaces/{workspace_id}/tasks/{task_id}/complete", {
            "output": output
        })
        return response["data"]

    async def send_chat_message(self, workspace_id: int, agent_id: int, message: str) -> Dict[str, Any]:
        """Send a chat message."""
        response = await self.api_client.post(f"/workspaces/{workspace_id}/agents/{agent_id}/chat", {
            "message": message
        })
        return response["data"]

    async def request_human_assistance(self, params: RequestHumanAssistanceParams) -> Dict[str, Any]:
        """Requests human assistance for a task."""
        response = await self.api_client.post(
            f"/workspaces/{params.workspace_id}/tasks/{params.task_id}/human-assistance",
            {
                "type": params.type,
                "question": params.question,
                "agentDump": params.agent_dump
            }
        )
        return response["data"]

    async def get_task_detail(self, params: GetTaskDetailParams) -> Dict[str, Any]:
        """Gets detailed information about a specific task."""
        response = await self.api_client.get(f"/workspaces/{params.workspace_id}/tasks/{params.task_id}/detail")
        return response["data"]

    async def get_agents(self, params: GetAgentsParams) -> Dict[str, Any]:
        """Gets a list of agents in a workspace."""
        response = await self.api_client.get(f"/workspaces/{params.workspace_id}/agents")
        return response["data"]

    async def create_task(self, params: CreateTaskParams) -> Dict[str, Any]:
        """Creates a new task in a workspace."""
        response = await self.api_client.post(f"/workspaces/{params.workspace_id}/task", {
            "assignee": params.assignee,
            "description": params.description,
            "body": params.body,
            "input": params.input,
            "expectedOutput": params.expected_output,
            "dependencies": params.dependencies
        })
        return response["data"]

    async def add_log_to_task(self, params: AddLogToTaskParams) -> Dict[str, Any]:
        """Adds a log entry to a task."""
        response = await self.api_client.post(
            f"/workspaces/{params.workspace_id}/tasks/{params.task_id}/log",
            {
                "severity": params.severity,
                "type": params.type,
                "body": params.body
            }
        )
        return response["data"]

    async def update_task_status(self, params: UpdateTaskStatusParams) -> Dict[str, Any]:
        """Updates the status of a task."""
        response = await self.api_client.put(
            f"/workspaces/{params.workspace_id}/tasks/{params.task_id}/status",
            {
                "status": params.status
            }
        )
        return response["data"]

    async def call_integration(self, integration: IntegrationCallRequest) -> Dict[str, Any]:
        """
        Calls an integration endpoint through the OpenServ platform.
        This method allows agents to interact with external services and APIs that are integrated with OpenServ.
        """
        response = await self.api_client.post(
            f"/workspaces/{integration.workspace_id}/integration/{integration.integration_id}/proxy",
            integration.details.model_dump()
        )
        return response["data"]

    def convert_to_openai_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI format."""
        openai_tools = []
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            }
            openai_tools.append(openai_tool)
        return openai_tools

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Default error handler that logs the error."""
        handler = self.on_error or (
            lambda err, ctx: logger.error(
                "Error in agent operation",
                extra={"error": str(err), **(ctx or {})}
            )
        )
        handler(error, context)
