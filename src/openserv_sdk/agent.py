"""
Agent implementation for OpenServ.
"""

import os
import json
import logging
import traceback
from typing import Dict, Any, List, Optional, Union, TypeVar, Generic
from pydantic import BaseModel
from openai import AsyncOpenAI

from openserv_sdk.config import Config, APIConfig, OpenAIConfig, ServerConfig
from openserv_sdk.client import OpenServClient, RuntimeClient
from openserv_sdk.server import AgentServer
from openserv_sdk.capability import Capability
from openserv_sdk.exceptions import ConfigurationError, RuntimeError, ToolError
from openserv_sdk.types import (
    AgentOptions, ProcessParams, RespondChatMessageAction, AgentKind,
    TaskStatus, Workspace, AgentBase, Task, DoTaskAction, IntegrationCallRequest,
    GetTasksParams, GetTaskDetailParams, GetAgentsParams, GetFilesParams,
    UploadFileParams, MarkTaskAsErroredParams, CompleteTaskParams,
    SendChatMessageParams, CreateTaskParams, AddLogToTaskParams,
    RequestHumanAssistanceParams, UpdateTaskStatusParams
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
            reload=options.reload or False,
            on_error=options.on_error
        )
        
        self.config.validate_api_key()
        
        # Initialize private instance variables
        self._api_client: Optional[OpenServClient] = None
        self._runtime_client: Optional[RuntimeClient] = None
        self._openai_client: Optional[AsyncOpenAI] = None
        self._server: Optional[AgentServer] = None
        self._tools: List[Capability[BaseModel]] = []
        
        # Initialize clients
        self.runtime_client = RuntimeClient(self.config.api)
        self.api_client = OpenServClient(self.config.api)
        
        # Initialize server
        self.server = AgentServer(ServerConfig(
            host=self.config.host,
            port=self.config.port
        ))
        self.server.set_agent(self)
        
    @property
    def tools(self) -> List[Capability[BaseModel]]:
        """Get the tools list."""
        return self._tools

    @tools.setter
    def tools(self, value: List[Capability[BaseModel]]) -> None:
        """Set the tools list."""
        self._tools = value

    @property
    def server(self) -> Optional[AgentServer]:
        """Get the server instance."""
        return self._server

    @server.setter
    def server(self, value: Optional[AgentServer]) -> None:
        """Set the server instance."""
        self._server = value

    @property
    def openai_client(self) -> Optional[AsyncOpenAI]:
        """Get the OpenAI client instance."""
        return self._openai_client

    @openai_client.setter
    def openai_client(self, value: Optional[AsyncOpenAI]) -> None:
        """Set the OpenAI client instance."""
        self._openai_client = value

    @property
    def api_client(self) -> OpenServClient:
        """Get the API client instance."""
        if not self._api_client:
            self._api_client = OpenServClient(self.config.api)
        return self._api_client

    @api_client.setter
    def api_client(self, client: OpenServClient) -> None:
        """Set the API client instance (for testing)."""
        self._api_client = client

    @property
    def runtime_client(self) -> RuntimeClient:
        """Get the runtime client instance."""
        if not self._runtime_client:
            self._runtime_client = RuntimeClient(self.config.api)
        return self._runtime_client

    @runtime_client.setter
    def runtime_client(self, client: RuntimeClient) -> None:
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
        if any(t.name == capability.name for t in self._tools):
            raise ValueError(f'Capability with name "{capability.name}" already exists')
        
        self._tools.append(capability)
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

    async def handle_tool_route(self, tool_name: str, body: Dict[str, Any]) -> str:
        """Handle a tool route request."""
        try:
            logger.info(f"Handling tool route: {tool_name}")
            logger.debug(f"Request body: {body}")

            tool = next((t for t in self._tools if t.name == tool_name), None)
            if not tool:
                raise ToolError(tool_name=tool_name, message="Tool not found")

            # Extract args and messages from the request body
            args = body.get("args", {})
            messages = body.get("messages", [])

            # Run the tool
            result = await tool.run({"args": args}, messages)
            return str(result)

        except Exception as error:
            logger.error("Error: %s", str(error), exc_info=True)
            if self.config.on_error:
                self.config.on_error(error, {"context": "handle_tool_route"})
            raise

    async def process(self, params: ProcessParams) -> Dict[str, Any]:
        """Process a conversation with OpenAI."""
        try:
            # Check for OpenAI API key before attempting to use it
            if not self.config.openai.api_key:
                raise ConfigurationError(
                    'OpenAI API key is required for process(). Please provide it in options or set OPENAI_API_KEY environment variable.'
                )

            # Create client if not exists
            if not self._openai_client:
                self._openai_client = AsyncOpenAI(api_key=self.config.openai.api_key)

            current_messages = params.messages.copy()
            completion = None
            iteration_count = 0
            MAX_ITERATIONS = 10

            while iteration_count < MAX_ITERATIONS:
                if not self._openai_client:
                    raise ConfigurationError("OpenAI client not initialized")

                completion = await self._openai_client.chat.completions.create(
                    model=self.config.openai.model,
                    messages=current_messages,
                    tools=self.openai_tools if self._tools else None
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
                        continue

                    try:
                        result = await self.handle_tool_route(
                            tool_name=tool_call.function.name,
                            body={"args": json.loads(tool_call.function.arguments)}
                        )
                        tool_results.append({
                            'tool_call_id': tool_call.id,
                            'output': result
                        })
                    except Exception as e:
                        logger.error(f"Tool execution failed: {str(e)}")
                        if self.config.on_error:
                            self.config.on_error(e, {"context": "process_tool_call"})
                        raise

                # Add assistant's response with tool calls
                current_messages.append({
                    'role': 'assistant',
                    'content': None,
                    'tool_calls': [
                        {
                            'id': tc.id,
                            'type': 'function',
                            'function': {
                                'name': tc.function.name,
                                'arguments': tc.function.arguments
                            }
                        } for tc in last_message.tool_calls
                    ]
                })

                # Add tool results
                for result in tool_results:
                    current_messages.append({
                        'role': 'tool',
                        'content': result['output'],
                        'tool_call_id': result['tool_call_id']
                    })

                iteration_count += 1

            raise RuntimeError(f"Max iterations ({MAX_ITERATIONS}) reached without completion")

        except Exception as error:
            logger.error("Error: %s", str(error), exc_info=True)
            if self.config.on_error:
                self.config.on_error(error, {"context": "process"})
            raise

    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Handle errors by logging and calling the error handler if provided."""
        logger.error(f"Error: {str(error)}", exc_info=True)
        if self.config.on_error:
            try:
                self.config.on_error(error, context)
            except Exception as e:
                logger.error(f"Error handler failed: {str(e)}", exc_info=True)

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

    async def start(self) -> None:
        """Start the agent server."""
        if not self._server:
            self._server = AgentServer(ServerConfig(
                host=self.config.host,
                port=self.config.port
            ))
            self._server.set_agent(self)
        await self._server.start()

    async def stop(self) -> None:
        """Stop the agent server."""
        if self._server:
            await self._server.stop()
            self._server = None

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
            # Execute the task
            response = await self._runtime_client.execute_task(
                workspace_id=action.workspace.id,
                task_id=action.task.id,
                tools=[self._convert_tool_to_json_schema(t) for t in self._tools],
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
            # Get the chat response
            response = await self._runtime_client.handle_chat(
                tools=[self._convert_tool_to_json_schema(t) for t in self._tools],
                messages=messages,
                action=action.model_dump()
            )

            # Send the response
            await self.send_chat_message(
                workspace_id=action.workspace.id,
                agent_id=action.me.id,
                message=str(response)
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
        response = await self._api_client.get(f"/workspaces/{workspace_id}/files")
        return response["data"]

    async def upload_file(self, workspace_id: int, path: str, file: Union[str, bytes], task_ids: Optional[List[int]] = None, skip_summarizer: bool = False) -> Dict[str, Any]:
        """Upload a file to a workspace."""
        response = await self._api_client.post(f"/workspaces/{workspace_id}/files", {
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

        response = await self._api_client.get(f"/workspaces/{params.workspace_id}/tasks")
        return response["data"]

    async def mark_task_as_errored(self, workspace_id: int, task_id: int, error: str) -> Dict[str, Any]:
        """Mark a task as errored."""
        response = await self._api_client.post(f"/workspaces/{workspace_id}/tasks/{task_id}/error", {
            "error": error
        })
        return response["data"]

    async def complete_task(self, workspace_id: int, task_id: int, output: str) -> Dict[str, Any]:
        """Complete a task."""
        response = await self._api_client.post(f"/workspaces/{workspace_id}/tasks/{task_id}/complete", {
            "output": output
        })
        return response["data"]

    async def send_chat_message(self, workspace_id: int, agent_id: int, message: str) -> Dict[str, Any]:
        """Send a chat message."""
        response = await self._api_client.post(f"/workspaces/{workspace_id}/agents/{agent_id}/chat", {
            "message": message
        })
        return response["data"]

    async def request_human_assistance(self, params: RequestHumanAssistanceParams) -> Dict[str, Any]:
        """Requests human assistance for a task."""
        response = await self._api_client.post(
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
        response = await self._api_client.get(f"/workspaces/{params.workspace_id}/tasks/{params.task_id}/detail")
        return response["data"]

    async def get_agents(self, params: GetAgentsParams) -> Dict[str, Any]:
        """Gets a list of agents in a workspace."""
        response = await self._api_client.get(f"/workspaces/{params.workspace_id}/agents")
        return response["data"]

    async def create_task(self, params: CreateTaskParams) -> Dict[str, Any]:
        """Creates a new task in a workspace."""
        response = await self._api_client.post(f"/workspaces/{params.workspace_id}/task", {
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
        response = await self._api_client.post(
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
        response = await self._api_client.put(
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
        response = await self._api_client.post(
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
