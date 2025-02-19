"""
Agent implementation for OpenServ.
"""

import os
import json
import logging
import asyncio
import traceback
from typing import Dict, Any, List, Optional, Union, TypeVar, Generic
from pydantic import BaseModel
from openai import AsyncOpenAI
import aiohttp
import mimetypes

from openserv_sdk.config import Config, APIConfig, OpenAIConfig, ServerConfig
from openserv_sdk.client import OpenServClient, RuntimeClient
from openserv_sdk.server import AgentServer
from openserv_sdk.capability import Capability
from openserv_sdk.exceptions import ConfigurationError, RuntimeError, ToolError
from openserv_sdk.types import (
    AgentOptions, ProcessParams, RespondChatMessageAction,
    TaskStatus, DoTaskAction, IntegrationCallRequest,
    GetTasksParams, GetTaskDetailParams, GetAgentsParams,
    UploadFileParams, SendChatMessageParams, CreateTaskParams, AddLogToTaskParams,
    RequestHumanAssistanceParams, UpdateTaskStatusParams, GetFilesParams, UploadedFile
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
        self._openai_api_key = options.openai_api_key
        
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

    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Handle errors by logging and calling the error handler if provided."""
        logger.error(f"Error: {str(error)}", exc_info=True)
        if self.config.on_error:
            try:
                self.config.on_error(error, context)
            except Exception as e:
                logger.error(f"Error handler failed: {str(e)}", exc_info=True)

    async def process(self, params: ProcessParams) -> Dict[str, Any]:
        """Process a request with the agent."""
        if not self._openai_api_key:
            raise ConfigurationError("OpenAI API key is required")

        try:
            if not self._openai_client:
                self._openai_client = AsyncOpenAI(api_key=self._openai_api_key)

            current_messages = [{"role": "user", "content": msg["content"]} for msg in params.messages]
            completion = None
            iteration_count = 0
            MAX_ITERATIONS = 10

            while iteration_count < MAX_ITERATIONS:
                completion = await self._openai_client.chat.completions.create(
                    messages=current_messages,
                    model="gpt-4",
                    tools=self.convert_to_openai_tools([{
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.schema.model_json_schema()
                    } for tool in self._tools]) if self._tools else None
                )

                if not completion.choices:
                    raise RuntimeError("No response from OpenAI")

                last_message = completion.choices[0].message
                if not last_message.tool_calls:
                    return {"result": last_message.content}

                # Process tool calls
                tool_results = await asyncio.gather(*[
                    self._handle_tool_call(tool_call, current_messages)
                    for tool_call in last_message.tool_calls
                ])

                # Add results to messages
                current_messages.extend([
                    {"role": "assistant", "content": last_message.content},
                    *[{"role": "tool", "content": result} for result in tool_results]
                ])
                iteration_count += 1

            raise RuntimeError("Max iterations reached without completion")
        except Exception as e:
            self.handle_error(e, {"context": "process"})
            raise

    async def _handle_tool_call(self, tool_call: Any, messages: List[Dict[str, str]]) -> str:
        """Handle a single tool call from OpenAI."""
        try:
            if not tool_call.function:
                raise RuntimeError("Tool call function is missing")

            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            tool = next((t for t in self._tools if t.name == name), None)
            if not tool:
                raise ToolError(tool_name=name, message=f"Tool '{name}' not found")

            result = await tool.run({"args": args}, messages)
            return str(result)
        except Exception as e:
            error_message = str(e)
            self.handle_error(e, {
                "tool_call": tool_call,
                "context": "tool_execution"
            })
            return json.dumps({"error": error_message})

    async def handle_tool_route(self, tool_name: str, body: Dict[str, Any]) -> str:
        """Handle a tool route request."""
        try:
            logger.info(f"Handling tool route: {tool_name}")
            logger.debug(f"Request body: {body}")

            tool = next((t for t in self._tools if t.name == tool_name), None)
            if not tool:
                raise ToolError(tool_name=tool_name, message="Tool not found")

            args = body.get("args", {})
            messages = body.get("messages", [])
            action = body.get("action")

            # Validate args against schema
            validated_args = tool.schema.model_validate(args)
            
            result = await tool.run({"args": validated_args.model_dump(), "action": action}, messages)
            return str(result)
        except Exception as error:
            self.handle_error(error, {
                "request": {"tool_name": tool_name, "body": body},
                "context": "handle_tool_route"
            })
            raise

    async def handle_root_route(self, body: Dict[str, Any]) -> None:
        """Handle the root route for task execution and chat message responses."""
        logger.info("Handling root route request with body type: %s", body.get('type'))
        try:
            if body.get('type') == 'do-task':
                logger.info("Processing do-task action")
                action = DoTaskAction.model_validate(body)
                # Create task and store its future
                task = asyncio.create_task(self.do_task(action))
                # Add error handler
                task.add_done_callback(self._handle_task_completion)
            elif body.get('type') == 'respond-chat-message':
                logger.info("Processing respond-chat-message action")
                action = RespondChatMessageAction.model_validate(body)
                chat_task = asyncio.create_task(self.respond_to_chat(action))
                chat_task.add_done_callback(self._handle_chat_completion)
            else:
                raise ValueError('Invalid action type')
        except Exception as error:
            logger.error("Root route handler failed: %s", str(error), exc_info=True)
            raise

    def _handle_task_completion(self, future: asyncio.Future) -> None:
        """Handle task completion and any errors."""
        try:
            future.result()  # This will raise any exceptions that occurred
        except Exception as error:
            logger.error("Task failed: %s", str(error), exc_info=True)
            self.handle_error(error, {"context": "task_completion"})

    def _handle_chat_completion(self, future: asyncio.Future) -> None:
        """Handle chat completion and any errors."""
        try:
            future.result()
        except Exception as error:
            logger.error("Chat failed: %s", str(error), exc_info=True)
            self.handle_error(error, {"context": "chat_completion"})

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
        logger.info(f"Processing task: {action.task}")
        logger.info(f"Task ID: {action.task.id}")
        logger.info(f"Workspace ID: {action.workspace.id}")

        messages = [
            {'role': 'system', 'content': self.config.system_prompt}
        ]

        if action.task.description:
            messages.append({
                'role': 'user',
                'content': action.task.description
            })

        try:
            # Update status to in-progress
            try:
                logger.info(f"Setting task {action.task.id} status to IN_PROGRESS")
                await self.update_task_status(UpdateTaskStatusParams(
                    workspace_id=action.workspace.id,
                    task_id=action.task.id,
                    status=TaskStatus.IN_PROGRESS
                ))
            except Exception as status_error:
                logger.warning(f"Failed to update task status: {str(status_error)}")

            # Execute the task and let the runtime handle the response
            logger.info(f"Executing task {action.task.id}")
            
            tools_json = [Agent._convert_tool_to_json_schema(t) for t in self._tools]

            action_data = {
                'type': action.type,
                'me': {
                    'id': action.me.id,
                    'name': action.me.name,
                    'kind': action.me.kind,
                    'is_built_by_agent_builder': action.me.is_built_by_agent_builder,
                    'system_prompt': action.me.system_prompt,
                    'capabilities_description': action.me.capabilities_description
                },
                'task': {
                    'id': action.task.id,
                    'description': action.task.description,
                    'body': action.task.body,
                    'expected_output': action.task.expected_output,
                    'input': action.task.input,
                    'dependencies': [
                        {
                            'id': d.id,
                            'description': d.description,
                            'output': d.output,
                            'status': d.status,
                            'attachments': [
                                {
                                    'id': a.id,
                                    'path': a.path,
                                    'full_url': a.full_url,
                                    'summary': a.summary
                                } for a in d.attachments
                            ]
                        } for d in action.task.dependencies
                    ],
                    'human_assistance_requests': [
                        {
                            'id': r.id,
                            'agent_dump': r.agent_dump,
                            'human_response': r.human_response,
                            'question': r.question,
                            'status': r.status,
                            'type': r.type
                        } for r in action.task.human_assistance_requests
                    ]
                },
                'workspace': {
                    'id': action.workspace.id,
                    'goal': action.workspace.goal,
                    'bucket_folder': action.workspace.bucket_folder,
                    'agents': [
                        {
                            'id': a.id,
                            'name': a.name,
                            'kind': a.kind,
                            'capabilities_description': a.capabilities_description
                        } for a in action.workspace.agents
                    ]
                },
                'integrations': [
                    {
                        'id': i.id,
                        'connection_id': i.connection_id,
                        'provider_config_key': i.provider_config_key,
                        'provider': i.provider,
                        'created': i.created,
                        'metadata': i.metadata,
                        'scopes': i.scopes,
                        'open_api': i.open_api
                    } for i in action.integrations
                ],
                'memories': [
                    {
                        'id': m.id,
                        'memory': m.memory,
                        'created_at': m.created_at.isoformat()
                    } for m in action.memories
                ]
            }

            logger.info(f"Tools JSON: {json.dumps(tools_json, indent=2)}")
            logger.info(f"Messages: {json.dumps(messages, indent=2)}")
            logger.info(f"Action data: {json.dumps(action_data, indent=2)}")

            try:
                await self._runtime_client.execute_task(
                    workspace_id=action.workspace.id,
                    task_id=action.task.id,
                    tools=tools_json,
                    messages=messages,
                    action=action_data
                )
            except Exception as exec_error:
                logger.error(f"Task execution failed: {str(exec_error)}")
                await self.mark_task_as_errored(
                    workspace_id=action.workspace.id,
                    task_id=action.task.id,
                    error=str(exec_error)
                )
                raise

        except Exception as error:
            logger.error(f"Task {action.task.id} execution failed with error: {str(error)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")

            try:
                await self.mark_task_as_errored(
                    workspace_id=action.workspace.id,
                    task_id=action.task.id,
                    error=str(error)
                )
            except Exception as mark_error:
                logger.error(f"Failed to mark task as errored: {str(mark_error)}")
            
            self.handle_error(error, {"context": "task_execution"})
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
                    'created_at': msg.created_at.isoformat()
                })

        try:
            # Get the chat response
            response = await self._runtime_client.handle_chat(
                tools=[Agent._convert_tool_to_json_schema(t) for t in self._tools],
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

    def _convert_tool_to_json_schema(tool: Capability[BaseModel]) -> Dict[str, Any]:
        """Convert a tool to JSON schema format."""
        schema = tool.schema.model_json_schema()
        # Remove title from properties if present
        if "properties" in schema:
            for prop in schema["properties"].values():
                if isinstance(prop, dict):
                    prop.pop("title", None)
        
        return {
            'name': tool.name,
            'description': tool.description,
            'parameters': schema
        }

    async def get_files(self, workspace_id: Union[int, GetFilesParams]) -> Dict[str, Any]:
        """Get files in a workspace."""
        if isinstance(workspace_id, GetFilesParams):
            params = workspace_id
        else:
            params = GetFilesParams(workspace_id=workspace_id)

        response = await self._api_client.get(f"/workspaces/{params.workspace_id}/files")
        return response["data"]

    async def upload_file(self, params: UploadFileParams) -> Dict[str, Any]:
        """Upload a file to a workspace."""
        data = aiohttp.FormData()
        data.add_field('path', params.path)
        if params.task_ids is not None:
            data.add_field('taskIds', json.dumps(params.task_ids))
        if params.skip_summarizer is not None:
            data.add_field('skipSummarizer', str(params.skip_summarizer).lower())
        
        # Handle both string and bytes file content
        content_type = params.content_type or (
            mimetypes.guess_type(params.path)[0] or 'application/octet-stream'
        )
        
        if isinstance(params.file, str):
            data.add_field('file', params.file.encode(), 
                          filename=params.path,
                          content_type=content_type)
        else:
            data.add_field('file', params.file, 
                          filename=params.path,
                          content_type=content_type)

        response = await self._api_client.post(
            f"/workspaces/{params.workspace_id}/file",
            data=data
        )
        return response["data"]

    async def get_file_content(self, workspace_id: int, file_id: str) -> bytes:
        """Get file content as bytes."""

        response = await self.api_client.get(f"/workspaces/{workspace_id}/file/{file_id}/content")

        if isinstance(response, dict) and "data" in response:
            return response["data"]
        return response

    async def save_output_file(self, workspace_id: int, file_name: str, content: Union[str, bytes], task_id: Optional[int] = None) -> str:
        """Save an output file and return its access URL."""
        params = UploadFileParams(
            workspace_id=workspace_id,
            path=file_name,
            file=content,
            task_ids=[task_id] if task_id else None
        )
        result = await self.upload_file(params)
        return result["url"]

    async def read_file_content(self, workspace_id: int, file_id: str, encoding: Optional[str] = None) -> Union[str, bytes]:
        """Read content from an uploaded file.
        
        Args:
            workspace_id: ID of the workspace containing the file
            file_id: ID of the file to read
            encoding: Optional encoding to use for text files (e.g., 'utf-8')
        
        Returns:
            str if encoding is provided, bytes otherwise
        """
        content = await self.get_file_content(workspace_id=workspace_id, file_id=file_id)
        if encoding:
            return content.decode(encoding)
        return content

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
        try:
            response = await self._api_client.post(f"/workspaces/{workspace_id}/task/{task_id}/error", {
                "error": error
            })
            if isinstance(response, bytes):
                return {"status": "error", "error": error}
            return response.get("data", {"status": "error", "error": error})
        except Exception as e:
            logger.error(f"Failed to mark task as errored: {str(e)}")
            return {"status": "error", "error": error}

    async def complete_task(self, workspace_id: int, task_id: int, output: str) -> Dict[str, Any]:
        """Complete a task."""
        response = await self._api_client.post(f"/workspaces/{workspace_id}/tasks/{task_id}/complete", {
            "output": output
        })
        return response["data"]

    async def send_chat_message(self, workspace_id: int, agent_id: int, message: str) -> Dict[str, Any]:
        """Send a chat message."""
        params = SendChatMessageParams(
            workspace_id=workspace_id,
            agent_id=agent_id,
            message=message
        )
        response = await self._api_client.post(
            f"/workspaces/{params.workspace_id}/agents/{params.agent_id}/chat",
            {"message": params.message}
        )
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
        """Update a task's status."""
        try:
            response = await self._api_client.post(
                f"/workspaces/{params.workspace_id}/task/{params.task_id}/status",
                {"status": params.status}
            )
            return response.get("data", {"status": "success"})
        except Exception as e:
            logger.error(f"Failed to update task status: {str(e)}")
            return {"status": "error", "error": str(e)}

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
