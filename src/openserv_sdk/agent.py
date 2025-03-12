"""
Agent implementation for OpenServ.
"""

import os
import json
import logging
import asyncio
import traceback
from typing import Dict, Any, List, Optional, Union, TypeVar, Generic, Callable
from pydantic import BaseModel
from openai import AsyncOpenAI
import aiohttp
import mimetypes
from datetime import datetime

from openserv_sdk.config import Config, APIConfig, OpenAIConfig, ServerConfig
from openserv_sdk.client import OpenServClient, RuntimeClient
from openserv_sdk.server import AgentServer
from openserv_sdk.capability import Capability
from openserv_sdk.exceptions import ConfigurationError, RuntimeError, ToolError, AuthenticationError
from openserv_sdk.types import (
    AgentOptions, ProcessParams, RespondChatMessageAction,
    TaskStatus, DoTaskAction, IntegrationCallRequest,
    GetTasksParams, GetTaskDetailParams, GetAgentsParams,
    UploadFileParams, SendChatMessageParams, CreateTaskParams, AddLogToTaskParams,
    RequestHumanAssistanceParams, UpdateTaskStatusParams, GetFilesParams, UploadedFile,
    GetSecretsParams, GetSecretValueParams, SecretCollection, SecretValue
)
from openserv_sdk.logger import logger

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
            log_level=options.log_level or 'info',
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
        
        # Initialize server with enhanced config
        self.server = AgentServer(ServerConfig(
            host=self.config.host,
            port=self.config.port,
            debug=options.debug or False,
            version=options.version or "1.0.0",
            require_https=options.require_https or False,
            trusted_hosts=options.trusted_hosts,
            ssl_keyfile=options.ssl_keyfile,
            ssl_certfile=options.ssl_certfile,
            ssl_ca_certs=options.ssl_ca_certs,
            workers=options.workers or 1,
            limit_concurrency=options.limit_concurrency,
            timeout_keep_alive=options.timeout_keep_alive or 5
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

            # Prepare messages with system prompt
            messages = []
            if self.config.system_prompt:
                messages.append({"role": "system", "content": self.config.system_prompt})
            messages.extend([{"role": "user", "content": msg["content"]} for msg in params.messages])

            completion = None
            iteration_count = 0
            MAX_ITERATIONS = 10

            while iteration_count < MAX_ITERATIONS:
                try:
                    completion = await self._openai_client.chat.completions.create(
                        model=self.config.openai.model,
                        messages=messages,
                        tools=self.openai_tools if self._tools else None,
                        tool_choice="auto"
                    )
                except Exception as e:
                    logger.error(f"OpenAI API error: {str(e)}")
                    raise RuntimeError(f"OpenAI API error: {str(e)}")

                if not completion.choices:
                    raise RuntimeError("No response from OpenAI")

                last_message = completion.choices[0].message
                if not last_message.tool_calls:
                    return {"result": last_message.content}

                # Process tool calls
                tool_results = []
                for tool_call in last_message.tool_calls:
                    try:
                        result = await self._handle_tool_call(tool_call, messages)
                        tool_results.append(result)
                    except Exception as e:
                        logger.error(f"Tool execution error: {str(e)}")
                        tool_results.append(f"Error: {str(e)}")

                # Add results to messages
                messages.extend([
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
        except json.JSONDecodeError as e:
            raise ToolError(tool_name=tool_call.function.name, message=f"Invalid tool arguments: {str(e)}")
        except Exception as e:
            error_message = str(e)
            self.handle_error(e, {
                "tool_call": tool_call,
                "context": "tool_execution"
            })
            raise ToolError(tool_name=tool_call.function.name, message=error_message)

    async def handle_tool_route(self, tool_name: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a tool route."""
        try:
            if not hasattr(self, tool_name) or not callable(getattr(self, tool_name)):
                raise ToolError(f"Tool not found: {tool_name}")
            
            # Extract arguments from the request body
            args = body.get("args", {})
            messages = body.get("messages", [])
            
            # Call the tool method with the arguments
            result = await getattr(self, tool_name)(**args)
            return {"result": result}
        except Exception as e:
            self.handle_error(e, {
                "tool_name": tool_name,
                "body": body
            })
            if isinstance(e, ToolError):
                raise e
            raise ToolError(f"Error in tool '{tool_name}': {str(e)}")

    async def handle_root_route(self, body: Dict[str, Any]) -> None:
        """Handle a request to the root route."""
        try:
            if "action" not in body:
                raise ValueError("Missing 'action' in request body")

            action = body["action"]
            if action["type"] == "DO_TASK":
                asyncio.create_task(
                    self.do_task(action),
                    name=f"task_{action.get('taskId')}"
                )
            elif action["type"] == "RESPOND_TO_CHAT":
                asyncio.create_task(
                    self.respond_to_chat(action),
                    name=f"chat_{action.get('messageId')}"
                )
            else:
                raise ValueError(f"Unknown action type: {action['type']}")
        except Exception as e:
            self.handle_error(e, {
                "body": body,
                "context": "root_route"
            })
            raise

    async def start(self) -> None:
        """Start the agent server."""
        await self.server.start()

    async def stop(self) -> None:
        """Stop the agent server."""
        await self.server.stop()

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
            
            tools_json = [self._convert_tool_to_json_schema(t) for t in self._tools]
            action_data = action.model_dump()
            
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

    @staticmethod
    def _convert_tool_to_json_schema(tool: Capability[BaseModel]) -> Dict[str, Any]:
        """Convert a tool to JSON schema format."""
        schema = tool.schema.model_json_schema()
        # Remove title from schema if present
        if "title" in schema:
            del schema["title"]
        # Remove title from properties if present
        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop in schema["properties"].values():
                if isinstance(prop, dict) and "title" in prop:
                    del prop["title"]
        return {
            'name': tool.name,
            'description': tool.description,
            'parameters': schema
        }

    async def respond_to_chat(self, action: RespondChatMessageAction) -> None:
      
        """Respond to a chat message."""
        try:
            result = await self.process(ProcessParams(
                messages=[{"content": msg.message} for msg in action.messages],
                action=action
            ))
            
            if isinstance(result, dict) and "result" in result:
                await self.api_client.send_chat_message(
                    workspace_id=action.workspace.id,
                    agent_id=action.me.id,
                    message=result["result"]
                )
            else:
                raise RuntimeError("Invalid chat response format")
        except Exception as e:
            self.handle_error(e, {
                "action": action,
                "context": "respond_to_chat"
            })
            # Send error message to chat
            await self.api_client.send_chat_message(
                workspace_id=action.workspace.id,
                agent_id=action.me.id,
                message=f"Error: {str(e)}"
            )

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
        """Mark a task as errored with the given error message."""
        try:
            response = await self._api_client.post(
                f"/workspaces/{workspace_id}/task/{task_id}/error",
                {"error": error}
            )
            return response
        except Exception as e:
            logger.error(f"Failed to mark task as errored: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def complete_task(self, workspace_id: int, task_id: int, output: str) -> Dict[str, Any]:
        """Complete a task."""
        response = await self._api_client.post(
            f"/workspaces/{workspace_id}/task/{task_id}/complete",
            {"output": output}
        )
        return response["data"]

    async def send_chat_message(self, workspace_id: int, agent_id: int, message: str) -> Dict[str, Any]:
        """Send a chat message."""
        params = SendChatMessageParams(
            workspace_id=workspace_id,
            agent_id=agent_id,
            message=message
        )
        response = await self._api_client.post(
            f"/workspaces/{params.workspace_id}/agent-chat/{params.agent_id}/message",
            {"message": params.message}
        )
        return response["data"]

    async def request_human_assistance(self, params: RequestHumanAssistanceParams) -> Dict[str, Any]:
        """Requests human assistance for a task."""
        response = await self._api_client.post(
            f"/workspaces/{params.workspace_id}/task/{params.task_id}/human-assistance",
            {
                "type": params.type,
                "question": params.question,
                "agentDump": params.agent_dump
            }
        )
        return response["data"]

    async def get_task_detail(self, params: GetTaskDetailParams) -> Dict[str, Any]:
        """Gets detailed information about a specific task."""
        response = await self._api_client.get(f"/workspaces/{params.workspace_id}/task/{params.task_id}/detail")
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
            f"/workspaces/{params.workspace_id}/task/{params.task_id}/log",
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
                {"status": params.status.value if isinstance(params.status, TaskStatus) else params.status}
            )
            return response["data"]
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

    async def get_secrets(self, params: Union[int, GetSecretsParams]) -> Dict[str, Any]:
        """
        Get secrets from a workspace collection.
        
        Args:
            params: Either a workspace ID or GetSecretsParams object
            
        Returns:
            Dictionary containing the secrets collection data
            
        Raises:
            APIError: If the API request fails
            AuthenticationError: If authentication fails
        """
        if isinstance(params, int):
            params = GetSecretsParams(workspace_id=params)

        url = f"/workspaces/{params.workspace_id}/secrets"
        if params.collection_id:
            url += f"/{params.collection_id}"

        try:
            response = await self.api_client.get(url)
            return SecretCollection(**response["data"]).model_dump()
        except Exception as e:
            self.handle_error(e, {
                "params": params.model_dump(),
                "context": "get_secrets"
            })
            raise

    async def get_secret_value(self, params: GetSecretValueParams) -> str:
        """
        Get the value of a specific secret.
        
        Args:
            params: Parameters specifying the secret to retrieve
            
        Returns:
            The secret value as a string
            
        Raises:
            APIError: If the API request fails
            AuthenticationError: If authentication fails
        """
        try:
            response = await self.api_client.get(
                f"/workspaces/{params.workspace_id}/secrets/{params.collection_id}/{params.secret_id}/value"
            )
            return SecretValue(**response["data"]).value
        except Exception as e:
            self.handle_error(e, {
                "params": params.model_dump(),
                "context": "get_secret_value"
            })
            raise
