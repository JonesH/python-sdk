"""
Agent implementation for OpenServ.
"""

import os
import json
import logging
import asyncio
import traceback
from typing import Dict, Any, List, Optional, Union, TypeVar, Generic, Callable, Type, Awaitable, cast
from pydantic import BaseModel
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

PLATFORM_URL = os.getenv('OPENSERV_API_URL', 'https://api.openserv.ai')
RUNTIME_URL = os.getenv('OPENSERV_RUNTIME_URL', 'https://agents.openserv.ai')
DEFAULT_PORT = int(os.getenv('PORT', '7378'))

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
        """
        Initialize the agent with options.
        
        Args:
            options: Configuration options for the agent
                - api_key: OpenServ API key (can also be set via OPENSERV_API_KEY env var)
                - system_prompt: System prompt that defines agent behavior
                - openai_api_key: OpenAI API key (can also be set via OPENAI_API_KEY env var)
                - port: Port for the agent server (default: 7378)
                - host: Host for the agent server (default: '0.0.0.0')
                - platform_url: URL for the OpenServ API (default: 'https://api.openserv.ai')
                - runtime_url: URL for the OpenServ Runtime API (default: 'https://agents.openserv.ai')
                - on_error: Error handler function
        """
        # Get API key from options or environment variable
        api_key = options.api_key or os.getenv('OPENSERV_API_KEY')
        if not api_key:
            raise ConfigurationError("OpenServ API key is required. Provide it in options or set OPENSERV_API_KEY environment variable.")
        
        # Get OpenAI API key from options or environment variable
        openai_api_key = options.openai_api_key or os.getenv('OPENAI_API_KEY')
        
        self.config = Config(
            api=APIConfig(
                api_key=api_key,
                platform_url=options.platform_url or PLATFORM_URL,
                runtime_url=options.runtime_url or RUNTIME_URL
            ),
            openai=OpenAIConfig(
                api_key=openai_api_key,
                model=options.model or "gpt-4"
            ),
            system_prompt=options.system_prompt,
            port=options.port or DEFAULT_PORT,
            host=options.host or '0.0.0.0',
            log_level=options.log_level or 'info',
            reload=options.reload or False,
            on_error=options.on_error
        )
        
        # Initialize private instance variables
        self._api_client: Optional[OpenServClient] = None
        self._runtime_client: Optional[RuntimeClient] = None
        self._openai_client: Optional[Any] = None
        self._server: Optional[AgentServer] = None
        self._tools: List[Capability[BaseModel]] = []
        
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
        
        logger.info(f"Agent initialized with system prompt: {self.config.system_prompt[:50]}...")
        logger.info(f"Agent will listen on {self.config.host}:{self.config.port}")
        
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
    def openai_client(self) -> Optional[Any]:
        """Get the OpenAI client instance."""
        return self._openai_client

    @openai_client.setter
    def openai_client(self, value: Optional[Any]) -> None:
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

    def add_capability(self, capability: Union[Capability[T], Dict[str, Any]]) -> 'Agent':
        """
        Add a capability to the agent.
        
        Args:
            capability: Either a Capability object or a dictionary with capability properties
                If a dictionary, must contain: name, description, schema, run
                
        Returns:
            The agent instance for chaining
            
        Raises:
            ValueError: If capability with the same name already exists or if dictionary is missing required keys
        """
        # If capability is a dictionary, convert it to a Capability object
        if isinstance(capability, dict):
            required_keys = ['name', 'description', 'schema', 'run']
            missing_keys = [key for key in required_keys if key not in capability]
            if missing_keys:
                raise ValueError(f"Missing required keys in capability dictionary: {missing_keys}")
                
            capability = Capability(
                name=capability['name'],
                description=capability['description'],
                schema=capability['schema'],
                run=capability['run']
            )
            
        # Check for duplicate name
        if any(tool.name == capability.name for tool in self._tools):
            raise ValueError(f"Duplicate capability name: {capability.name}")
            
        # Bind the agent to the capability
        capability.bind_agent(self)
        
        # Add the capability to the tools list
        self._tools.append(capability)
        
        logger.info(f"Added capability: {capability.name}")
        return self

    def add_capabilities(self, capabilities: List[Union[Capability[T], Dict[str, Any]]]) -> 'Agent':
        """Add multiple capabilities at once."""
        for capability in capabilities:
            self.add_capability(capability)
        return self
        
    def capability(self, name: str = None, description: str = None, schema: type[BaseModel] = None):
        """
        Decorator for adding a capability to the agent.
        
        Example:
            @agent.capability(name="greet", description="Greet a user", schema=GreetSchema)
            async def greet(params: dict, messages: list) -> str:
                name = params["args"]["name"]
                return f"Hello, {name}!"
                
        Args:
            name: Name of the capability (defaults to function name if not provided)
            description: Optional description of the capability
            schema: Optional Pydantic model class for parameter validation
            
        Returns:
            Decorator function
        """
        def decorator(func):
            # Use function name if name not provided
            capability_name = name or func.__name__
            
            # If no description is provided, use the function's docstring
            func_description = description or func.__doc__ or f"Execute the {capability_name} capability"
            
            # Use provided schema or create a default one
            schema_class = schema
            
            if schema_class is None:
                # Create a dynamic schema class
                class DynamicSchema(BaseModel):
                    pass
                
                schema_class = DynamicSchema
            
            # Create and add the capability
            capability = Capability(
                name=capability_name,
                description=func_description,
                schema=schema_class,
                run=func
            )
            self.add_capability(capability)
            return func
        return decorator

    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """
        Handle errors by logging and calling the error handler if provided.
        
        This method logs the error and calls the custom error handler if provided.
        It ensures that all errors are properly logged and handled.
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
        """
        # Prepare error message with context
        error_message = f"Error: {str(error)}"
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items() if k != "action")
            error_message += f" (Context: {context_str})"
            
        # Log the error with traceback
        logger.error(error_message, exc_info=True)
        
        # Call custom error handler if provided
        if self.config.on_error:
            try:
                self.config.on_error(error, context)
            except Exception as handler_error:
                logger.error(f"Error handler failed: {str(handler_error)}", exc_info=True)

    async def process(self, params: ProcessParams) -> Dict[str, Any]:
        """
        Process a request using the OpenAI API.
        
        This method sends messages to OpenAI with the agent's capabilities as tools,
        handles any tool calls that are returned, and returns the final response.
        
        Args:
            params: ProcessParams object containing:
                - messages: List of chat messages to process
                - action: Optional action context
            
        Returns:
            Response from OpenAI API after processing all tool calls
            
        Raises:
            RuntimeError: If OpenAI API call fails or returns an empty response
            ConfigurationError: If OpenAI API key is not provided
        """
        if not self.config.openai.api_key:
            raise ConfigurationError("OpenAI API key is required for process method. Provide it in options or set OPENAI_API_KEY environment variable.")
            
        try:
            # Import OpenAI here to make it optional
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError(
                    "The 'openai' package is required to use the process method. "
                    "Install it with 'pip install \"openserv-sdk[openai]\"' or 'pip install openai==0.28.1'"
                )
                
            if not self.openai_client:
                # Initialize OpenAI client with compatibility for different versions
                try:
                    # For newer versions of OpenAI package
                    self.openai_client = AsyncOpenAI(api_key=self.config.openai.api_key)
                except TypeError as e:
                    if "unexpected keyword argument 'proxies'" in str(e):
                        # For newer versions that don't support proxies
                        logger.info("Using OpenAI client without proxies")
                        self.openai_client = AsyncOpenAI(
                            api_key=self.config.openai.api_key,
                            base_url="https://api.openai.com/v1"
                        )
                    else:
                        # Try older initialization format for OpenAI < 1.0.0
                        logger.info("Falling back to older OpenAI client initialization")
                        from openai import AsyncOpenAI as LegacyAsyncOpenAI
                        self.openai_client = LegacyAsyncOpenAI(api_key=self.config.openai.api_key)
                
            # Prepare messages
            messages = params.messages.copy()
            
            # Add system prompt if provided and not already in messages
            if self.config.system_prompt and not any(msg.get('role') == 'system' for msg in messages):
                messages.insert(0, {"role": "system", "content": self.config.system_prompt})
                
            # Prepare tools for OpenAI API
            tools = self.openai_tools if self.tools else None
            
            # Maximum number of iterations for tool calls
            max_iterations = 10
            iteration_count = 0
            
            while iteration_count < max_iterations:
                iteration_count += 1
                logger.debug(f"Process iteration {iteration_count}/{max_iterations}")
                
                # Call OpenAI API
                response = await self.openai_client.chat.completions.create(
                    model=self.config.openai.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto" if tools else None
                )
                
                if not response or not response.choices:
                    raise RuntimeError("Empty response from OpenAI API")
                    
                choice = response.choices[0]
                message = choice.message
                
                # If no tool calls, return the message
                if not message.tool_calls:
                    return message.model_dump()
                
                # Handle tool calls
                tool_results = []
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        # Parse arguments
                        arguments = json.loads(tool_call.function.arguments)
                        
                        # Call the tool
                        result = await self._handle_tool_call(
                            {"name": tool_name, "arguments": arguments},
                            messages
                        )
                        
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": result
                        })
                    except Exception as e:
                        logger.error(f"Error calling tool {tool_name}: {str(e)}")
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": f"Error: {str(e)}"
                        })
                
                # Add assistant message and tool results to messages
                messages.append(message.model_dump())
                messages.extend(tool_results)
            
            # If we've reached the maximum number of iterations, return the last message
            raise RuntimeError(f"Reached maximum number of iterations ({max_iterations}) without resolution")
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            self.handle_error(e, {"messages": params.messages})
            raise RuntimeError(f"Failed to process request: {str(e)}") from e

    async def _handle_tool_call(self, tool_call: Any, messages: List[Dict[str, str]]) -> str:
        """
        Handle a tool call from OpenAI API.
        
        Args:
            tool_call: Tool call object from OpenAI API
            messages: List of messages for context
            
        Returns:
            Result of tool execution as string
            
        Raises:
            ToolError: If tool execution fails
        """
        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]
        
        # Find the tool
        tool = next((t for t in self._tools if t.name == tool_name), None)
        if not tool:
            raise ToolError(tool_name, f"Tool not found: {tool_name}")
            
        try:
            # Execute the tool
            result = await tool.run(
                {"args": arguments, "action": None},
                messages
            )
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            self.handle_error(e, {"tool": tool_name, "arguments": arguments})
            raise ToolError(tool_name, str(e), e)

    async def handle_tool_route(self, tool_name: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a tool route request.
        
        This method processes a request to execute a specific tool/capability.
        It validates the tool exists, extracts arguments and messages, and executes the tool.
        
        Args:
            tool_name: Name of the tool to execute
            body: Request body containing arguments and messages
            
        Returns:
            Dictionary containing the tool execution result
            
        Raises:
            ToolError: If tool execution fails or tool not found
        """
        logger.info(f"Handling tool route: {tool_name}")
        
        try:
            # Extract arguments and messages
            args = body.get("args", {})
            messages = body.get("messages", [])
            action = body.get("action")
            
            logger.debug(f"Tool arguments: {json.dumps(args, indent=2)}")
            logger.debug(f"Messages count: {len(messages)}")
            
            # Find the tool
            tool = next((t for t in self._tools if t.name == tool_name), None)
            if not tool:
                error_msg = f"Tool not found: {tool_name}"
                logger.error(error_msg)
                raise ToolError(tool_name, error_msg)
                
            # Execute the tool
            logger.info(f"Executing tool: {tool_name}")
            result = await tool.run(
                {"args": args, "action": action},
                messages
            )
            
            logger.info(f"Tool execution successful: {result[:50]}...")
            return {"result": result}
            
        except ToolError as e:
            # Re-raise ToolError instances
            logger.error(f"Tool error: {str(e)}")
            self.handle_error(e, {"tool": tool_name, "body": body})
            raise
            
        except Exception as e:
            # Wrap other exceptions in ToolError
            logger.error(f"Error handling tool route {tool_name}: {str(e)}")
            self.handle_error(e, {"tool": tool_name, "body": body})
            raise ToolError(tool_name, str(e), e)

    async def handle_root_route(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a root route request.
        
        This method processes incoming requests to the root endpoint of the agent server.
        It determines the action type and dispatches to the appropriate handler.
        
        Args:
            body: Request body containing action type and parameters
            
        Returns:
            Response object with status and any relevant data
            
        Raises:
            RuntimeError: If action execution fails or action type is unknown
        """
        try:
            # Extract action from the request body
            if "action" not in body:
                raise RuntimeError("Missing 'action' field in request body")
                
            action = body["action"]
            if not isinstance(action, dict):
                raise RuntimeError("'action' must be an object")
                
            # Extract action type
            action_type = action.get("type")
            if not action_type:
                raise RuntimeError("Missing 'type' field in action")
                
            # Handle different action types
            if action_type == "do-task":
                # Create a background task to handle the request
                asyncio.create_task(
                    self.do_task(action),
                    name=f"task_{action.get('task', {}).get('id', 'unknown')}"
                )
                return {"status": "ok", "message": "Task processing started"}
                
            elif action_type == "respond-chat-message":
                # Create a background task to handle the request
                asyncio.create_task(
                    self.respond_to_chat(action),
                    name=f"chat_{action.get('messages', [{}])[-1].get('id', 'unknown')}"
                )
                return {"status": "ok", "message": "Chat response processing started"}
                
            else:
                raise RuntimeError(f"Unknown action type: {action_type}")
                
        except Exception as e:
            logger.error(f"Error handling root route: {str(e)}")
            self.handle_error(e, {"body": body})
            raise RuntimeError(f"Failed to handle request: {str(e)}") from e

    async def start(self) -> None:
        """Start the agent server."""
        logger.info(f"Starting agent server on {self.config.host}:{self.config.port}")
        await self.server.start()

    async def stop(self) -> None:
        """Stop the agent server."""
        logger.info("Stopping agent server")
        await self.server.stop()

    async def do_task(self, action: DoTaskAction) -> None:
        """
        Handle a task execution request.
        
        This method processes a task execution request from the OpenServ platform.
        It updates the task status, executes the task, and handles any errors.
        
        Args:
            action: DoTaskAction object containing task details and context
            
        Raises:
            RuntimeError: If task execution fails
        """
        if not action.task or not action.task.id:
            logger.error("Invalid task in action")
            return
            
        logger.info(f"Processing task: {action.task.description}")
        logger.info(f"Task ID: {action.task.id}")
        logger.info(f"Workspace ID: {action.workspace.id}")

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

            # Prepare initial messages
            messages = []
            
            # Add system prompt if provided
            if self.config.system_prompt:
                messages.append({
                    'role': 'system',
                    'content': self.config.system_prompt
                })

            # Add task description as user message
            if action.task.description:
                messages.append({
                    'role': 'user',
                    'content': action.task.description
                })
                
            # Add task body if provided
            if action.task.body:
                messages.append({
                    'role': 'user',
                    'content': action.task.body
                })

            # Execute the task and let the runtime handle the response
            logger.info(f"Executing task {action.task.id}")
            
            tools_json = [self._convert_tool_to_json_schema(t) for t in self._tools]
            action_data = action.model_dump()
            
            logger.debug(f"Tools JSON: {json.dumps(tools_json, indent=2)}")
            logger.debug(f"Messages: {json.dumps(messages, indent=2)}")
            logger.debug(f"Action data: {json.dumps(action_data, indent=2)}")

            try:
                await self._runtime_client.execute_task(
                    workspace_id=action.workspace.id,
                    task_id=action.task.id,
                    tools=tools_json,
                    messages=messages,
                    action=action_data
                )
                logger.info(f"Task {action.task.id} execution request sent successfully")
            except Exception as exec_error:
                logger.error(f"Task execution failed: {str(exec_error)}")
                await self.mark_task_as_errored(
                    workspace_id=action.workspace.id,
                    task_id=action.task.id,
                    error=str(exec_error)
                )
                raise RuntimeError(f"Failed to execute task: {str(exec_error)}") from exec_error

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
            
            self.handle_error(error, {"context": "task_execution", "action": action})
            raise RuntimeError(f"Task execution failed: {str(error)}") from error

    @staticmethod
    def _convert_tool_to_json_schema(tool: Capability[BaseModel]) -> Dict[str, Any]:
        """
        Convert a tool to JSON schema format for OpenAI API.
        
        This method converts a Capability object to the JSON schema format
        required by the OpenAI API for function calling.
        
        Args:
            tool: The Capability object to convert
            
        Returns:
            Dictionary containing the tool's JSON schema
        """
        schema = tool.schema.model_json_schema()
        
        # Clean up schema for OpenAI
        # Remove title from schema if present
        if "title" in schema:
            del schema["title"]
            
        # Remove title from properties if present
        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop in schema["properties"].values():
                if isinstance(prop, dict) and "title" in prop:
                    del prop["title"]
                    
        # Ensure required field is present
        if "required" not in schema:
            schema["required"] = []
            
        return {
            'name': tool.name,
            'description': tool.description,
            'parameters': schema
        }

    async def respond_to_chat(self, action: RespondChatMessageAction) -> None:
        """
        Respond to a chat message.
        
        This method processes a chat message from the OpenServ platform and generates a response.
        It uses the agent's capabilities and the OpenAI API to generate the response.
        
        Args:
            action: RespondChatMessageAction object containing message details and context
            
        Raises:
            RuntimeError: If chat response generation fails
        """
        if not action.messages or len(action.messages) == 0:
            logger.error("No messages in action")
            return
            
        logger.info(f"Responding to chat in workspace {action.workspace.id}")
        logger.info(f"Agent ID: {action.me.id}")
        logger.info(f"Message count: {len(action.messages)}")
        
        try:
            # Prepare messages for processing
            messages = []
            
            # Add system prompt if provided
            if self.config.system_prompt:
                messages.append({
                    'role': 'system',
                    'content': self.config.system_prompt
                })
                
            # Add chat messages
            for msg in action.messages:
                role = 'assistant' if msg.sender_id == action.me.id else 'user'
                messages.append({
                    'role': role,
                    'content': msg.message
                })
            
            # Process the messages
            result = await self.process(ProcessParams(
                messages=messages,
                action=action
            ))
            
            # Extract the response content
            response_content = result.get('content', '')
            if not response_content:
                logger.warning("Empty response content from process method")
                response_content = "I'm sorry, I couldn't generate a response."
                
            # Send the response
            logger.info(f"Sending chat response: {response_content[:50]}...")
            await self.api_client.send_chat_message(
                workspace_id=action.workspace.id,
                agent_id=action.me.id,
                message=response_content
            )
            logger.info("Chat response sent successfully")
            
        except Exception as e:
            logger.error(f"Failed to respond to chat: {str(e)}")
            self.handle_error(e, {
                "action": action,
                "context": "respond_to_chat"
            })
            
            # Send error message to chat
            try:
                await self.api_client.send_chat_message(
                    workspace_id=action.workspace.id,
                    agent_id=action.me.id,
                    message=f"I'm sorry, I encountered an error: {str(e)}"
                )
            except Exception as send_error:
                logger.error(f"Failed to send error message: {str(send_error)}")

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

    async def mark_task_as_errored(self, workspace_id: int, task_id: int, error: str) -> None:
        """
        Mark a task as errored.
        
        This method updates the task status to ERROR and sets the error message.
        It handles 404 errors gracefully, as the task might have been deleted or moved.
        
        Args:
            workspace_id: ID of the workspace containing the task
            task_id: ID of the task to mark as errored
            error: Error message to set
            
        Raises:
            APIError: If the API request fails (except for 404 errors)
        """
        try:
            await self._api_client.mark_task_as_errored(
                workspace_id=workspace_id,
                task_id=task_id,
                error=error
            )
            logger.info(f"Task {task_id} marked as errored")
        except Exception as e:
            # If it's a 404 error, log it but don't raise
            if "404 Not Found" in str(e):
                logger.warning(f"Failed to mark task as errored: {str(e)}")
            else:
                logger.error(f"Failed to mark task as errored: {str(e)}")
                self.handle_error(e, {
                    "workspace_id": workspace_id,
                    "task_id": task_id,
                    "error": error
                })

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

    async def update_task_status(self, params: UpdateTaskStatusParams) -> Optional[Dict[str, Any]]:
        """
        Update the status of a task.
        
        This method updates the task status and handles errors gracefully.
        
        Args:
            params: UpdateTaskStatusParams object containing:
                - workspace_id: ID of the workspace containing the task
                - task_id: ID of the task to update
                - status: New status for the task
                
        Returns:
            Response from the API or None if the request fails
            
        Raises:
            No exceptions are raised, errors are logged
        """
        try:
            response = await self._api_client.update_task_status(params)
            logger.info(f"Task {params.task_id} status updated to {params.status}")
            return response
        except Exception as e:
            logger.error(f"Failed to update task status: {str(e)}")
            self.handle_error(e, {"params": params.model_dump()})
            return None

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
