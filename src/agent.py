"""
Main Agent implementation for the OpenServ Agent library.
"""

import logging
from typing import Optional, List, Dict, Any, TypeVar, Generic, Callable, Awaitable, cast, Union
import openai
import asyncio
import signal
from pydantic import BaseModel

# Configure logging to show INFO and above
logging.basicConfig(level=logging.INFO)

from .config import Config
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
        """Initialize the Agent with the given options."""
        logger.info("Initializing Agent with options: %s", options.model_dump())
        
        # Create configuration
        self.config = Config.from_env(system_prompt=options.system_prompt)
        if options.api_key:
            self.config.api.api_key = options.api_key
        if options.openai_api_key:
            self.config.openai.api_key = options.openai_api_key
        if options.port:
            self.config.server.port = options.port
            
        # Validate configuration
        self.config.validate_api_key()
        
        # Initialize components
        self.tools: List[Capability[BaseModel]] = []
        self._openai: Optional[openai.OpenAI] = None
        self.api_client = OpenServClient(self.config.api)
        self.runtime_client = RuntimeClient(self.config.api)
        
        # Set up server
        self.server = AgentServer(self.config.server)
        self.server.set_agent(self)
        
    @property
    def openai_client(self) -> openai.OpenAI:
        """Get or create the OpenAI client instance."""
        if not self._openai:
            if not self.config.openai.api_key:
                raise ConfigurationError('OpenAI API key is required')
            self._openai = openai.OpenAI(api_key=self.config.openai.api_key)
        return self._openai

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
            raise ValueError(f'Tool with name "{capability.name}" already exists')
        self.tools.append(capability)
        return self

    def add_capabilities(self, capabilities: List[Capability[T]]) -> 'Agent':
        """Add multiple capabilities to the agent."""
        for capability in capabilities:
            self.add_capability(capability)
        return self

    async def process(self, params: ProcessParams) -> Dict[str, Any]:
        """Process a conversation with OpenAI."""
        logger.info("Starting process with %d messages", len(params.messages))
        try:
            current_messages = params.messages.copy()
            max_iterations = 10
            iteration_count = 0

            while iteration_count < max_iterations:
                logger.debug("Process iteration %d/%d", iteration_count + 1, max_iterations)
                
                completion = await self.openai_client.chat.completions.create(
                    model=self.config.openai.model,
                    messages=current_messages,
                    tools=self.openai_tools if self.tools else None
                )

                if not completion.choices or not completion.choices[0].message:
                    raise RuntimeError('No response from OpenAI')

                last_message = completion.choices[0].message
                logger.debug("Received message from OpenAI: %s", last_message)

                if not last_message.tool_calls:
                    logger.info("No tool calls requested, returning completion")
                    return completion.model_dump()

                tool_results = []
                for tool_call in last_message.tool_calls:
                    if not tool_call.function:
                        raise RuntimeError('Tool call function is missing')

                    name = tool_call.function.name
                    args = tool_call.function.arguments
                    logger.info("Executing tool '%s' with args: %s", name, args)

                    try:
                        tool = next((t for t in self.tools if t.name == name), None)
                        if not tool:
                            raise RuntimeError(f'Tool "{name}" not found')

                        result = await tool.run({"args": args}, current_messages)
                        logger.debug("Tool '%s' execution result: %s", name, result)
                        tool_results.append({
                            'role': 'tool',
                            'content': str(result),
                            'tool_call_id': tool_call.id
                        })
                    except Exception as error:
                        logger.error("Tool execution failed: %s", str(error), exc_info=True)
                        tool_results.append({
                            'role': 'tool',
                            'content': str({'error': str(error)}),
                            'tool_call_id': tool_call.id
                        })

                current_messages.extend([last_message, *tool_results])
                iteration_count += 1

            raise RuntimeError('Max iterations reached without completion')
        except Exception as error:
            logger.error("Process failed: %s", str(error), exc_info=True)
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

    async def handle_tool_route(self, tool_name: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Handle execution of a specific tool/capability."""
        try:
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                raise ValueError(f'Tool "{tool_name}" not found')

            args = tool.schema.model_validate(body.get('args', {}))
            messages = body.get('messages', [])
            result = await tool.run(args, messages)
            return {'result': result}
        except Exception as error:
            logger.error("Tool route handler failed: %s", str(error), exc_info=True)
            raise

    def start(self) -> None:
        """Start the agent's HTTP server with signal handling."""
        loop = asyncio.get_event_loop()
        
        def handle_signal(sig: int) -> None:
            sig_name = signal.Signals(sig).name
            logger.info("Received %s. Starting graceful shutdown...", sig_name)
            # Schedule the shutdown coroutine
            loop.create_task(self.stop())
        
        # Set up signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))
            
        self.server.start()

    async def stop(self) -> None:
        """Stop the agent and clean up resources."""
        logger.info("Stopping server and closing clients...")
        
        try:
            await self.server.shutdown()
        except Exception as e:
            logger.error("Error during server shutdown: %s", e)
            
        try:
            await self.api_client.close()
            await self.runtime_client.close()
        except Exception as e:
            logger.error("Error during client cleanup: %s", e)

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

    async def get_tasks(self, workspace_id: int) -> Dict[str, Any]:
        """Get tasks in a workspace."""
        response = await self.api_client.get(f"/workspaces/{workspace_id}/tasks")
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

    async def request_human_assistance(self, workspace_id: int, task_id: int, type: str, question: str) -> Dict[str, Any]:
        """Request human assistance."""
        response = await self.api_client.post(f"/workspaces/{workspace_id}/tasks/{task_id}/human-assistance", {
            "type": type,
            "question": question
        })
        return response["data"]

    async def get_task_detail(self, params: GetTaskDetailParams) -> Dict[str, Any]:
        """Gets detailed information about a specific task."""
        response = await self.api_client.get(f"/workspaces/{params.workspace_id}/tasks/{params.task_id}/detail")
        return response["data"]

    async def get_agents(self, params: GetAgentsParams) -> Dict[str, Any]:
        """Gets a list of agents in a workspace."""
        response = await self.api_client.get(f"/workspaces/{params.workspace_id}/agents")
        return response["data"]

    async def get_tasks(self, params: GetTasksParams) -> Dict[str, Any]:
        """Gets a list of tasks in a workspace."""
        response = await self.api_client.get(f"/workspaces/{params.workspace_id}/tasks")
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
