"""
OpenServ Python SDK

A framework for building non-deterministic AI agents with advanced cognitive capabilities
like reasoning, decision-making, and inter-agent collaboration within the OpenServ platform.
"""

from .agent import Agent, AgentOptions
from .capability import Capability
from .logger import create_logger, logger
from .schema_types import (
    AgentKind, TaskStatus, Action, DoTaskActionSchema, RespondChatMessageActionSchema,
    GetFilesParams, GetSecretsParams, GetSecretValueParams, UploadFileParams,
    MarkTaskAsErroredParams, CompleteTaskParams, SendChatMessageParams,
    GetTaskDetailParams, GetAgentsParams, GetTasksParams, CreateTaskParams,
    AddLogToTaskParams, RequestHumanAssistanceParams, UpdateTaskStatusParams,
    ProcessParams, IntegrationCallRequest, ChatCompletionMessageParam,
    ProxyConfiguration
)

__all__ = [
    # Core classes
    'Agent',
    'AgentOptions',
    'Capability',
    
    # Logger
    'create_logger',
    'logger',
    
    # Types and schemas
    'AgentKind',
    'TaskStatus',
    'Action',
    'DoTaskActionSchema',
    'RespondChatMessageActionSchema',
    
    # API parameters
    'GetFilesParams',
    'GetSecretsParams',
    'GetSecretValueParams',
    'UploadFileParams',
    'MarkTaskAsErroredParams',
    'CompleteTaskParams',
    'SendChatMessageParams',
    'GetTaskDetailParams',
    'GetAgentsParams',
    'GetTasksParams',
    'CreateTaskParams',
    'AddLogToTaskParams',
    'RequestHumanAssistanceParams',
    'UpdateTaskStatusParams',
    'ProcessParams',
    'IntegrationCallRequest',
    'ChatCompletionMessageParam',
    'ProxyConfiguration'
]

__version__ = '0.1.0'