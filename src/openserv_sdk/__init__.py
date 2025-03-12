"""
OpenServ SDK core components.
"""

import logging
import os

# Configure logging once for the entire package
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Silence noisy loggers
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

from openserv_sdk.types import AgentOptions
from openserv_sdk.agent import Agent
from openserv_sdk.capability import Capability
from openserv_sdk.exceptions import (
    OpenServError,
    ConfigurationError,
    APIError,
    AuthenticationError,
    ToolError,
    ValidationError,
    RuntimeError
)

__version__ = '0.3.0'

__all__ = [
    'Agent',
    'AgentOptions',
    'Capability',
    'OpenServError',
    'ConfigurationError',
    'APIError',
    'AuthenticationError',
    'ToolError',
    'ValidationError',
    'RuntimeError'
] 
