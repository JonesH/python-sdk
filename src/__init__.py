"""
OpenServ Agent library.
"""

import logging

from .types import AgentOptions
from .agent import Agent
from .capability import Capability
from .exceptions import (
    OpenServError,
    ConfigurationError,
    APIError,
    AuthenticationError,
    ToolError,
    ValidationError,
    RuntimeError
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

__version__ = '0.1.0' 
