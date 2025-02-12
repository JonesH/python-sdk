"""
Configuration management for the OpenServ Agent library.
"""

import os
from typing import Optional, Callable, Dict, Any
from pydantic import BaseModel, Field

class APIConfig(BaseModel):
    """API configuration for OpenServ clients."""
    api_key: str
    platform_url: str = 'https://api.openserv.ai'
    runtime_url: str = 'https://agents.openserv.ai'

class OpenAIConfig(BaseModel):
    """Configuration for OpenAI integration."""
    api_key: str
    model: str = 'gpt-4'

class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = '0.0.0.0'
    port: int = 7378

class Config(BaseModel):
    """Main configuration class."""
    api: APIConfig
    openai: OpenAIConfig
    system_prompt: str
    port: int = 7378
    host: str = '0.0.0.0'
    log_level: str = 'debug'
    reload: bool = False
    on_error: Optional[Callable[[Exception, Dict[str, Any]], None]] = None
    
    def validate_api_key(self):
        """Validate that API key is present."""
        if not self.api.api_key:
            raise ValueError('OpenServ API key is required')

    @classmethod
    def from_env(cls, system_prompt: str) -> 'Config':
        """Create a configuration instance from environment variables."""
        return cls(system_prompt=system_prompt) 
