"""
Configuration classes for OpenServ SDK.
"""

from dataclasses import dataclass
from typing import Optional, List

@dataclass
class APIConfig:
    """API configuration."""
    api_key: str
    platform_url: str = "https://api.openserv.ai"
    runtime_url: str = "https://agents.openserv.ai"

@dataclass
class OpenAIConfig:
    """OpenAI configuration."""
    api_key: str
    model: str = "gpt-4"

@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 7378
    debug: bool = False
    version: str = "1.0.0"
    
    # Security settings
    require_https: bool = False
    trusted_hosts: Optional[List[str]] = None
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_ca_certs: Optional[str] = None
    
    # Performance settings
    workers: int = 1
    limit_concurrency: Optional[int] = None
    timeout_keep_alive: int = 5
    
    # Rate limiting
    rate_limit: bool = True
    rate_limit_per_second: int = 10
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.require_https and (not self.ssl_certfile or not self.ssl_keyfile):
            raise ValueError("SSL certificate and key files are required when HTTPS is enabled")
        
        if self.workers < 1:
            raise ValueError("Number of workers must be at least 1")
        
        if self.timeout_keep_alive < 0:
            raise ValueError("Timeout keep alive must be non-negative")

@dataclass
class Config:
    """Main configuration class."""
    api: APIConfig
    openai: Optional[OpenAIConfig] = None
    system_prompt: Optional[str] = None
    port: int = 7378
    host: str = "0.0.0.0"
    log_level: str = "info"
    reload: bool = False
    on_error: Optional[callable] = None
    
    def validate_api_key(self):
        """Validate that API key is present."""
        if not self.api.api_key:
            raise ValueError(
                "OpenServ API key is required. Please provide it in options or set OPENSERV_API_KEY environment variable."
            )

    @classmethod
    def from_env(cls, system_prompt: str) -> 'Config':
        """Create a configuration instance from environment variables."""
        return cls(system_prompt=system_prompt) 
