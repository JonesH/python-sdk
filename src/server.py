"""
FastAPI server implementation for the OpenServ Agent.
"""

import json
import logging
import os
from fastapi import FastAPI, Request, HTTPException
from typing import Optional, Dict, Any
import uvicorn
import asyncio

from .config import ServerConfig
from .exceptions import ToolError

logger = logging.getLogger(__name__)

class AgentServer:
    """HTTP server for the Agent."""
    def __init__(self, config: ServerConfig):
        self.config = config
        self.app = FastAPI()
        self._agent = None
        self._server: Optional[uvicorn.Server] = None
        
        # Set up routes
        @self.app.post("/")
        async def root(request: Request):
            """Root route for task execution and chat message responses."""
            if not self._agent:
                raise HTTPException(status_code=500, detail="Agent not initialized")
            
            body = await request.json()
            logger.debug("Request body: %s", body)
            
            await self._agent.handle_root_route(body)
            return "OK"
            
        @self.app.post("/tools/{tool_name}")
        async def tool(tool_name: str, request: Request):
            """Tool route for executing specific capabilities."""
            if not self._agent:
                raise HTTPException(status_code=500, detail="Agent not initialized")
                
            body = await request.json()
            result = await self._agent.handle_tool_route(tool_name, body)
            return result

    def set_agent(self, agent: Any) -> None:
        """Set the agent instance for request handling."""
        self._agent = agent

    async def start(self) -> None:
        """Start the HTTP server."""
        logger.info("Agent server starting on port %s", self.config.port)
        
        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level=self.config.log_level
        )
        
        self._server = uvicorn.Server(config)
        logger.info("Server configuration complete, starting server")
        
        # Start the server in a background task
        await self._server.serve()

    async def stop(self) -> None:
        """Gracefully shut down the server."""
        if self._server:
            logger.info("Shutting down server...")
            self._server.should_exit = True
            await self._server.shutdown()
            self._server = None 