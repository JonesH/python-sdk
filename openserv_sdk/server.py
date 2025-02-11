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
from .logger import logger

logger = logging.getLogger(__name__)

class AgentServer:
    """Server implementation for handling agent requests."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.app = FastAPI()
        self.agent = None
        
        @self.app.post("/")
        async def handle_root(request: Request):
            try:
                body = await request.json()
                if self.agent:
                    await self.agent.handle_root_route(body)
                    return {"status": "OK"}
                else:
                    raise HTTPException(status_code=500, detail="Agent not initialized")
            except Exception as e:
                logger.error("Error handling request: %s", str(e), exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
            
        @self.app.get("/health")
        async def health_check():
            return {"status": "ok"}

        @self.app.post("/tools/{tool_name}")
        async def handle_tool(tool_name: str, request: Request):
            try:
                if not self.agent:
                    raise HTTPException(status_code=500, detail="Agent not initialized")
                
                body = await request.json()
                logger.debug(f"Tool request received for {tool_name}: {body}")
                
                result = await self.agent.handle_tool_route(tool_name, body)
                return {"result": result}
            except Exception as e:
                logger.error(f"Error handling tool request: {str(e)}", exc_info=True)
                if isinstance(e, ToolError):
                    raise HTTPException(status_code=400, detail=str(e))
                raise HTTPException(status_code=500, detail=str(e))
    
    def set_agent(self, agent):
        """Set the agent instance for handling requests."""
        self.agent = agent
        
    async def start(self):
        """Start the server."""
        logger.info("Agent server starting on port %d", self.config.port)
        logger.info("Server configuration complete, starting server")
        
        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    async def stop(self):
        """Stop the server."""
        # Uvicorn handles shutdown automatically 