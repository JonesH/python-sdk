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
        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "up", "version": "1.0.0"}
        
        @self.app.post("/")
        async def root(request: Request):
            """Root route for task execution and chat message responses."""
            if not self._agent:
                raise HTTPException(status_code=500, detail="Agent not initialized")
            
            try:
                body = await request.json()
                logger.debug("Request body: %s", body)
                
                await self._agent.handle_root_route(body)
                return {"status": "OK", "message": "Request accepted for processing"}
            except Exception as e:
                logger.exception("Error handling root request: %s", str(e))
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing request: {str(e)}"
                )
            
        @self.app.post("/tools/{tool_name}")
        async def tool(tool_name: str, request: Request):
            """Tool route for executing specific capabilities."""
            if not self._agent:
                raise HTTPException(status_code=500, detail="Agent not initialized")
                
            try:
                body = await request.json()
                logger.debug("Tool request for %s: %s", tool_name, body)
                
                # Ensure body contains necessary parameters
                if 'args' not in body:
                    body['args'] = {}
                if 'messages' not in body:
                    body['messages'] = []
                
                result = await self._agent.handle_tool_route(tool_name, body)
                return result
            except Exception as e:
                logger.exception("Error handling tool request for %s: %s", tool_name, str(e))
                raise HTTPException(
                    status_code=500, 
                    detail=f"Error executing tool {tool_name}: {str(e)}"
                )
        
        @self.app.post("/task-complete")
        async def task_complete(request: Request):
            """Endpoint to explicitly mark a task as complete."""
            if not self._agent:
                raise HTTPException(status_code=500, detail="Agent not initialized")
                
            try:
                body = await request.json()
                logger.info(f"Task completion request: {body}")
                
                workspace_id = body.get('workspace_id')
                task_id = body.get('task_id')
                output = body.get('output', 'Task completed by agent')
                
                if not workspace_id or not task_id:
                    raise HTTPException(
                        status_code=400,
                        detail="Missing required parameters: workspace_id and task_id"
                    )
                
                await self._agent.complete_task(workspace_id, task_id, output)
                return {"status": "success", "message": f"Task {task_id} marked as complete"}
            except Exception as e:
                logger.exception("Error completing task: %s", str(e))
                raise HTTPException(
                    status_code=500,
                    detail=f"Error completing task: {str(e)}"
                )

    def set_agent(self, agent: Any) -> None:
        """Set the agent instance for request handling."""
        self._agent = agent

    def start(self) -> None:
        """Start the HTTP server."""
        logger.info("Agent server starting on port %s", self.config.port)
        
        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )
        
        self._server = uvicorn.Server(config)
        logger.info("Server configuration complete, starting server")
        
        try:
            # Run the server
            self._server.run()
        except Exception as e:
            logger.error("Server error: %s", e)
            raise

    async def shutdown(self) -> None:
        """Gracefully shut down the server."""
        if self._server:
            logger.info("Shutting down server...")
            self._server.should_exit = True
            try:
                await self._server.shutdown()
            except Exception as e:
                logger.error("Error during server shutdown: %s", e) 
