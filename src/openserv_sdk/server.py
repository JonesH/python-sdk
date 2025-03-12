"""
FastAPI server implementation for the OpenServ Agent.
"""

import json
import logging
import os
from fastapi import FastAPI, Request, HTTPException, Response, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import APIKeyHeader
from typing import Optional, Dict, Any, Callable
import uvicorn
import asyncio
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from datetime import datetime

from .config import ServerConfig
from .exceptions import ToolError, AuthenticationError
from .logger import logger

logger = logging.getLogger(__name__)

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware to add security headers."""
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Process the request and add security headers to the response."""
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

class AgentServer:
    """Server implementation for handling agent requests."""
    
    def __init__(self, config: ServerConfig):
        """
        Initialize the server with configuration.
        
        Args:
            config: Server configuration object
        """
        self.config = config
        self.app = FastAPI(
            title="OpenServ Agent", 
            docs_url="/docs" if config.debug else None, 
            redoc_url="/redoc" if config.debug else None,
            version=config.version
        )
        self.agent = None
        self.server = None
        self.server_task = None
        
        # Add middleware
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        self.app.add_middleware(SecurityMiddleware)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        if self.config.trusted_hosts:
            self.app.add_middleware(
                TrustedHostMiddleware, 
                allowed_hosts=self.config.trusted_hosts
            )
            
        if self.config.require_https:
            self.app.add_middleware(HTTPSRedirectMiddleware)
        
        # Error handler
        @self.app.exception_handler(Exception)
        async def generic_exception_handler(request: Request, exc: Exception):
            """Handle all unhandled exceptions."""
            logger.error("Unhandled exception", exc_info=exc)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "detail": str(exc) if self.config.debug else None
                }
            )
        
        # Root route handler
        @self.app.post("/")
        async def handle_root(request: Request) -> Response:
            """Handle root route for agent actions."""
            try:
                body = await request.json()
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing request body: {str(e)}", exc_info=True)
                return JSONResponse(
                    status_code=422,
                    content={"error": "Invalid JSON payload"}
                )

            try:
                if self.agent:
                    result = await self.agent.handle_root_route(body)
                    return JSONResponse(content=result)
                else:
                    raise HTTPException(status_code=500, detail="Agent not initialized")
            except AuthenticationError as e:
                logger.error("Authentication error: %s", str(e))
                return JSONResponse(
                    status_code=401,
                    content={"error": str(e)}
                )
            except Exception as e:
                logger.error("Error handling request: %s", str(e), exc_info=True)
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)}
                )
            
        # Health check endpoint
        @self.app.get("/health")
        def health_check():
            """Health check endpoint."""
            return {"status": "ok", "version": self.config.version}

        # Tool route handler
        @self.app.post("/tools/{tool_name}")
        async def handle_tool(tool_name: str, request: Request):
            """Handle tool execution requests."""
            try:
                if not self.agent:
                    raise HTTPException(status_code=500, detail="Agent not initialized")
                
                body = await request.json()
                result = await self.agent.handle_tool_route(tool_name, body)
                return result
            except ToolError as e:
                logger.error(f"Tool error: {str(e)}")
                return JSONResponse(
                    status_code=400,
                    content={"error": str(e), "tool": e.tool_name}
                )
            except HTTPException as e:
                logger.error(f"HTTP error: {e.status_code}: {e.detail}")
                raise
            except Exception as e:
                logger.error(f"Error handling tool request: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
    
    def set_agent(self, agent):
        """
        Set the agent instance for handling requests.
        
        Args:
            agent: The Agent instance to use for handling requests
        """
        self.agent = agent
        logger.info(f"Agent set for server: {agent.__class__.__name__}")
        
    async def start(self):
        """Start the server."""
        config = uvicorn.Config(
            app=self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
            reload=False,
            workers=self.config.workers,
            limit_concurrency=self.config.limit_concurrency,
            timeout_keep_alive=self.config.timeout_keep_alive,
            ssl_keyfile=self.config.ssl_keyfile,
            ssl_certfile=self.config.ssl_certfile,
            ssl_ca_certs=self.config.ssl_ca_certs
        )
        
        server = uvicorn.Server(config)
        self.server = server
        
        # Start the server in a separate task
        self.server_task = asyncio.create_task(server.serve())
        
        logger.info(f"Server started on http://{self.config.host}:{self.config.port}")
        
    async def stop(self):
        """Stop the server."""
        if hasattr(self, 'server') and self.server:
            self.server.should_exit = True
            await self.server.shutdown()
            if hasattr(self, 'server_task') and self.server_task:
                self.server_task.cancel()
                try:
                    await self.server_task
                except asyncio.CancelledError:
                    pass
            logger.info("Server stopped") 