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
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

class AgentServer:
    """Server implementation for handling agent requests."""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.app = FastAPI(title="OpenServ Agent", docs_url=None, redoc_url=None)
        self.agent = None
        
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
            logger.error("Unhandled exception", exc_info=exc)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "detail": str(exc) if self.config.debug else None
                }
            )
        
        @self.app.post("/")
        async def handle_root(request: Request) -> Response:
            """Handle root route."""
            try:
                body = await request.json()
            except json.JSONDecodeError as e:
                logger.error(f"Error handling request: {str(e)}", exc_info=True)
                return JSONResponse(
                    status_code=422,
                    content={"error": "Invalid JSON payload"}
                )

            try:
                if self.agent:
                    await self.agent.handle_root_route(body)
                    return JSONResponse(content={"status": "OK"})
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
            
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "ok",
                "timestamp": datetime.now().isoformat(),
                "version": self.config.version
            }

        @self.app.post("/tools/{tool_name}")
        async def handle_tool(tool_name: str, request: Request):
            """Handle tool execution requests."""
            try:
                if not self.agent:
                    raise HTTPException(status_code=500, detail="Agent not initialized")
                
                body = await request.json()
                logger.debug(f"Tool request received for {tool_name}: {body}")
                
                result = await self.agent.handle_tool_route(tool_name, body)
                return {"result": result}
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in tool request: {str(e)}")
                raise HTTPException(status_code=422, detail="Invalid JSON payload")
            except ToolError as e:
                logger.error(f"Tool error: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))
            except AuthenticationError as e:
                logger.error(f"Authentication error: {str(e)}")
                raise HTTPException(status_code=401, detail=str(e))
            except Exception as e:
                logger.error(f"Error handling tool request: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail="Internal server error" if not self.config.debug else str(e)
                )
    
    def set_agent(self, agent):
        """Set the agent instance for handling requests."""
        self.agent = agent
        
    async def start(self):
        """Start the server."""
        logger.info("Agent server starting on %s:%d", self.config.host, self.config.port)
        logger.info("Server configuration complete, starting server")
        
        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
            ssl_keyfile=self.config.ssl_keyfile,
            ssl_certfile=self.config.ssl_certfile,
            ssl_ca_certs=self.config.ssl_ca_certs,
            workers=self.config.workers,
            limit_concurrency=self.config.limit_concurrency,
            timeout_keep_alive=self.config.timeout_keep_alive
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    async def stop(self):
        """Stop the server."""
        # Uvicorn handles shutdown automatically 