"""
API client implementations for OpenServ and Runtime services.
"""

import httpx
from typing import Any, Dict, Optional, List, Union, BinaryIO
from .config import APIConfig
from .exceptions import APIError, AuthenticationError
from .types import UpdateTaskStatusParams, TaskStatus
import logging
import json
from datetime import datetime
import aiohttp
import os

# Configure logger
logger = logging.getLogger(__name__)

# Configure logging to show INFO and above
logging.basicConfig(level=logging.INFO)

# Set httpx logger to debug level
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class BaseClient:
    """Base class for API clients."""
    def __init__(self, config: APIConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            headers={'x-openserv-key': config.api_key},
            verify=False if config.platform_url.startswith('https://') else True
        )
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
        form_data: Optional[aiohttp.FormData] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Make an HTTP request and handle common error cases.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path
            json_data: Optional JSON data for request body
            params: Optional query parameters
            form_data: Optional form data for multipart requests
            headers: Optional additional headers
            
        Returns:
            Response data as dictionary or None
            
        Raises:
            APIError: For API-related errors
            AuthenticationError: For authentication failures
        """
        try:
            # Start with base headers
            request_headers = {'x-openserv-key': self.config.api_key}
            if headers:
                request_headers.update(headers)

            # Prepare request data
            content = None
            files = None
            
            if json_data is not None:
                content = json.dumps(json_data, cls=DateTimeEncoder).encode('utf-8')
                request_headers['Content-Type'] = 'application/json'
            elif form_data is not None:
                # Convert aiohttp FormData to httpx files format
                files = {}
                for field_name, field_value in form_data._fields:
                    if isinstance(field_value[0], bytes):
                        content_type = field_value[3].get('content-type', 'application/octet-stream') if len(field_value) > 3 else 'application/octet-stream'
                        files[field_name] = (field_value[2], field_value[0], content_type)
                    else:
                        files[field_name] = (None, str(field_value[0]))

            # Construct full URL
            url = self.config.platform_url + path if not path.startswith('http') else path
            
            # Make the request
            response = await self.client.request(
                method,
                url,
                content=content,
                params=params,
                headers=request_headers,
                files=files
            )
            
            # Handle response
            response.raise_for_status()
            
            # Check if response is JSON
            content_type = response.headers.get('content-type', '')
            if 'application/json' in content_type:
                return response.json()
            elif 'text/' in content_type:
                return {'data': response.text}
            else:
                return {'data': response.content}
                
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            try:
                error_data = e.response.json()
                error_message = error_data.get('error', str(e))
            except:
                error_message = str(e)
                error_data = None
                
            if status_code == 401:
                raise AuthenticationError(error_message, status_code, error_data)
            else:
                raise APIError(error_message, status_code, error_data)
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {str(e)}")
        except Exception as e:
            raise APIError(f"Unexpected error: {str(e)}")

class OpenServClient(BaseClient):
    """Client for interacting with the OpenServ API."""
    
    async def get(self, path: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make a GET request to the API."""
        return await self._request("GET", path, params=params)
    
    async def post(self, path: str, data: Optional[Dict[str, Any]] = None, form_data: Optional[aiohttp.FormData] = None) -> Dict[str, Any]:
        """Make a POST request to the API."""
        if form_data:
            return await self._request("POST", path, form_data=form_data)
        return await self._request("POST", path, json_data=data)
    
    async def put(self, path: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a PUT request to the API."""
        return await self._request("PUT", path, json_data=data)
    
    async def delete(self, path: str) -> Dict[str, Any]:
        """Make a DELETE request to the API."""
        return await self._request("DELETE", path)
    
    async def send_chat_message(self, workspace_id: int, agent_id: int, message: str) -> Dict[str, Any]:
        """Send a chat message."""
        return await self.post(
            f"/workspaces/{workspace_id}/agent-chat/{agent_id}/message",
            {"message": message}
        )

    async def update_task_status(self, params: UpdateTaskStatusParams) -> Dict[str, Any]:
        """Update a task's status."""
        try:
            response = await self._request('PUT', f"/workspaces/{params.workspace_id}/tasks/{params.task_id}/status", json_data={"status": params.status.value if isinstance(params.status, TaskStatus) else params.status})
            return response["data"]
        except Exception as e:
            logger.error(f"Failed to update task status: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def mark_task_as_errored(self, workspace_id: int, task_id: int, error: str) -> Dict[str, Any]:
        """Mark a task as errored with the given error message."""
        try:
            response = await self._request('PUT', f"/workspaces/{workspace_id}/tasks/{task_id}/error", json_data={"error": error})
            return response
        except Exception as e:
            logger.error(f"Failed to mark task as errored: {str(e)}")
            return {"status": "error", "error": str(e)}

class RuntimeClient(BaseClient):
    """Client for interacting with the OpenServ Runtime API."""
    
    async def get(self, path: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make a GET request to the runtime API."""
        url = f"{self.config.runtime_url}{path}"
        return await self._request("GET", url, params=params)
    
    async def post(self, path: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a POST request to the runtime API."""
        url = f"{self.config.runtime_url}{path}"
        return await self._request("POST", url, json_data=data)
    
    async def execute_task(
        self,
        workspace_id: int,
        task_id: int,
        tools: List[Dict[str, Any]],
        messages: List[Dict[str, str]],
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a task on the runtime."""
        try:
            # First try the new endpoint format
            url = "/runtime/execute"
            payload = {
                "workspaceId": workspace_id,
                "taskId": task_id,
                "tools": tools,
                "messages": messages,
                "action": action
            }
            logger.info(f"Executing task with payload: {json.dumps(payload, indent=2)}")
            return await self.post(url, payload)
        except Exception as e:
            logger.error(f"Failed to execute task with new endpoint format: {str(e)}")
            # Fall back to the old endpoint format
            try:
                url = f"/workspaces/{workspace_id}/tasks/{task_id}/execute"
                payload = {
                    "tools": tools,
                    "messages": messages,
                    "action": action
                }
                logger.info(f"Falling back to old endpoint format: {url}")
                logger.info(f"Payload: {json.dumps(payload, indent=2)}")
                return await self.post(url, payload)
            except Exception as fallback_error:
                logger.error(f"Failed to execute task with fallback endpoint: {str(fallback_error)}")
                raise APIError(f"Failed to execute task: {str(e)}")

    async def handle_chat(
        self,
        tools: list,
        messages: list,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle a chat message."""
        url = f"{self.config.runtime_url}/runtime/chat"

        # Add required fields to action if not present
        if isinstance(action, dict):
            if 'workspaceId' not in action and 'workspace' in action:
                action['workspaceId'] = action['workspace'].get('id')
            if 'taskId' not in action and 'task' in action:
                action['taskId'] = action['task'].get('id')

        payload = {
            'workspaceId': action.get('workspaceId'),
            'taskId': action.get('taskId'),
            'tools': tools,
            'messages': messages,
            'action': action
        }

        # Log the exact payload being sent
        logger.info(f"Sending chat request with payload: {json.dumps(payload, indent=2)}")

        try:
            response = await self._request('POST', url, json_data=payload)
            logger.info(f"Chat response: {json.dumps(response, indent=2) if response else 'None'}")
            return response
        except Exception as e:
            logger.error(f"Chat request failed with error: {str(e)}")
            if isinstance(e, httpx.HTTPStatusError):
                logger.error(f"Response content: {e.response.content}")
            raise 
