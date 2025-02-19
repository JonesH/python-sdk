"""
API client implementations for OpenServ and Runtime services.
"""

import httpx
from typing import Any, Dict, Optional, List, Union, BinaryIO
from .config import APIConfig
from .exceptions import APIError, AuthenticationError
import logging
import json
from datetime import datetime
import aiohttp

# Configure logging to show INFO and above
logging.basicConfig(level=logging.INFO)

# Set httpx logger to debug level
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

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
            request_headers = {}
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

            response = await self.client.request(
                method,
                path,
                content=content,
                params=params,
                headers=request_headers,
                files=files
            )
            
            logger.info("Response status: %d", response.status_code)
            logger.debug("Response headers: %s", response.headers)
            
            response.raise_for_status()
            
            # Handle different content types
            content_type = response.headers.get('content-type', '')
            if 'application/json' in content_type:
                return response.json() if response.content else None
            elif 'text/html' in content_type or 'text/plain' in content_type:
                return {'status': response.text}
            else:
                # For binary content, return raw bytes
                return response.content
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            
            # Try to get error details from response
            error_details = None
            try:
                if e.response.content:
                    error_details = e.response.json()
            except json.JSONDecodeError:
                # If response is not JSON, use text content
                error_details = {'error': e.response.text} if e.response.text else None
                
            raise APIError(
                str(e),
                status_code=e.response.status_code,
                response=error_details
            )
        except httpx.RequestError as e:
            raise APIError(f"Request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise APIError(f"Invalid JSON response: {str(e)}")

class OpenServClient(BaseClient):
    """Client for making requests to the OpenServ API."""
    
    def __init__(self, config: APIConfig):
        super().__init__(config)
        
    async def get(self, path: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make a GET request."""
        url = f"{self.config.platform_url}{path}"
        return await self._request('GET', url, params=params)
        
    async def post(
        self,
        path: str,
        data: Union[Dict[str, Any], aiohttp.FormData],
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make a POST request.
        
        Args:
            path: The API endpoint path
            data: The request data, either as a dict for JSON or FormData for multipart
            headers: Optional custom headers to include in the request
            
        Returns:
            Response data as dictionary
        """
        url = f"{self.config.platform_url}{path}"
        return await self._request(
            'POST',
            url,
            json_data=data if not isinstance(data, aiohttp.FormData) else None,
            form_data=data if isinstance(data, aiohttp.FormData) else None,
            headers=headers
        )
        
    async def put(self, path: str, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make a PUT request."""
        url = f"{self.config.platform_url}{path}"
        return await self._request('PUT', url, json_data=data, headers=headers)
        
    async def delete(self, path: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Make a DELETE request."""
        url = f"{self.config.platform_url}{path}"
        return await self._request('DELETE', url, headers=headers)

class RuntimeClient(BaseClient):
    """Client for making requests to the OpenServ Runtime API."""
    
    def __init__(self, config: APIConfig):
        super().__init__(config)
        
    async def execute_task(
        self,
        workspace_id: int,
        task_id: int,
        tools: list,
        messages: list,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a task."""
        url = f"{self.config.runtime_url}/runtime/execute"
        payload = {
            'workspaceId': workspace_id,
            'taskId': task_id,
            'tools': tools,
            'messages': messages,
            'action': action
        }
        logger.info(f"Executing task with payload: {json.dumps(payload, indent=2)}")
        try:
            response = await self._request('POST', url, json_data=payload)
            if isinstance(response, bytes):
                return {"status": "success"}
            logger.info(f"Task execution response: {json.dumps(response, indent=2) if response else 'None'}")
            return response or {}
        except Exception as e:
            logger.error(f"Failed to execute task: {str(e)}")
            if isinstance(e, httpx.HTTPStatusError):
                logger.error(f"Response content: {e.response.content}")
            raise APIError(f"Failed to execute task: {str(e)}")
        
    async def handle_chat(
        self,
        tools: list,
        messages: list,
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle a chat message."""
        url = f"{self.config.runtime_url}/runtime/chat"
        return await self._request('POST', url, json_data={
            'tools': tools,
            'messages': messages,
            'action': action
        }) 
