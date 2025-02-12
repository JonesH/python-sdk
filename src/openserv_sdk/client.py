"""
API client implementations for OpenServ and Runtime services.
"""

import httpx
from typing import Any, Dict, Optional, List
from .config import APIConfig
from .exceptions import APIError, AuthenticationError
import logging
import json
from datetime import datetime

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
            headers={
                'Content-Type': 'application/json',
                'x-openserv-key': config.api_key
            },
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
    ) -> Optional[Dict[str, Any]]:
        """Make an HTTP request and handle common error cases."""
        try:
            # Pre-serialize JSON with our custom encoder
            content = None
            headers = {}
            if json_data is not None:
                content = json.dumps(json_data, cls=DateTimeEncoder).encode('utf-8')
                headers['Content-Type'] = 'application/json'

            response = await self.client.request(
                method,
                path,
                content=content,
                params=params,
                headers=headers,
            )
            
            logger.info("Response status: %d", response.status_code)
            logger.debug("Response headers: %s", response.headers)
            logger.debug("Response content: %s", response.content)
            
            response.raise_for_status()
            
            # Handle different content types
            content_type = response.headers.get('content-type', '')
            if 'application/json' in content_type:
                return response.json() if response.content else None
            elif 'text/html' in content_type or 'text/plain' in content_type:
                return {'status': response.text}
            else:
                return None
                
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
        
    async def get(self, path: str) -> Dict[str, Any]:
        """Make a GET request."""
        url = f"{self.config.platform_url}{path}"
        return await self._request('GET', url)
        
    async def post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a POST request."""
        url = f"{self.config.platform_url}{path}"
        return await self._request('POST', url, json_data=data)
        
    async def put(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a PUT request."""
        url = f"{self.config.platform_url}{path}"
        return await self._request('PUT', url, json_data=data)
        
    async def delete(self, path: str) -> Dict[str, Any]:
        """Make a DELETE request."""
        url = f"{self.config.platform_url}{path}"
        return await self._request('DELETE', url)

class RuntimeClient(BaseClient):
    """Client for making requests to the OpenServ Runtime API."""
    
    def __init__(self, config: APIConfig):
        super().__init__(config)
        
    async def execute_task(self, workspace_id: int, task_id: int, tools: list, messages: list, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task."""
        url = f"{self.config.runtime_url}/runtime/execute"
        return await self._request('POST', url, json_data={
            'workspaceId': workspace_id,
            'taskId': task_id,
            'tools': tools,
            'messages': messages,
            'action': action
        })
        
    async def handle_chat(self, tools: list, messages: list, action: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a chat message."""
        url = f"{self.config.runtime_url}/runtime/chat"
        return await self._request('POST', url, json_data={
            'tools': tools,
            'messages': messages,
            'action': action
        }) 
