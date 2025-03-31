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
            }
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
    """Client for the OpenServ Platform API."""
    def __init__(self, config: APIConfig):
        super().__init__(config)
        self.client.base_url = config.platform_url
    
    async def get_files(self, workspace_id: int) -> Dict[str, Any]:
        """Get files from a workspace."""
        return await self._request('GET', f'/workspaces/{workspace_id}/files')
    
    async def upload_file(
        self,
        workspace_id: int,
        path: str,
        file_content: Any,
        task_ids: Optional[list[int]] = None,
        skip_summarizer: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Upload a file to a workspace."""
        files = {'file': ('file', file_content)}
        data = {
            'path': path,
            'taskIds': str(task_ids) if task_ids else None,
            'skipSummarizer': str(skip_summarizer) if skip_summarizer is not None else None
        }
        return await self._request(
            'POST',
            f'/workspaces/{workspace_id}/file',
            files=files,
            json=data
        )

class RuntimeClient(BaseClient):
    """Client for the OpenServ Runtime API."""
    def __init__(self, config: APIConfig):
        super().__init__(config)
        # Ensure runtime_url doesn't end with a slash
        runtime_url = config.runtime_url.rstrip('/')
        self.client = httpx.AsyncClient(
            base_url=f"{runtime_url}/runtime",
            headers={
                'Content-Type': 'application/json',
                'x-openserv-key': config.api_key
            },
            timeout=300.0
        )
    
    async def execute_task(
        self,
        workspace_id: int,
        task_id: int,
        tools: list[Dict[str, Any]],
        messages: list[Dict[str, Any]],
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a task through the runtime."""
        return await self._request(
            'POST',
            '/execute',
            json_data={
                'workspace_id': workspace_id,
                'task_id': task_id,
                'tools': tools,
                'messages': messages,
                'action': action
            }
        )
    
    async def handle_chat(
        self,
        tools: List[Dict[str, Any]],
        messages: List[Dict[str, str]],
        action: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Handle a chat request."""
        return await self._request(
            "POST",
            "/chat",
            json_data={
                "tools": tools,
                "messages": messages,
                "action": action,
            },
        ) 
