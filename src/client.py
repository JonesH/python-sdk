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
    
    async def get(self, path: str, params: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """Make a GET request to the API."""
        return await self._request('GET', path, params=params)
        
    async def post(self, path: str, json_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Make a POST request to the API."""
        return await self._request('POST', path, json_data=json_data)
        
    async def put(self, path: str, json_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Make a PUT request to the API."""
        return await self._request('PUT', path, json_data=json_data)
    
    async def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Make an HTTP request and handle common error cases."""
        logger = logging.getLogger(__name__)
        try:
            # Pre-serialize JSON with our custom encoder
            content = None
            headers = {}
            if json_data is not None:
                content = json.dumps(json_data, cls=DateTimeEncoder).encode('utf-8')
                headers['Content-Type'] = 'application/json'
                logger.debug(f"Sending {method} request to {path} with data size: {len(content)} bytes")
            else:
                logger.debug(f"Sending {method} request to {path} without data")

            response = await self.client.request(
                method,
                path,
                content=content,
                params=params,
                headers=headers,
            )
            
            logger.info(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            logger.debug(f"Response content size: {len(response.content)} bytes")
            
            # Log the actual content for debugging, but limit length
            if len(response.content) < 1000:
                logger.debug(f"Response content: {response.content}")
            else:
                logger.debug(f"Response content (truncated): {response.content[:1000]}...")
            
            response.raise_for_status()
            
            # Handle different content types
            content_type = response.headers.get('content-type', '')
            logger.debug(f"Response content type: {content_type}")
            
            if 'application/json' in content_type:
                if response.content:
                    json_response = response.json()
                    if isinstance(json_response, dict):
                        logger.debug(f"JSON response keys: {list(json_response.keys())}")
                    return json_response
                return None
            elif 'text/html' in content_type or 'text/plain' in content_type:
                return {'status': response.text}
            else:
                # Special handling for empty or unspecified content types
                if path.endswith('/execute') and response.status_code == 200:
                    logger.info("Task execution request successful with status 200")
                    # Return a successful response even if there's no content
                    return {'success': True, 'status': 'Task execution initiated'}
                
                logger.warning(f"Unhandled content type: {content_type}")
                # Try to parse as JSON anyway if there's content
                if response.content:
                    try:
                        json_response = response.json()
                        logger.info("Successfully parsed response as JSON despite missing content-type")
                        return json_response
                    except json.JSONDecodeError:
                        logger.warning("Could not parse response as JSON")
                        return {'raw_content': response.text if response.text else 'Empty response'}
                return {'status': 'No content'}
                
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
                
            logger.error(f"HTTP error {e.response.status_code}: {error_details}")
            raise APIError(
                str(e),
                status_code=e.response.status_code,
                response=error_details
            )
        except httpx.RequestError as e:
            logger.error(f"Request error: {str(e)}")
            raise APIError(f"Request failed: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
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
        task_ids: Optional[List[int]] = None,
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
            f'/workspaces/{workspace_id}/files',
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
        tools: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a task through the runtime."""
        logger = logging.getLogger(__name__)
        
        # Log detailed info about the request
        logger.info(f"Executing task {task_id} for workspace {workspace_id}")
        logger.info(f"Tools provided: {', '.join(t.get('name', 'unknown') for t in tools)}")
        logger.info(f"Number of messages: {len(messages)}")
        
        # Create the request payload
        json_data = {
            'workspace_id': workspace_id,
            'task_id': task_id,
            'tools': tools,
            'messages': messages,
            'action': action,
            'auto_complete': True  # Signal to platform to automatically complete the task when processed
        }
        
        # Execute the request
        try:
            response = await self._request(
                'POST',
                '/execute',
                json_data=json_data
            )
            
            # Special handling for task execution responses
            if response:
                if isinstance(response, dict):
                    logger.info(f"Task execution response received: {list(response.keys())}")
                    if 'success' in response and response['success']:
                        logger.info(f"Task {task_id} successfully initiated")
                else:
                    logger.info(f"Task execution response type: {type(response)}")
            else:
                # If no response data, still consider the task successful if no exception was raised
                logger.info("No detailed response data received but execution request was successful")
                response = {'success': True, 'status': 'Task execution successfully initiated'}
            
            return response
        except Exception as error:
            logger.error(f"Task execution error: {str(error)}", exc_info=True)
            # Return a response indicating failure
            return {
                'success': False,
                'error': str(error)
            }
    
    async def handle_chat(
        self,
        tools: List[Dict[str, Any]],
        messages: List[Dict[str, str]],
        action: Dict[str, Any],
        single_use: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Handle a chat request."""
        json_data = {
            "tools": tools,
            "messages": messages,
            "action": action,
        }
        
        if single_use:
            json_data["single_use"] = True
            
        return await self._request(
            "POST",
            "/chat",
            json_data=json_data,
        ) 