import os
import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any
from dotenv import load_dotenv

# Add the src directory to Python path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path) 

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

# Load environment variables from .env
load_dotenv()

@pytest.fixture
def mock_env_keys():
    """Mock environment variables for testing."""
    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-openai-key',
        'OPENSERV_API_KEY': 'test-openserv-key',
        'PORT': '7378'
    }):
        yield

@pytest.fixture
def mock_openai():
    """Mock AsyncOpenAI client for testing."""
    with patch('openai.AsyncOpenAI') as mock:
        mock_client = AsyncMock()
        mock_completion = AsyncMock()
        mock_completion.choices = [
            AsyncMock(message=AsyncMock(
                content="Test response",
                tool_calls=None
            ))
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        mock.return_value = mock_client
        yield mock

@pytest.fixture(autouse=True)
def mock_environment(mock_env_keys):
    """Automatically mock environment for all tests."""
    yield

@pytest.fixture
def mock_agent():
    """Mock Agent for testing."""
    mock = MagicMock()
    mock.process.return_value = {"response": "Test response"}
    return mock 

@pytest.fixture
def mock_agent_response() -> Dict[str, Any]:
    return {
        "id": 1,
        "name": "test-agent",
        "kind": "EXTERNAL",
        "capabilities_description": "test capabilities",
        "system_prompt": "You are a test agent",
        "is_built_by_agent_builder": False
    }

@pytest.fixture
def mock_task_response() -> Dict[str, Any]:
    return {
        "id": 1,
        "description": "test task",
        "body": "test body",
        "expected_output": "test output",
        "input": "test input",
        "dependencies": [],
        "human_assistance_requests": []
    }

@pytest.fixture
def mock_workspace_response() -> Dict[str, Any]:
    return {
        "id": 1,
        "goal": "test goal",
        "bucket_folder": "test-folder",
        "agents": [
            {
                "id": 2,
                "name": "test agent",
                "kind": "EXTERNAL",
                "capabilities_description": "test capabilities"
            }
        ]
    } 