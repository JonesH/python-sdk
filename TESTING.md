# Testing Guide

## Overview

This document describes the testing setup and procedures for the OpenServ Python SDK.

## Test Structure

Our test suite is organized as follows:

- `test_agent.py`: Tests for the main Agent class functionality
- `test_api.py`: Tests for API interactions
- `test_capability.py`: Tests for capability management
- `test_types.py`: Tests for type definitions and validation
- `test_marketing_agent.py`: Tests for the marketing agent example

## Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
```

### Configuration

Create a `pytest.ini` in the project root:

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
markers =
    asyncio: mark test as async
    integration: mark test as integration test
addopts = 
    --verbose
    --cov=src
    --cov-report=term-missing
    --cov-report=html
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src tests/

# Run specific test file
pytest tests/test_agent.py

# Run tests with specific marker
pytest -m "integration"

# Run tests and generate coverage report
pytest --cov=src --cov-report=html tests/
```

## Writing Tests

### Using Fixtures

Common test fixtures are defined in `tests/conftest.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.agent import Agent
from src.types import AgentOptions

@pytest.fixture
def mock_api_key():
    return "test-openserv-key"

@pytest.fixture
def mock_openai():
    with patch('openai.OpenAI') as mock:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='Test response',
                        role='assistant'
                    )
                )
            ]
        ))
        mock.return_value = mock_client
        yield mock

@pytest.fixture
def base_agent(mock_api_key):
    return Agent(AgentOptions(
        api_key=mock_api_key,
        system_prompt="You are a test agent"
    ))
```

### Test Examples

1. Testing Agent Initialization:
```python
def test_agent_initialization(mock_api_key):
    agent = Agent(AgentOptions(
        system_prompt="Test prompt",
        api_key=mock_api_key
    ))
    
    assert agent.config.system_prompt == "Test prompt"
    assert agent.config.api.api_key == mock_api_key
```

2. Testing Async Functions:
```python
@pytest.mark.asyncio
async def test_process_request(mock_openai, base_agent):
    result = await base_agent.process({
        "messages": [{"role": "user", "content": "Hello"}]
    })
    assert result.choices[0].message.content == "Test response"
```

3. Testing Error Handling:
```python
def test_configuration_error():
    with pytest.raises(ConfigurationError):
        Agent(AgentOptions(system_prompt="Test"))
```

## Best Practices

1. **Test Isolation**: Each test should be independent and not rely on the state of other tests.

2. **Mock External Services**: Always mock external services like OpenAI API calls:
```python
@pytest.mark.asyncio
async def test_openai_integration(mock_openai, base_agent):
    result = await base_agent.process({"messages": [{"role": "user", "content": "Test"}]})
    assert mock_openai.chat.completions.create.called
```

3. **Type Testing**: Test type validation using Pydantic models:
```python
def test_validate_input_types():
    with pytest.raises(ValidationError):
        GreetArgs(name=123)  # Should be string
```

4. **Error Handling**: Test both success and error cases:
```python
@pytest.mark.asyncio
async def test_error_handling(base_agent):
    with pytest.raises(APIError):
        await base_agent.invalid_method()
```

5. **Async Testing**: Use `pytest-asyncio` for testing async functions:
```python
@pytest.mark.asyncio
async def test_async_capability(base_agent):
    result = await base_agent.process_async_task()
    assert result is not None
```

## Coverage Requirements

- Minimum coverage requirement: 80%
- Critical paths should have 100% coverage
- Integration tests should cover main user workflows

## Troubleshooting

Common testing issues and solutions:

1. **Async Test Failures**
   - Ensure `pytest-asyncio` is installed
   - Use `@pytest.mark.asyncio` decorator
   - Check for proper async/await usage

2. **Mock Issues**
   - Verify mock setup in conftest.py
   - Check mock return values
   - Ensure proper patch paths

3. **Coverage Problems**
   - Run with `--cov-report=term-missing`
   - Check excluded paths in setup.cfg
   - Verify test discovery paths 