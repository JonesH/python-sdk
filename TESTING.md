# Testing Guide

## Overview

This document describes the testing setup and procedures for the OpenServ Python SDK.

## Test Structure

Our test suite is organized as follows:

```
tests/
├── unit/
│   ├── test_agent_unit.py     # Unit tests for Agent class
│   ├── test_capability.py     # Tests for capability management
│   └── test_types.py         # Tests for type definitions and validation
├── e2e/
│   ├── test_agent_e2e.py     # End-to-end tests for Agent functionality
│   └── test_integration_types.py  # Integration tests for type handling
├── conftest.py              # Shared test fixtures and configurations
└── test_capabilities.py     # Legacy capability tests (to be removed)
```

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

# Install the package in editable mode with test dependencies
pip install -e ".[test]"
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run unit tests only
python -m pytest tests/unit/ -v

# Run e2e tests only
python -m pytest tests/e2e/ -v

# Run with coverage report
python -m pytest --cov=openserv_sdk --cov-report=term-missing tests/

# Run a specific test file
python -m pytest tests/unit/test_types.py -v

# Run tests matching a specific name pattern
python -m pytest -k "test_file_operations" -v
```

## Test Configuration

The test suite uses the following configuration files:

- `pyproject.toml`: Contains pytest configuration and test dependencies
- `.env.test`: Contains test environment variables
- `conftest.py`: Contains shared fixtures and test utilities

### Environment Variables

Required test environment variables (defined in `.env.test`):

```
OPENSERV_API_KEY=test-openserv-key
OPENAI_API_KEY=test-openai-key
PORT=7378
```

## Test Categories

### Unit Tests

Located in `tests/unit/`, these tests cover:
- Agent initialization and configuration
- Type validation and model behavior
- Capability management
- Error handling and validation

### End-to-End Tests

Located in `tests/e2e/`, these tests cover:
- File operations (upload, download, listing)
- Task management (creation, updates, status changes)
- Chat message handling
- Integration with external services
- Human assistance workflows

## Writing Tests

### Using Fixtures

Common test fixtures are available in `conftest.py`:

```python
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
```

### Test Examples

1. Testing Agent Initialization:
```python
def test_agent_initialization():
    agent = Agent(AgentOptions(
        system_prompt="Test prompt",
        api_key="test-key",
        openai_api_key="test-openai-key"
    ))
    assert agent.config.system_prompt == "Test prompt"
```

2. Testing Async Operations:
```python
@pytest.mark.asyncio
async def test_file_operations():
    agent = Agent(AgentOptions(...))
    files = await agent.get_files(GetFilesParams(workspace_id=1))
    assert isinstance(files, list)
```

3. Testing Error Handling:
```python
def test_invalid_capability():
    with pytest.raises(ValueError):
        agent.add_capability(None)
```

## Best Practices

1. **Mock External Services**: Always mock external API calls
2. **Use Fixtures**: Leverage shared fixtures for common setup
3. **Type Testing**: Include validation tests for all models
4. **Error Cases**: Test both success and error scenarios
5. **Async Testing**: Use `pytest.mark.asyncio` for async tests
6. **Isolation**: Each test should be independent

## Coverage Requirements

- Minimum coverage: 80%
- Critical paths: 100% coverage
- Run coverage reports regularly with `pytest --cov`

## Troubleshooting

Common issues and solutions:

1. **Missing Dependencies**
   ```bash
   pip install -e ".[test]"  # Installs test dependencies
   ```

2. **Async Test Failures**
   - Ensure `pytest-asyncio` is installed
   - Use `@pytest.mark.asyncio` decorator
   - Check for proper async/await usage

3. **Environment Variables**
   - Verify `.env.test` is present
   - Check environment variable values
   - Use `mock_env_keys` fixture

4. **Mock Issues**
   - Verify mock setup in conftest.py
   - Check mock return values
   - Ensure proper patch paths 