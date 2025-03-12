# OpenServ Python SDK

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

The official Python SDK for building AI agents with OpenServ. Create powerful AI agents with custom capabilities, secure authentication, and real-time communication.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Usage Guide](#usage-guide)
  - [Basic Usage](#basic-usage)
  - [Advanced Features](#advanced-features)
- [Examples](#examples)
- [Development](#development)
- [Support](#support)

## Installation

```bash
# Install directly from GitHub
pip install git+https://github.com/openserv-labs/python-sdk.git

# For development installation with test dependencies
git clone https://github.com/openserv-labs/python-sdk.git
cd python-sdk
pip install -e ".[test]"
```

## Quick Start

1. Get your API key from [OpenServ Platform](https://openserv.ai)

2. Set up environment variables:
```bash
export OPENSERV_API_KEY=your-api-key
```

3. Create your first agent:
```python
from openserv_sdk import Agent, AgentOptions
from pydantic import BaseModel, Field

# Define your capability parameters
class GreetingParams(BaseModel):
    name: str = Field(..., description="Name to greet")
    language: str = Field(default="en", description="Language code")

# Initialize the agent
agent = Agent(AgentOptions(
    api_key=os.getenv("OPENSERV_API_KEY"),
    name="Greeting Agent",
    description="A simple agent that greets users"
))

# Add a capability
@agent.capability("greet")
async def greet(params: dict, messages: list) -> str:
    name = params["args"]["name"]
    language = params["args"].get("language", "en")
    return f"Hello, {name}!" if language == "en" else f"¡Hola, {name}!"

# Run the agent
if __name__ == "__main__":
    import asyncio
    asyncio.run(agent.start())
```

## Features

- 🚀 **Easy Integration**: Simple API for building and deploying AI agents
- 🔒 **Security First**: Built-in authentication and secure communication
- 🛠 **Extensible**: Custom capabilities with Pydantic validation
- 📝 **Type Safety**: Full type hints and runtime validation
- 🔄 **Modern Python**: Async/await support with Python 3.8+
- 🌐 **Flexible Transport**: HTTP and WebSocket support
- 🧩 **Modular Design**: Easy to extend and customize
- 📊 **Observability**: Built-in logging and error handling

## Usage Guide

### Basic Usage

1. **Creating an Agent**
```python
agent = Agent(AgentOptions(
    api_key="your-api-key",
    name="My Agent",
    description="Agent description"
))
```

2. **Adding Capabilities**
```python
from pydantic import BaseModel, Field

class MathParams(BaseModel):
    x: float = Field(..., description="First number")
    y: float = Field(..., description="Second number")

@agent.capability("add")
async def add_numbers(params: dict, messages: list) -> dict:
    args = params["args"]
    result = args["x"] + args["y"]
    return {"result": result}
```

3. **Error Handling**
```python
@agent.capability("divide")
async def divide(params: dict, messages: list) -> dict:
    try:
        x, y = params["args"]["x"], params["args"]["y"]
        if y == 0:
            raise ValueError("Division by zero")
        return {"result": x / y}
    except Exception as e:
        return {"error": str(e)}
```

### Advanced Features

1. **Custom Middleware**
```python
from fastapi import Request
from openserv_sdk.middleware import BaseMiddleware

class LoggingMiddleware(BaseMiddleware):
    async def process(self, request: Request):
        print(f"Processing request: {request.url}")
        return await super().process(request)

agent.add_middleware(LoggingMiddleware())
```

2. **WebSocket Support**
```python
@agent.websocket("realtime")
async def handle_realtime(websocket):
    while True:
        data = await websocket.receive_json()
        result = process_data(data)
        await websocket.send_json(result)
```

3. **Integration with External Services**
```python
@agent.capability("fetch_data")
async def fetch_external_data(params: dict, messages: list) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(params["args"]["url"]) as response:
            return {"data": await response.json()}
```

## Examples

Check out our [examples directory](./examples) for complete implementations:

- `basic_agent.py`: Simple agent with greeting capabilities
- `math_agent.py`: Agent performing mathematical operations
- `integration_agent.py`: Agent integrating with external services
- `websocket_agent.py`: Real-time agent using WebSocket

## Development

### Setup

1. Clone the repository:
```bash
git clone https://github.com/openserv-labs/python-sdk.git
cd python-sdk
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[test]"
```

### Testing

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/test_agent.py
pytest -m "integration"  # Run integration tests
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- 📚 [Documentation](https://docs.openserv.ai/resources/python-sdk)
- 🐛 [GitHub Issues](https://github.com/openserv-labs/python-sdk/issues)
- 📧 [Email Support](mailto:support@openserv.ai)
- 💬 [Discord Community](https://discord.gg/openserv)

## Acknowledgments

Built with ❤️ by [OpenServ Labs](https://openserv.ai)