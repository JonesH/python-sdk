"""
# OpenServ Python SDK

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A powerful Python framework for building non-deterministic AI agents with advanced cognitive capabilities like reasoning, decision-making, and inter-agent collaboration within the OpenServ platform. Built with strong typing, extensible architecture, and a fully autonomous agent runtime.

## Features

- 🔌 Advanced cognitive capabilities with reasoning and decision-making
- 🤝 Inter-agent collaboration and communication
- 🔌 Extensible agent architecture with custom capabilities
- 🔧 Fully autonomous agent runtime with shadow agents
- 🌐 Framework-agnostic - integrate agents from any AI framework
- ⛓️ Blockchain-agnostic - compatible with any chain implementation
- 🤖 Task execution and chat message handling
- 🔄 Asynchronous task management
- 📁 File operations and management
- 🤝 Smart human assistance integration
- 📝 Strong type hints with Pydantic validation
- 📊 Built-in logging and error handling
- 🎯 Three levels of control for different development needs

## Installation

### Option 1: Install from GitHub
```bash
pip install git+https://github.com/openserv-labs/python-sdk.git
```

### Option 2: Local Installation
```bash
# Clone the repository
git clone https://github.com/openserv-labs/python-sdk.git

# Navigate to the project directory
cd python-sdk

# Install in editable mode
pip install -e .
```

### Option 3: PyPI (Coming Soon)
```bash
pip install openserv-sdk
```

## Quick Start

Create a simple agent with a greeting capability:

```python
from openserv_sdk import Agent, AgentOptions, Capability
from pydantic import BaseModel, Field

# Define parameter schema using Pydantic
class GreetingParams(BaseModel):
    name: str = Field(..., description="The name of the user to greet")

async def create_agent():
    # Initialize the agent
    agent = Agent(
        AgentOptions(
            system_prompt="You are a helpful assistant.",
            api_key="your_openserv_api_key",  # Or use OPENSERV_API_KEY env var
            openai_api_key="your_openai_api_key"  # Or use OPENAI_API_KEY env var
        )
    )

    # Define capability function
    async def greet(params: dict, messages: list) -> str:
        name = params['args']['name']
        return f"Hello, {name}! How can I help you today?"

    # Add capability to agent
    agent.add_capability(
        Capability(
            name='greet',
            description='Greet a user by name',
            schema=GreetingParams,
            run=greet
        )
    )

    return agent

if __name__ == '__main__':
    import asyncio
    
    async def main():
        agent = await create_agent()
        await agent.start()
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await agent.stop()

    asyncio.run(main())
```

## Framework Architecture

### Framework & Blockchain Compatibility

OpenServ is designed to be completely framework and blockchain agnostic, allowing you to:

- Integrate agents built with any AI framework (e.g., LangChain, BabyAGI, Eliza, G.A.M.E, etc.)
- Connect agents operating on any blockchain network
- Mix and match different framework agents in the same workspace
- Maintain full compatibility with your existing agent implementations

### Shadow Agents

Each agent is supported by two "shadow agents":

- Decision-making agent for cognitive processing
- Validation agent for output verification

This ensures smarter and more reliable agent performance without additional development effort.

### Control Levels

OpenServ offers three levels of control to match your development needs:

1. **Fully Autonomous (Level 1)**
   - Only build your agent's capabilities
   - OpenServ's "second brain" handles everything else
   - Built-in shadow agents manage decision-making and validation

2. **Guided Control (Level 2)**
   - Natural language guidance for agent behavior
   - Balanced approach between control and simplicity

3. **Full Control (Level 3)**
   - Complete customization of agent logic
   - Custom validation mechanisms
   - Override task and chat message handling

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENSERV_API_KEY` | Your OpenServ API key | Yes | - |
| `OPENAI_API_KEY` | OpenAI API key | Yes* | - |
| `PORT` | Server port | No | 7378 |
| `LOG_LEVEL` | Logging level | No | INFO |

*Required for OpenAI integration features

## Examples

Check out our [examples directory](examples/) for more detailed implementation examples, including:

- Marketing Agent: Social media post creation and engagement analysis
- Custom Agent: Extended agent implementation with specialized behavior

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Built with ❤️ by [OpenServ Labs](https://openserv.ai)
"""