# OpenServ Python SDK

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful Python framework for building non-deterministic AI agents with advanced cognitive capabilities like reasoning, decision-making, and inter-agent collaboration within the OpenServ platform.

## Installation

```bash
pip install openserv-sdk
```

## Quick Start

Create a simple agent with a greeting capability:

```python
from openserv_sdk import Agent, AgentOptions
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

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENSERV_API_KEY` | Your OpenServ API key | Yes | - |
| `OPENAI_API_KEY` | OpenAI API key | Yes* | - |
| `PORT` | Server port | No | 7378 |

*Required for OpenAI integration features

## Core Concepts

### Capabilities

Capabilities are the building blocks of your agent. Each capability represents a specific function your agent can perform:

```python
from pydantic import BaseModel, Field
from openserv_sdk import Agent, Capability

# Define parameter schema
class SummarizeParams(BaseModel):
    text: str = Field(..., description="Text content to summarize")
    max_length: int = Field(default=100, description="Maximum length of summary")

async def summarize(params: dict, messages: list) -> str:
    """Summarize text content."""
    args = params['args']
    # Your summarization logic here
    return f"Summary of text ({len(args['text'])} chars): ..."

# Add capability to agent
agent.add_capability(
    Capability(
        name='summarize',
        description='Summarize a piece of text',
        schema=SummarizeParams,
        run=summarize
    )
)
```

### Tasks

Tasks are units of work that agents can execute:

```python
# Create a task
task = await agent.create_task(
    workspace_id=123,
    assignee=456,
    description="Analyze customer feedback",
    body="Process the latest survey results",
    input="survey_results.csv",
    expected_output="A summary of key findings",
    dependencies=[]
)

# Add progress logs
await agent.add_log_to_task(
    workspace_id=123,
    task_id=task.id,
    severity="info",
    type="text",
    body="Starting analysis..."
)

# Update task status
await agent.update_task_status(
    workspace_id=123,
    task_id=task.id,
    status="in-progress"
)
```

### Chat Interactions

Agents can participate in chat conversations:

```python
# Send a chat message
await agent.send_chat_message(
    workspace_id=123,
    agent_id=456,
    message="How can I assist you today?"
)
```

### File Operations

Agents can work with files in their workspace:

```python
# Upload a file
await agent.upload_file(
    workspace_id=123,
    path="reports/analysis.txt",
    file="Analysis results...",
    task_ids=[456]
)

# Get workspace files
files = await agent.get_files(workspace_id=123)
```

## Examples

Check out our [examples directory](examples/) for more detailed implementation examples, including:

- Marketing Agent: Social media post creation and engagement analysis
- Custom Agent: Extended agent implementation with specialized behavior

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Built with ❤️ by [OpenServ Labs](https://openserv.ai)