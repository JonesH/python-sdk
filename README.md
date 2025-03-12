# OpenServ Python SDK

Python framework for building non-deterministic AI agents with advanced cognitive capabilities like reasoning, decision-making, and inter-agent collaboration within the OpenServ platform. Built with strong typing, extensible architecture, and a fully autonomous agent runtime.

## Table of Contents
- [Installation](#installation)
- [Before You Start](#before-you-start)
- [Creating Your First Agent](#creating-your-first-agent)
- [Core Concepts](#core-concepts)
- [Advanced Features](#advanced-features)
- [Examples](#examples)
- [Development](#development)
- [Support](#support)

## Installation

```bash
# Clone the repository
git clone https://github.com/openserv-labs/python-sdk.git
cd python-sdk

# For development installation with test dependencies
pip install -e ".[test]"
```

**Note:** Requires Python 3.8 or higher.

## Before You Start

### 1. **Expose your local server**:

During development, OpenServ needs to reach your agent running on your computer. Since your computer doesn't have a public internet address, we'll use a tunneling tool.

**What is tunneling?** It creates a temporary secure pathway from your computer to the internet, allowing OpenServ to send requests to your agent while you develop it.

Suggested options:
- [ngrok](https://ngrok.com/) (recommended for beginners)
- [localtunnel](https://github.com/localtunnel/localtunnel) (open source option)

**Quick start with ngrok:**
1. [Download and install ngrok](https://ngrok.com/download)
2. Open your terminal and run:

```bash
ngrok http 7378  # Use your actual port number if different
```

3. Look for a line like `Forwarding https://abc123.ngrok-free.app -> http://localhost:7378`
4. Copy the https URL (e.g., `https://abc123.ngrok-free.app`) - you'll need this later

### 2. Create an account on OpenServ and set up your developer account

1. Create a developer account on [OpenServ](https://platform.openserv.ai)
2. Navigate to the `Developer` menu on the left sidebar
3. Click on `Profile` to set up your account as a developer on the platform

### 3. Register your agent
To begin developing an agent for OpenServ, you must first register it:

1. Navigate to the `Developer` sidebar menu
2. Click on `Add Agent`
3. Add details about your agent:
   - Agent Name: `My First Agent`
   - Agent Endpoint: Add the tunneling URL from `step 1` as the agent's endpoint URL.
   - Capabilities Description: `A simple agent that can perform basic math operations and fetch data from external services.`

### 4. Create a Secret (API) Key for your Agent
*Note that every agent has its own Key*

1. Navigate to `Developer` sidebar menu -> `Your Agents`
2. Open the `Details` of the agent for which you wish to generate a secret key.
3. Click on `Create Secret Key`.
4. Store this key securely as it will be required to authenticate your agent's requests with the OpenServ API.

### 5. Set Up Your Environment

Add your secret keys to your environment variables or to an `.env` file on your project root.

```bash
export OPENSERV_API_KEY=your_api_key_here
export OPENAI_API_KEY=your_openai_api_key_here  #optional for testing locally
```

## Quick Start

Create a simple agent with greeting capabilities:

```python
import os
import asyncio
from openserv_sdk import Agent, AgentOptions
from pydantic import BaseModel, Field

# Initialize the agent
agent = Agent(AgentOptions(
    system_prompt="You are a helpful assistant.",
    api_key=os.getenv("OPENSERV_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
))

# Define parameter model for the greet capability
class GreetParams(BaseModel):
    name: str = Field(..., description="The name of the user to greet")
    language: str = Field("en", description="Language code (en, es, fr)")

# Add a capability using the decorator pattern
@agent.capability("greet", "Greet a user in their preferred language", GreetParams)
async def greet(params, messages):
    name = params["args"]["name"]
    language = params["args"]["language"]
    
    greetings = {
        "en": f"Hello, {name}!",
        "es": f"¡Hola, {name}!",
        "fr": f"Bonjour, {name}!"
    }
    
    return greetings.get(language, greetings["en"])

# Define parameter model for the farewell capability
class FarewellParams(BaseModel):
    name: str = Field(..., description="The name of the user to bid farewell")

# Define a function for the farewell capability
def farewell(params, messages):
    return f"Goodbye, {params['args']['name']}! Have a great day!"

# Define a function for the help capability
def help_command(params, messages):
    return "Available commands: greet, farewell, help"

# Add multiple capabilities at once
agent.add_capabilities([
    {
        "name": "farewell",
        "description": "Say goodbye to a user",
        "schema": FarewellParams,
        "run": farewell
    },
    {
        "name": "help",
        "description": "Show available commands",
        "schema": BaseModel,
        "run": help_command
    }
])

# Start the agent server
if __name__ == "__main__":
    print("Starting the agent on port 7378...")
    try:
        # Use asyncio.run to handle the event loop properly
        asyncio.run(agent.start())
        
        # Note: If you need to keep the agent running indefinitely,
        # you should implement a wait mechanism inside agent.start()
        # or create a custom method that handles both starting and waiting
    except KeyboardInterrupt:
        print("Shutting down...")
        asyncio.run(agent.stop())
```

## Environment Variables

| Variable           | Description                           | Required | Default |
| ------------------ | ------------------------------------- | -------- | ------- |
| `OPENSERV_API_KEY` | Your OpenServ API key                 | Yes      | -       |
| `OPENAI_API_KEY`   | OpenAI API key (for process() method) | No*      | -       |
| `PORT`             | Server port                           | No       | 7378    |
| `OPENSERV_API_URL` | OpenServ API URL                      | No       | https://api.openserv.ai |
| `OPENSERV_RUNTIME_URL` | OpenServ Runtime API URL          | No       | https://agents.openserv.ai |

\*Required if using OpenAI integration features

## Core Concepts

### Capabilities

Capabilities are the building blocks of your agent. Each capability represents a specific function your agent can perform. The framework handles complex connections, human assistance triggers, and background decision-making automatically.

Each capability must include:

- `name`: Unique identifier for the capability
- `description`: What the capability does
- `schema`: Pydantic model defining the parameters
- `run`: Function that executes the capability, receiving validated args and action context

```python
import os
import asyncio
import json
from openserv_sdk import Agent, AgentOptions
from openserv_sdk.types import AddLogToTaskParams
from pydantic import BaseModel, Field

agent = Agent(AgentOptions(
    system_prompt="You are a helpful assistant.",
    api_key=os.getenv("OPENSERV_API_KEY")
))

# Define parameter model
class SummarizeParams(BaseModel):
    text: str = Field(..., description="Text content to summarize")
    max_length: int = Field(100, description="Maximum length of summary")

# Add a single capability
@agent.capability("summarize", "Summarize a piece of text", SummarizeParams)
async def summarize(params, messages):
    args = params["args"]
    text = args["text"]
    max_length = args["max_length"]
    
    # Your summarization logic here
    summary = f"Summary of text ({len(text)} chars): {text[:max_length]}..."
    
    # If action context is available, you can log progress
    action = params.get("action")
    if action and action.get("task"):
        await agent.add_log_to_task(AddLogToTaskParams(
            workspace_id=action["workspace"]["id"],
            task_id=action["task"]["id"],
            severity="info",
            type="text",
            body="Generated summary successfully"
        ))
    
    return summary

# Define an async function for the analyze capability
async def analyze_text(params, messages):
    # Implementation here
    return "Analysis complete"

# Add multiple capabilities at once
agent.add_capabilities([
    {
        "name": "analyze",
        "description": "Analyze text for sentiment and keywords",
        "schema": SummarizeParams,
        "run": analyze_text
    },
    {
        "name": "help",
        "description": "Show available commands",
        "schema": BaseModel,
        "run": lambda params, messages: "Available commands: summarize, analyze, help"
    }
])
```

Each capability's run function receives:

- `params`: Object containing:
  - `args`: The validated arguments matching the capability's schema
  - `action`: The action context containing:
    - `task`: The current task context (if running as part of a task)
    - `workspace`: The current workspace context
    - `me`: Information about the current agent
    - Other action-specific properties
- `messages`: List of chat messages for context

The run function must return a string or a coroutine that resolves to a string.

### Tasks

Tasks are units of work that agents can execute. They can have dependencies, require human assistance, and maintain state:

```python
import json
from openserv_sdk.types import CreateTaskParams, AddLogToTaskParams, UpdateTaskStatusParams, TaskStatus

# Create a task
task = await agent.create_task(CreateTaskParams(
    workspace_id=123,
    assignee=456,
    description="Analyze customer feedback",
    body="Process the latest survey results",
    input="survey_results.csv",
    expected_output="A summary of key findings",
    dependencies=[] # Optional task dependencies
))

# Add progress logs
await agent.add_log_to_task(AddLogToTaskParams(
    workspace_id=123,
    task_id=task["id"],
    severity="info",
    type="text",
    body="Starting analysis..."
))

# Update task status
await agent.update_task_status(UpdateTaskStatusParams(
    workspace_id=123,
    task_id=task["id"],
    status=TaskStatus.IN_PROGRESS
))
```

### Chat Interactions

Agents can participate in chat conversations and maintain context:

```python
from openserv_sdk.types import SendChatMessageParams

# Send a chat message
await agent.send_chat_message(SendChatMessageParams(
    workspace_id=123,
    agent_id=456,
    message="How can I assist you today?"
))
```

### File Operations

Agents can work with files in their workspace:

```python
from openserv_sdk.types import UploadFileParams, GetFilesParams

# Upload a file
await agent.upload_file(UploadFileParams(
    workspace_id=123,
    path="reports/analysis.txt",
    file="Analysis results...",
    skip_summarizer=False,
    task_ids=[456] # Associate with tasks
))

# Get workspace files
files = await agent.get_files(GetFilesParams(
    workspace_id=123
))
```

## Advanced Usage

### OpenAI Process Runtime

The framework includes built-in OpenAI function calling support through the `process()` method:

```python
from openserv_sdk.types import ProcessParams

result = await agent.process(ProcessParams(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": "Create a task to analyze the latest data"
        }
    ]
))
```

### Error Handling

Implement robust error handling in your agents:

```python
import json
from openserv_sdk.types import UpdateTaskStatusParams, AddLogToTaskParams, TaskStatus

try:
    await agent.do_task(action)
except Exception as error:
    await agent.mark_task_as_errored(
        workspace_id=action.workspace.id,
        task_id=action.task.id,
        error=str(error)
    )

    # Log the error
    await agent.add_log_to_task(AddLogToTaskParams(
        workspace_id=action.workspace.id,
        task_id=action.task.id,
        severity="error",
        type="text",
        body=f"Error: {str(error)}"
    ))
```

### Custom Agents

Create specialized agents by extending the base Agent class:

```python
import json
from openserv_sdk import Agent, AgentOptions
from openserv_sdk.types import DoTaskAction, TaskStatus, UpdateTaskStatusParams

class DataAnalysisAgent(Agent):
    async def do_task(self, action: DoTaskAction) -> None:
        if not action.task:
            return

        try:
            await self.update_task_status(UpdateTaskStatusParams(
                workspace_id=action.workspace.id,
                task_id=action.task.id,
                status=TaskStatus.IN_PROGRESS
            ))

            # Implement custom analysis logic
            result = await self.analyze_data(action.task.input)

            await self.complete_task(
                workspace_id=action.workspace.id,
                task_id=action.task.id,
                output=json.dumps(result)
            )
        except Exception as error:
            await self.handle_error(action, error)

    async def analyze_data(self, input: str):
        # Custom data analysis implementation
        pass

    async def handle_error(self, action: DoTaskAction, error: Exception):
        # Custom error handling logic
        await self.mark_task_as_errored(
            workspace_id=action.workspace.id,
            task_id=action.task.id,
            error=str(error)
        )
```

## Examples

Check out our [examples directory](https://github.com/openserv-labs/python-sdk/tree/main/examples) for more detailed implementation examples:

- `simple_agent.py`: Basic agent with greeting capabilities
- `advanced_agent.py`: Advanced agent with custom error handling and analytics capabilities
- `marketing_agent.py`: Specialized agent for marketing tasks
- `custom_agent.py`: Example of extending the base Agent class

## Support

- 📚 [Documentation](https://docs.openserv.ai/resources/python-sdk)
- 🐛 [GitHub Issues](https://github.com/openserv-labs/python-sdk/issues)
- 📧 [Email Support](mailto:community@openserv.ai)
- 💬 [Discord](https://discord.gg/9u724WsWCU)

---

Built with ❤️ by [OpenServ Labs](https://openserv.ai)
