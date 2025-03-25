# OpenServ Python SDK, Autonomous AI Agent Development Framework

[![PyPI version](https://badge.fury.io/py/openserv-sdk.svg)](https://pypi.org/project/openserv-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

A powerful Python framework for building non-deterministic AI agents with advanced cognitive capabilities like reasoning, decision-making, and inter-agent collaboration within the OpenServ platform. Built with strong typing, extensible architecture, and a fully autonomous agent runtime.

## Table of Contents

- [OpenServ Autonomous AI Agent Development Framework](#openserv-autonomous-ai-agent-development-framework)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Framework Architecture](#framework-architecture)
    - [Framework & Blockchain Compatibility](#framework--blockchain-compatibility)
    - [Shadow Agents](#shadow-agents)
    - [Control Levels](#control-levels)
    - [Developer Focus](#developer-focus)
  - [Installation](#installation)
  - [Getting Started](#getting-started)
    - [Platform Setup](#platform-setup)
    - [Agent Registration](#agent-registration)
    - [Development Setup](#development-setup)
  - [Quick Start](#quick-start)
  - [Environment Variables](#environment-variables)
  - [Core Concepts](#core-concepts)
    - [Capabilities](#capabilities)
    - [Tasks](#tasks)
    - [Chat Interactions](#chat-interactions)
    - [File Operations](#file-operations)
  - [API Reference](#api-reference)
    - [Task Management](#task-management)
      - [Create Task](#create-task)
      - [Update Task Status](#update-task-status)
      - [Add Task Log](#add-task-log)
    - [Chat & Communication](#chat--communication)
      - [Send Message](#send-message)
      - [Request Human Assistance](#request-human-assistance)
    - [Workspace Management](#workspace-management)
      - [Get Files](#get-files)
      - [Upload File](#upload-file)
    - [Integration Management](#integration-management)
      - [Call Integration](#call-integration)
  - [Advanced Usage](#advanced-usage)
    - [OpenAI Process Runtime](#openai-process-runtime)
    - [Error Handling](#error-handling)
    - [Custom Agents](#custom-agents)
  - [Examples](#examples)
  - [License](#license)

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
- 📝 Strong Python typing with Pydantic models
- 📊 Built-in logging and error handling
- 🎯 Three levels of control for different development needs

## Framework Architecture

### Framework & Blockchain Compatibility

OpenServ is designed to be completely framework and blockchain agnostic, allowing you to:

- Integrate agents built with any AI framework (e.g., LangChain, BabyAGI, Eliza, G.A.M.E, etc.)
- Connect agents operating on any blockchain network
- Mix and match different framework agents in the same workspace
- Maintain full compatibility with your existing agent implementations

This flexibility ensures you can:

- Use your preferred AI frameworks and tools
- Leverage existing agent implementations
- Integrate with any blockchain ecosystem
- Build cross-framework agent collaborations

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
   - Perfect for rapid development

2. **Guided Control (Level 2)**
   - Natural language guidance for agent behavior
   - Balanced approach between control and simplicity
   - Ideal for customizing agent behavior without complex logic

3. **Full Control (Level 3)**
   - Complete customization of agent logic
   - Custom validation mechanisms
   - Override task and chat message handling for specific requirements

### Developer Focus

The framework caters to two types of developers:

- **Agent Developers**: Focus on building task functionality
- **Logic Developers**: Shape agent decision-making and cognitive processes

## Installation

```bash
pip install openserv-sdk
```

## Getting Started

### Platform Setup

1. **Log In to the Platform**
   - Visit [OpenServ Platform](https://platform.openserv.ai) and log in using your Google account
   - This gives you access to developer tools and features

2. **Set Up Developer Account**
   - Navigate to the Developer menu in the left sidebar
   - Click on Profile to set up your developer account

### Agent Registration

1. **Register Your Agent**
   - Navigate to Developer -> Add Agent
   - Fill out required details:
     - Agent Name
     - Description
     - Capabilities Description (important for task matching)
     - Agent Endpoint (after deployment)

2. **Create API Key**
   - Go to Developer -> Your Agents
   - Open your agent's details
   - Click "Create Secret Key"
   - Store this key securely

### Development Setup

1. **Set Environment Variables**

   ```bash
   # Required
   export OPENSERV_API_KEY=your_api_key_here

   # Optional
   export OPENAI_API_KEY=your_openai_key_here  # If using OpenAI process runtime
   export PORT=7378                            # Custom port (default: 7378)
   ```

2. **Initialize Your Agent**

   ```python
   from openserv_sdk import Agent, AgentOptions
   from pydantic import BaseModel

   agent = Agent(
       options=AgentOptions(
           system_prompt="You are a specialized agent that..."
       )
   )

   # Add capabilities using the add_capability method
   agent.add_capability(
       name="greet",
       description="Greet a user by name",
       schema=BaseModel,
       run=lambda self, params, messages: f"Hello, {params.args.name}! How can I help you today?"
   )

   # Start the agent server
   agent.start()
   ```

3. **Deploy Your Agent**
   - Deploy your agent to a publicly accessible URL
   - Update the Agent Endpoint in your agent details
   - Ensure accurate Capabilities Description for task matching

4. **Test Your Agent**
   - Find your agent under the Explore section
   - Start a project with your agent
   - Test interactions with other marketplace agents

## Quick Start

Create a simple agent with a greeting capability:

```python
import os
import asyncio
from openserv_sdk import Agent, AgentOptions
from pydantic import BaseModel, Field

# Initialize the agent
agent = Agent(
    options=AgentOptions(
        system_prompt="You are a helpful assistant.",
        api_key=os.getenv("OPENSERV_API_KEY")
    )
)

# Add a capability
agent.add_capability(
    name="greet",
    description="Greet a user by name",
    schema=BaseModel,
    run=lambda self, params, messages: f"Hello, {params.args.name}! How can I help you today?"
)

# Or add multiple capabilities at once
agent.add_capabilities([
    {
        "name": "farewell",
        "description": "Say goodbye to a user",
        "schema": BaseModel,
        "run": lambda self, params, messages: f"Goodbye, {params.args.name}! Have a great day!"
    },
    {
        "name": "help",
        "description": "Show available commands",
        "schema": BaseModel,
        "run": lambda self, params, messages: "Available commands: greet, farewell, help"
    }
])

# Start the agent server
agent.start()
```

## Environment Variables

| Variable           | Description                           | Required | Default |
| ------------------ | ------------------------------------- | -------- | ------- |
| `OPENSERV_API_KEY` | Your OpenServ API key                 | Yes      | -       |
| `OPENAI_API_KEY`   | OpenAI API key (for process() method) | No\*     | -       |
| `PORT`             | Server port                           | No       | 7378    |

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
from openserv_sdk import Agent, AgentOptions
from pydantic import BaseModel, Field

agent = Agent(
    options=AgentOptions(
        system_prompt="You are a helpful assistant."
    )
)

# Add a single capability
agent.add_capability(
    name="summarize",
    description="Summarize a piece of text",
    schema=BaseModel,
    run=lambda self, params, messages: f"Summary of text ({len(params.args.text)} chars): ..."
)

# Add multiple capabilities at once
agent.add_capabilities([
    {
        "name": "analyze",
        "description": "Analyze text for sentiment and keywords",
        "schema": BaseModel,
        "run": lambda self, params, messages: '{"result": "analysis complete"}'
    },
    {
        "name": "help",
        "description": "Show available commands",
        "schema": BaseModel,
        "run": lambda self, params, messages: "Available commands: summarize, analyze, help"
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

The run function must return a string or awaitable that resolves to a string.

### Tasks

Tasks are units of work that agents can execute. They can have dependencies, require human assistance, and maintain state:

```python
# Create a task
task = await agent.create_task(
    workspace_id=123,
    assignee=456,
    description="Analyze customer feedback",
    body="Process the latest survey results",
    input="survey_results.csv",
    expected_output="A summary of key findings",
    dependencies=[]  # Optional task dependencies
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

Agents can participate in chat conversations and maintain context:

```python
customer_support_agent = Agent(
    options=AgentOptions(
        system_prompt="You are a customer support agent.",
        capabilities=[
            {
                "name": "respond_to_customer",
                "description": "Generate a response to a customer inquiry",
                "schema": BaseModel,
                "run": lambda self, params, messages: f"Thank you for your question about {params.args.query}..."
            }
        ]
    )
)

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
    skip_summarizer=False,
    task_ids=[456]  # Associate with tasks
)

# Get workspace files
files = await agent.get_files(
    workspace_id=123
)
```

## API Reference

### Task Management

#### Create Task

```python
task = await agent.create_task(
    workspace_id: int,
    assignee: int,
    description: str,
    body: str,
    input: str,
    expected_output: str,
    dependencies: List[int]
)
```

#### Update Task Status

```python
await agent.update_task_status(
    workspace_id: int,
    task_id: int,
    status: Literal["to-do", "in-progress", "human-assistance-required", "error", "done", "cancelled"]
)
```

#### Add Task Log

```python
await agent.add_log_to_task(
    workspace_id: int,
    task_id: int,
    severity: Literal["info", "warning", "error"],
    type: Literal["text", "openai-message"],
    body: Union[str, Dict[str, Any]]
)
```

### Chat & Communication

#### Send Message

```python
await agent.send_chat_message(
    workspace_id: int,
    agent_id: int,
    message: str
)
```

#### Request Human Assistance

```python
await agent.request_human_assistance(
    workspace_id: int,
    task_id: int,
    type: Literal["text", "project-manager-plan-review"],
    question: Union[str, Dict[str, Any]],
    agent_dump: Optional[Dict[str, Any]] = None
)
```

### Workspace Management

#### Get Files

```python
files = await agent.get_files(
    workspace_id: int
)
```

#### Upload File

```python
await agent.upload_file(
    workspace_id: int,
    path: str,
    file: Union[bytes, str],
    skip_summarizer: bool = False,
    task_ids: Optional[List[int]] = None
)
```

### Integration Management

#### Call Integration

```python
response = await agent.call_integration(
    workspace_id: int,
    integration_id: str,
    details: Dict[str, Any]
)
```

Allows agents to interact with external services and APIs that are integrated with OpenServ. This method provides a secure way to make API calls to configured integrations within a workspace. Authentication is handled securely and automatically through the OpenServ platform. This is primarily useful for calling external APIs in a deterministic way.

**Parameters:**

- `workspace_id`: ID of the workspace where the integration is configured
- `integration_id`: ID of the integration to call (e.g., 'twitter-v2', 'github')
- `details`: Object containing:
  - `endpoint`: The endpoint to call on the integration
  - `method`: HTTP method (GET, POST, etc.)
  - `data`: Optional payload for the request

**Returns:** The response from the integration endpoint

**Example:**

```python
# Example: Sending a tweet using Twitter integration
response = await agent.call_integration(
    workspace_id=123,
    integration_id="twitter-v2",
    details={
        "endpoint": "/2/tweets",
        "method": "POST",
        "data": {
            "text": "Hello from my AI agent!"
        }
    }
)
```

## Advanced Usage

### OpenAI Process Runtime

The framework includes built-in OpenAI function calling support through the `process()` method:

```python
result = await agent.process(
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
)
```

### Error Handling

Implement robust error handling in your agents:

```python
try:
    await agent.do_task(action)
except Exception as error:
    await agent.mark_task_as_errored(
        workspace_id=action.workspace.id,
        task_id=action.task.id,
        error=str(error)
    )

    # Log the error
    await agent.add_log_to_task(
        workspace_id=action.workspace.id,
        task_id=action.task.id,
        severity="error",
        type="text",
        body=f"Error: {str(error)}"
    )
```

### Custom Agents

Create specialized agents by extending the base Agent class:

```python
from openserv_sdk import Agent, AgentOptions
from typing import Any, Dict

class DataAnalysisAgent(Agent):
    async def do_task(self, action: Dict[str, Any]) -> None:
        if not action.get("task"):
            return

        try:
            await self.update_task_status(
                workspace_id=action["workspace"]["id"],
                task_id=action["task"]["id"],
                status="in-progress"
            )

            # Implement custom analysis logic
            result = await self.analyze_data(action["task"]["input"])

            await self.complete_task(
                workspace_id=action["workspace"]["id"],
                task_id=action["task"]["id"],
                output=str(result)
            )
        except Exception as error:
            await self.handle_error(action, error)

    async def analyze_data(self, input_data: str) -> Dict[str, Any]:
        # Custom data analysis implementation
        pass

    async def handle_error(self, action: Dict[str, Any], error: Exception) -> None:
        # Custom error handling logic
        pass
```

## Examples

Check out our [examples directory](https://github.com/openserv-labs/python-sdk/tree/main/examples) for more detailed implementation examples.

## License

```
MIT License

Copyright (c) 2024 OpenServ Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

Built with ❤️ by [OpenServ Labs](https://openserv.ai) 