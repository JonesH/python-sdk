# OpenServ Autonomous AI Agent Development Framework

[![PyPI version](https://badge.fury.io/py/openserv-sdk.svg)](https://pypi.org/project/openserv-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

A powerful Python framework for building non-deterministic AI agents with advanced cognitive capabilities like reasoning, decision-making, and inter-agent collaboration within the OpenServ platform. Built with strong typing, extensible architecture, and a fully autonomous agent runtime.

## Table of Contents

- [OpenServ Autonomous AI Agent Development Framework](#openserv-autonomous-ai-agent-development-framework)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Framework Architecture](#framework-architecture)
    - [Framework \& Blockchain Compatibility](#framework--blockchain-compatibility)
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
    - [Chat \& Communication](#chat--communication)
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
- 📝 Strong type hints with Pydantic models
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
   from openserv import Agent
   from pydantic import BaseModel
   from typing import Optional

   class GreetArgs(BaseModel):
       name: str

   agent = Agent(
       system_prompt="You are a specialized agent that..."
   )

   # Add capabilities using the add_capability method
   @agent.capability(
       name="greet",
       description="Greet a user by name"
   )
   async def greet(args: GreetArgs) -> str:
       return f"Hello, {args.name}! How can I help you today?"

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

Create a simple agent with greeting capabilities:

```python
from openserv import Agent, Capability
from openserv.types import AgentOptions
from pydantic import BaseModel
from typing import Dict, Any, List

# Define argument models
class GreetArgs(BaseModel):
    name: str

class FarewellArgs(BaseModel):
    name: str

# Initialize the agent
agent = Agent(options=AgentOptions(
    system_prompt="You are a helpful assistant.",
    api_key=os.getenv("OPENSERV_API_KEY")
))

# Create and add capabilities
greet_capability = Capability(
    name="greet",
    description="Greet a user by name",
    schema=GreetArgs,
    run=async lambda data, messages: f"Hello, {data['args'].name}! How can I help you today?"
)

farewell_capability = Capability(
    name="farewell",
    description="Say goodbye to a user",
    schema=FarewellArgs,
    run=async lambda data, messages: f"Goodbye, {data['args'].name}! Have a great day!"
)

help_capability = Capability(
    name="help",
    description="Show available commands",
    schema=BaseModel,
    run=async lambda data, messages: "Available commands: greet, farewell, help"
)

# Add capabilities to agent
agent.add_capabilities([
    greet_capability,
    farewell_capability,
    help_capability
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
- `run`: Async function that executes the capability, receiving data dict and messages

```python
from openserv import Agent, Capability
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class SummarizeArgs(BaseModel):
    text: str
    max_length: Optional[int] = 100

agent = Agent(options=AgentOptions(
    system_prompt="You are a helpful assistant."
))

# Create summarize capability
async def summarize_run(data: Dict[str, Any], messages: List[Dict[str, Any]]) -> str:
    args = data['args']
    text, max_length = args.text, args.max_length

    # Your summarization logic here
    summary = f"Summary of text ({len(text)} chars): ..."

    return summary

summarize_capability = Capability(
    name="summarize",
    description="Summarize a piece of text",
    schema=SummarizeArgs,
    run=summarize_run
)

# Add capability to agent
agent.add_capability(summarize_capability)

# Add multiple capabilities
class AnalyzeArgs(BaseModel):
    text: str

analyze_capability = Capability(
    name="analyze",
    description="Analyze text for sentiment and keywords",
    schema=AnalyzeArgs,
    run=async lambda data, messages: json.dumps({"result": "analysis complete"})
)

help_capability = Capability(
    name="help",
    description="Show available commands",
    schema=BaseModel,
    run=async lambda data, messages: "Available commands: summarize, analyze, help"
)

agent.add_capabilities([analyze_capability, help_capability])
```

Each capability function receives:

- `args`: The validated arguments matching the capability's Pydantic model
- `action`: The action context containing:
  - `task`: The current task context (if running as part of a task)
  - `workspace`: The current workspace context
  - `me`: Information about the current agent
  - Other action-specific properties

The function must return a string or coroutine that returns a string.

### Tasks

Tasks are units of work that agents can execute. They can have dependencies, require human assistance, and maintain state:

```python
from openserv.types import CreateTaskParams, AddLogToTaskParams, UpdateTaskStatusParams

# Create a task
task = await agent.create_task(CreateTaskParams(
    workspace_id=123,
    assignee=456,
    description="Analyze customer feedback",
    body="Process the latest survey results",
    input="survey_results.csv",
    expected_output="A summary of key findings",
    dependencies=[]  # Optional task dependencies
))

# Add progress logs
await agent.add_log_to_task(AddLogToTaskParams(
    workspace_id=123,
    task_id=task.id,
    severity="info",
    type="text",
    body="Starting analysis..."
))

# Update task status
await agent.update_task_status(UpdateTaskStatusParams(
    workspace_id=123,
    task_id=task.id,
    status="in-progress"
))
```

### Chat Interactions

Agents can participate in chat conversations and maintain context:

```python
class CustomerSupportAgent(Agent):
    def __init__(self):
        super().__init__(
            system_prompt="You are a customer support agent."
        )

    @agent.capability(
        name="respond_to_customer",
        description="Generate a response to a customer inquiry"
    )
    async def respond_to_customer(self, query: str, context: Optional[str] = None) -> str:
        # Generate response using the query and optional context
        return f"Thank you for your question about {query}..."

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
    workspace_id=int,
    assignee=int,
    description=str,
    body=str,
    input=str,
    expected_output=str,
    dependencies=List[int]
)
```

#### Update Task Status

```python
await agent.update_task_status(
    workspace_id=int,
    task_id=int,
    status=Literal["to-do", "in-progress", "human-assistance-required", "error", "done", "cancelled"]
)
```

#### Add Task Log

```python
await agent.add_log_to_task(
    workspace_id=int,
    task_id=int,
    severity=Literal["info", "warning", "error"],
    type=Literal["text", "openai-message"],
    body=Union[str, dict]
)
```

### Chat & Communication

#### Send Message

```python
await agent.send_chat_message(
    workspace_id=int,
    agent_id=int,
    message=str
)
```

#### Request Human Assistance

```python
await agent.request_human_assistance(
    workspace_id=int,
    task_id=int,
    type=Literal["text", "project-manager-plan-review"],
    question=Union[str, dict],
    agent_dump=Optional[dict]
)
```

### Workspace Management

#### Get Files

```python
files = await agent.get_files(
    workspace_id=int
)
```

#### Upload File

```python
await agent.upload_file(
    workspace_id=int,
    path=str,
    file=Union[bytes, str],
    skip_summarizer=Optional[bool],
    task_ids=Optional[List[int]]
)
```

### Integration Management

#### Call Integration

```python
response = await agent.call_integration(
    workspace_id=int,
    integration_id=str,
    details=dict
)
```

Allows agents to interact with external services and APIs that are integrated with OpenServ. This method provides a secure way to make API calls to configured integrations within a workspace. Authentication is handled securely and automatically through the OpenServ platform. This is primarily useful for calling external APIs in a deterministic way.

**Parameters:**

- `workspace_id`: ID of the workspace where the integration is configured
- `integration_id`: ID of the integration to call (e.g., 'twitter-v2', 'github')
- `details`: Dictionary containing:
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
class DataAnalysisAgent(Agent):
    async def do_task(self, action):
        if not action.task:
            return

        try:
            await self.update_task_status(
                workspace_id=action.workspace.id,
                task_id=action.task.id,
                status="in-progress"
            )

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

    async def handle_error(self, action, error):
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
