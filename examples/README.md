# OpenServ SDK Examples

This directory contains example implementations using the OpenServ Python SDK.

## Basic Agent Example

The `basic_agent.py` example demonstrates:
- Creating an agent with custom capabilities
- Handling parameters with Pydantic models
- Basic error handling
- Graceful shutdown

### Running the Example

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenServ API key:
```bash
export OPENSERV_API_KEY=your-api-key
```

3. Run the example:
```bash
python basic_agent.py
```

### Testing the Agent

Once the agent is running, you can test its capabilities:

1. Greeting Capability:
```python
# Example request
{
    "capability": "greet",
    "args": {
        "name": "Alice",
        "language": "es"  # Optional, defaults to "en"
    }
}

# Expected response
"Hola, Alice!"
```

2. Calculation Capability:
```python
# Example request
{
    "capability": "calculate",
    "args": {
        "x": 10,
        "y": 5,
        "operation": "multiply"
    }
}

# Expected response
{
    "result": 50,
    "operation": "multiply",
    "x": 10,
    "y": 5
}
```

## More Examples

More examples will be added to showcase different features and use cases of the OpenServ SDK. 