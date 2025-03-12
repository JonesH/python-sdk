"""
Simple example of an OpenServ agent using the Python SDK.
"""

import os
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from openserv_sdk import Agent, AgentOptions

# Load environment variables from .env file
load_dotenv()

# Define parameter models for capabilities
class GreetParams(BaseModel):
    name: str = Field(..., description="The name of the user to greet")
    language: str = Field("en", description="Language code (en, es, fr)")

class CalculateParams(BaseModel):
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")
    operation: str = Field(..., description="Operation to perform (add, subtract, multiply, divide)")

async def main():
    # Initialize the agent
    agent = Agent(AgentOptions(
        system_prompt="You are a helpful assistant.",
        api_key=os.getenv("OPENSERV_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        port=7378
    ))
    
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
    
    # Add a capability using the dictionary method
    agent.add_capability({
        "name": "calculate",
        "description": "Perform a mathematical operation on two numbers",
        "schema": CalculateParams,
        "run": async_calculate
    })
    
    # Start the agent server
    print(f"Starting agent on http://localhost:{agent.config.port}")
    await agent.start()
    
    try:
        # Keep the agent running until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down agent...")
    finally:
        await agent.stop()

async def async_calculate(params, messages):
    """Perform a mathematical operation on two numbers."""
    args = params["args"]
    a = args["a"]
    b = args["b"]
    operation = args["operation"].lower()
    
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return "Error: Cannot divide by zero"
        result = a / b
    else:
        return f"Error: Unknown operation '{operation}'"
    
    return f"Result: {result}"

if __name__ == "__main__":
    asyncio.run(main()) 