"""
Basic example showing how to create and run an OpenServ agent.
"""

import asyncio
import os
from typing import Dict, List

from openserv_sdk import Agent, AgentOptions
from pydantic import BaseModel, Field

class GreetingParams(BaseModel):
    """Parameters for the greeting capability."""
    name: str = Field(..., description="Name of the person to greet")
    language: str = Field(default="en", description="Language code (e.g., 'en', 'es', 'fr')")

class CalculationParams(BaseModel):
    """Parameters for the calculation capability."""
    x: float = Field(..., description="First number")
    y: float = Field(..., description="Second number")
    operation: str = Field(..., description="Operation to perform (add, subtract, multiply, divide)")

async def main():
    # Initialize the agent
    agent = Agent(AgentOptions(
        api_key=os.getenv("OPENSERV_API_KEY", "your-api-key"),
        agent_id="example-agent",
        name="Example Agent",
        description="A simple example agent with greeting and calculation capabilities"
    ))

    # Register greeting capability
    @agent.capability("greet")
    async def greet(params: Dict, messages: List[Dict]) -> str:
        name = params["args"]["name"]
        language = params["args"].get("language", "en")
        
        greetings = {
            "en": "Hello",
            "es": "Hola",
            "fr": "Bonjour"
        }
        
        greeting = greetings.get(language, greetings["en"])
        return f"{greeting}, {name}!"

    # Register calculation capability
    @agent.capability("calculate")
    async def calculate(params: Dict, messages: List[Dict]) -> Dict[str, float]:
        args = params["args"]
        x, y = args["x"], args["y"]
        operation = args["operation"]
        
        result = None
        if operation == "add":
            result = x + y
        elif operation == "subtract":
            result = x - y
        elif operation == "multiply":
            result = x * y
        elif operation == "divide":
            if y == 0:
                raise ValueError("Cannot divide by zero")
            result = x / y
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return {
            "result": result,
            "operation": operation,
            "x": x,
            "y": y
        }

    # Start the agent
    await agent.start()
    
    try:
        # Keep the agent running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        # Stop the agent gracefully
        await agent.stop()

if __name__ == "__main__":
    asyncio.run(main()) 