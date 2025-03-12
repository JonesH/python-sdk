"""
Advanced example of an OpenServ agent using the Python SDK.

This example demonstrates more advanced features of the SDK that match
the TypeScript implementation, including error handling, custom agent classes,
and more complex capabilities.
"""

import os
import asyncio
import json
from typing import Dict, Any, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from openserv_sdk import Agent, AgentOptions
from openserv_sdk.types import DoTaskAction, RespondChatMessageAction, TaskStatus
from openserv_sdk.exceptions import RuntimeError, ToolError

# Load environment variables from .env file
load_dotenv()

# Define parameter models for capabilities
class SearchParams(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, description="Maximum number of results to return")

class DataAnalysisParams(BaseModel):
    data: str = Field(..., description="Data to analyze (JSON string)")
    analysis_type: str = Field(..., description="Type of analysis to perform (summary, trends, forecast)")
    include_charts: bool = Field(False, description="Whether to include chart data in the response")

# Custom agent class that extends the base Agent
class AnalyticsAgent(Agent):
    """
    Custom agent implementation with specialized analytics capabilities.
    
    This agent overrides the do_task method to provide custom task handling
    and adds specialized analytics capabilities.
    """
    
    async def do_task(self, action: DoTaskAction) -> None:
        """
        Custom implementation of task handling.
        
        This method demonstrates how to override the default task handling
        to provide custom behavior.
        """
        if not action.task or not action.task.id:
            return
            
        print(f"AnalyticsAgent processing task: {action.task.description}")
        
        try:
            # Update task status
            await self.update_task_status({
                "workspace_id": action.workspace.id,
                "task_id": action.task.id,
                "status": TaskStatus.IN_PROGRESS
            })
            
            # Add a log entry
            await self.add_log_to_task({
                "workspace_id": action.workspace.id,
                "task_id": action.task.id,
                "severity": "info",
                "type": "text",
                "body": "Starting analytics task processing"
            })
            
            # Call the parent implementation to handle the task
            await super().do_task(action)
            
        except Exception as e:
            print(f"Error in custom task handler: {str(e)}")
            await self.mark_task_as_errored(
                workspace_id=action.workspace.id,
                task_id=action.task.id,
                error=str(e)
            )
            
            # Add error log
            await self.add_log_to_task({
                "workspace_id": action.workspace.id,
                "task_id": action.task.id,
                "severity": "error",
                "type": "text",
                "body": f"Error: {str(e)}"
            })

# Custom error handler
def handle_error(error: Exception, context: Dict[str, Any] = None):
    """Custom error handler for the agent."""
    print(f"Custom error handler called: {str(error)}")
    if context:
        print(f"Error context: {json.dumps(context, default=str)}")

async def main():
    """Main function to set up and run the agent."""
    # Initialize the agent with custom error handler
    agent = AnalyticsAgent(AgentOptions(
        system_prompt="You are an advanced analytics assistant that can search for information and analyze data.",
        api_key=os.getenv("OPENSERV_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        port=7378,
        on_error=handle_error
    ))
    
    # Add search capability
    @agent.capability("search", "Search for information", SearchParams)
    async def search(params, messages):
        """Search for information based on a query."""
        args = params["args"]
        query = args["query"]
        max_results = args["max_results"]
        
        # Simulate search results
        results = [
            {"title": f"Result {i} for '{query}'", "url": f"https://example.com/result{i}"} 
            for i in range(1, max_results + 1)
        ]
        
        return json.dumps(results, indent=2)
    
    # Add data analysis capability using the dictionary method
    agent.add_capability({
        "name": "analyze_data",
        "description": "Analyze data and provide insights",
        "schema": DataAnalysisParams,
        "run": analyze_data
    })
    
    # Start the agent server
    print(f"Starting advanced agent on http://localhost:{agent.config.port}")
    await agent.start()
    
    try:
        # Keep the agent running until interrupted
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down agent...")
    finally:
        await agent.stop()

async def analyze_data(params, messages):
    """Analyze data and provide insights."""
    args = params["args"]
    
    try:
        # Parse the data
        data_str = args["data"]
        analysis_type = args["analysis_type"]
        include_charts = args["include_charts"]
        
        # Parse JSON data
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            return "Error: Invalid JSON data"
        
        # Perform different types of analysis
        if analysis_type == "summary":
            result = {
                "summary": f"Analyzed {len(data)} data points",
                "mean": sum(data) / len(data) if isinstance(data, list) else "N/A",
                "timestamp": "2023-06-15T12:00:00Z"
            }
        elif analysis_type == "trends":
            result = {
                "trend": "upward" if isinstance(data, list) and len(data) > 1 and data[-1] > data[0] else "downward",
                "change_percent": "15%",
                "timestamp": "2023-06-15T12:00:00Z"
            }
        elif analysis_type == "forecast":
            result = {
                "forecast": "positive",
                "confidence": "85%",
                "timestamp": "2023-06-15T12:00:00Z"
            }
        else:
            return f"Error: Unknown analysis type '{analysis_type}'"
        
        # Add chart data if requested
        if include_charts:
            result["chart_data"] = {
                "type": "line",
                "data": [{"x": i, "y": val} for i, val in enumerate(data)] if isinstance(data, list) else []
            }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error analyzing data: {str(e)}"

if __name__ == "__main__":
    asyncio.run(main()) 