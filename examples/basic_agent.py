"""
Basic agent implementation for the OpenServ Python SDK.
"""

from src import Agent, Capability
from src import AgentOptions
from pydantic import BaseModel
import os
import json
import logging
from typing import Dict, Any, List
from src.types import RespondChatMessageAction
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Load system prompt from file
with open("examples/system_basic_agent.md", "r") as f:
    system_prompt = f.read()

# Define argument models
class GreetArgs(BaseModel):
    name: str

class FarewellArgs(BaseModel):
    name: str

class HelpArgs(BaseModel):  # Define an empty schema for the help command
    pass

# Define async functions for capabilities
async def greet_run(data, messages):
    # Handle both direct calls and runtime calls
    if hasattr(data, 'name'):
        name = data.name
    elif hasattr(data, 'args') and isinstance(data.args, dict):
        name = data.args.get("name", "")
    else:
        name = "there"
    return f"Hello, {name}! How can I help you today?"

async def farewell_run(data, messages):
    # Handle both direct calls and runtime calls
    if hasattr(data, 'name'):
        name = data.name
    elif hasattr(data, 'args') and isinstance(data.args, dict):
        name = data.args.get("name", "")
    else:
        name = "there"
    return f"Goodbye, {name}! Have a great day!"

async def help_run(data, messages):
    return "Available commands: greet, farewell, help"

# Custom agent class to override respond_to_chat method
class BasicAgent(Agent):
    async def respond_to_chat(self, action: RespondChatMessageAction) -> None:
        """Handle a chat message response request with direct handling instead of runtime processing."""
        messages = [
            {'role': 'system', 'content': self.config.system_prompt}
        ]

        last_message = None
        if action.messages:
            for msg in action.messages:
                message_obj = {
                    'role': 'user' if msg.author == 'user' else 'assistant',
                    'content': msg.message,
                    'id': msg.id,
                    'createdAt': msg.createdAt.isoformat()
                }
                messages.append(message_obj)
                if msg.author == 'user':
                    last_message = msg.message

        try:
            # Instead of using runtime, extract the name from the message and respond directly
            # This avoids the loop issue by not going through the runtime's tool calling system
            response = None
            
            # Simple message parsing for demo purposes
            if last_message and 'greet' in last_message.lower():
                # Extract name if present
                name = "there"  # Default
                
                # Handle "I am [Name]" pattern
                if "i am " in last_message.lower():
                    name_part = last_message.lower().split("i am ")[1]
                    name = name_part.split(',')[0].split('.')[0].strip()
                # Handle "my name is [Name]" pattern
                elif "my name is " in last_message.lower():
                    name_part = last_message.lower().split("my name is ")[1]
                    name = name_part.split(',')[0].split('.')[0].strip()
                # Handle direct name mentions
                else:
                    # Look for names before or after commas that aren't common words
                    parts = last_message.split(',')
                    for part in parts:
                        # Clean up the part
                        clean_part = part.strip().lower()
                        # Skip common words and commands
                        if (clean_part and not clean_part in ["i", "me", "greet", "hello", "hi"]
                                and len(clean_part) > 2):
                            name = clean_part
                            break
                
                # Capitalize the first letter of the name for politeness
                if name != "there":
                    name = name.capitalize()
                
                logger.info(f"Extracted name: {name}")
                
                # Use our greet capability directly
                greet_tool = next((t for t in self.tools if t.name == "greet"), None)
                if greet_tool:
                    response = await greet_tool.run(GreetArgs(name=name), messages)
            elif last_message and 'help' in last_message.lower():
                help_tool = next((t for t in self.tools if t.name == "help"), None)
                if help_tool:
                    response = await help_tool.run(HelpArgs(), messages)
            elif last_message and any(word in last_message.lower() for word in ['goodbye', 'bye', 'farewell']):
                # Extract name using the same improved logic
                name = "there"  # Default
                
                # Handle "I am [Name]" pattern
                if "i am " in last_message.lower():
                    name_part = last_message.lower().split("i am ")[1]
                    name = name_part.split(',')[0].split('.')[0].strip()
                # Handle "my name is [Name]" pattern
                elif "my name is " in last_message.lower():
                    name_part = last_message.lower().split("my name is ")[1]
                    name = name_part.split(',')[0].split('.')[0].strip()
                # Handle direct name mentions
                else:
                    # Look for names before or after commas that aren't common words
                    parts = last_message.split(',')
                    for part in parts:
                        # Clean up the part
                        clean_part = part.strip().lower()
                        # Skip common words and commands
                        if (clean_part and not clean_part in ["i", "me", "farewell", "goodbye", "bye"]
                                and len(clean_part) > 2):
                            name = clean_part
                            break
                
                # Capitalize the first letter of the name for politeness
                if name != "there":
                    name = name.capitalize()
                    
                logger.info(f"Extracted name: {name}")
                
                farewell_tool = next((t for t in self.tools if t.name == "farewell"), None)
                if farewell_tool:
                    response = await farewell_tool.run(FarewellArgs(name=name), messages)
            
            # If no specific command was detected, provide a default response
            if not response:
                response = "I'm a basic agent that can greet you, say goodbye, or provide help. Try asking for one of these!"
            
            # Send the response back to the user using the proper API client method
            if action.me and action.workspace:
                # Use the new post method added to BaseClient
                await self.api_client.post(
                    f"/workspaces/{action.workspace.id}/agent-chat/{action.me.id}/message",
                    {"message": response}
                )
                
            logger.info(f"Direct response sent: {response}")
            
        except Exception as error:
            logger.error("Chat response failed: %s", str(error), exc_info=True)
            # Don't re-raise the error to match TypeScript behavior

# Initialize the agent
def create_agent() -> Agent:
    
    # Get API keys from environment variables
    api_key = os.getenv('OPENSERV_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OPENSERV_API_KEY environment variable is not set")
    
    agent = BasicAgent(  # Use our custom BasicAgent class instead
        AgentOptions(
            api_key=api_key,
            openai_api_key=openai_api_key,
            system_prompt=system_prompt
        )
    )

    # Create and add capabilities
    greet_capability = Capability(
        name="greet",
        description="Greet a user by name",
        schema=GreetArgs,
        run=greet_run
    )

    farewell_capability = Capability(
        name="farewell",
        description="Say goodbye to a user",
        schema=FarewellArgs,
        run=farewell_run
    )

    help_capability = Capability(
        name="help",
        description="Show available commands",
        schema=HelpArgs,
        run=help_run
    )

    # Add capabilities to agent
    agent.add_capabilities([
        greet_capability,
        farewell_capability,
        help_capability
    ])

    return agent

if __name__ == '__main__':
    agent = create_agent()
    agent.start()
