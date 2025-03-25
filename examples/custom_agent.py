import os
import asyncio
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import from the local SDK
from openserv_sdk import Agent, AgentOptions
from openserv_sdk.types import RespondChatMessageActionSchema

# Load environment variables from .env file
load_dotenv()

class SophisticatedChatAgent(Agent):
    """
    A custom agent that overrides the default chat response behavior.
    """
    
    async def respond_to_chat(self, action: RespondChatMessageActionSchema) -> None:
        """
        Override the default chat response behavior with custom logic.
        
        Args:
            action: The chat action to handle.
        """
        await self.send_chat_message({
            "workspace_id": action.workspace.id,
            "agent_id": action.me.id,
            "message": "This is a custom message"
        })

# Create and start the agent in one step
if __name__ == "__main__":
    try:
        asyncio.run(SophisticatedChatAgent(
            options=AgentOptions(
                system_prompt="You are a helpful assistant.",
                api_key=os.environ.get("OPENSERV_API_KEY")
            )
        ).start())
    except KeyboardInterrupt:
        print("Agent stopped by user")
    except Exception as e:
        print(f"Error: {e}") 