import os
from dotenv import load_dotenv
from src.agent import Agent
from src.types import RespondChatMessageAction

# Load environment variables from .env file
load_dotenv()

class SophisticatedChatAgent(Agent):
    async def respond_to_chat(self, action: RespondChatMessageAction):
        """Override the respond_to_chat method to send a custom message."""
        await self.send_chat_message(
            workspace_id=action.workspace.id,
            agent_id=action.me.id,
            message='This is a custom message'
        )

# Initialize the custom agent
agent = SophisticatedChatAgent({
    'system_prompt': 'You are a helpful assistant.',
    'api_key': os.getenv('OPENSERV_API_KEY'),
    'openai_api_key': os.getenv('OPENAI_API_KEY')
})

# Start the agent
agent.start()