import os
from dotenv import load_dotenv
from typing import Dict, Any

from ..src import Agent, AgentOptions
from ..src.types import RespondChatMessageAction

load_dotenv()

class SophisticatedChatAgent(Agent):
    """A custom agent implementation with specialized chat handling."""
    
    async def respond_to_chat(self, action: RespondChatMessageAction) -> None:
        """Override the default chat response behavior."""
        await self.send_chat_message(
            workspace_id=action.workspace.id,
            agent_id=action.me.id,
            message="This is a custom message"
        )

async def create_custom_agent() -> Agent:
    """Create and configure the custom agent."""
    agent = SophisticatedChatAgent(
        AgentOptions(
            system_prompt="You are a helpful assistant.",
            api_key=os.getenv('OPENSERV_API_KEY'),
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
    )
    return agent

if __name__ == '__main__':
    import asyncio
    
    async def main():
        agent = await create_custom_agent()
        await agent.start()
        
        try:
            # Keep the agent running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await agent.stop()

    asyncio.run(main()) 