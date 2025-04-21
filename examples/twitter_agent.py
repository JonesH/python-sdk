import os
from dotenv import load_dotenv
from src.agent import Agent
from src.capability import Capability
from pydantic import BaseModel
from src.types import AgentOptions

# Load environment variables from .env file
load_dotenv()

# Ensure the OpenAI API key is set
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError('OPENAI_API_KEY environment variable is required')

# Define the schema for the capabilities
class GetTwitterAccountParams(BaseModel):
    pass

class SendMarketingTweetParams(BaseModel):
    tweetText: str

# Initialize the agent
system_prompt_path = os.path.join(os.path.dirname(__file__), 'system.md')

marketing_manager = Agent(AgentOptions(
    system_prompt=open(system_prompt_path).read(),
    api_key=os.getenv('OPENSERV_API_KEY'),
    openai_api_key=os.getenv('OPENAI_API_KEY')
))

# Add capabilities to the agent
marketing_manager.add_capabilities([
    Capability(
        name='getTwitterAccount',
        description='Gets the Twitter account for the current user',
        schema=GetTwitterAccountParams,
        run=lambda self, params, messages: self.call_integration({
            'workspace_id': params['action'].workspace.id,
            'integration_id': 'twitter-v2',
            'details': {
                'endpoint': '/2/users/me',
                'method': 'GET'
            }
        }).output.data.username
    ),
    Capability(
        name='sendMarketingTweet',
        description='Sends a marketing tweet to Twitter',
        schema=SendMarketingTweetParams,
        run=lambda self, params, messages: self.call_integration({
            'workspace_id': params['action'].workspace.id,
            'integration_id': 'twitter-v2',
            'details': {
                'endpoint': '/2/tweets',
                'method': 'POST',
                'data': {
                    'text': params['args'].tweetText
                }
            }
        }).output.data.text
    )
])

# Start the agent
marketing_manager.start()