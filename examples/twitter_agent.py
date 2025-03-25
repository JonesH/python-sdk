import os
import asyncio
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import sys

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import from the local SDK
from openserv_sdk import Agent, AgentOptions

# Load environment variables from .env file
load_dotenv()

# Check for required environment variables
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize the agent
marketing_manager = Agent(
    options=AgentOptions(
        system_prompt=Path(__file__).parent.joinpath("system.md").read_text(),
        api_key=os.environ.get("OPENSERV_API_KEY"),
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
)

# Add capabilities to the agent
marketing_manager.add_capabilities([
    {
        "name": "getTwitterAccount",
        "description": "Gets the Twitter account for the current user",
        "schema": BaseModel,  # Empty schema for no parameters
        "run": lambda self, params: self.call_integration(
            workspace_id=params["action"]["workspace"]["id"],
            integration_id="twitter-v2",
            details={
                "endpoint": "/2/users/me",
                "method": "GET"
            }
        )["output"]["data"]["username"]
    },
    {
        "name": "sendMarketingTweet",
        "description": "Sends a marketing tweet to Twitter",
        "schema": BaseModel,
        "run": lambda self, params: self._handle_tweet_response(
            self.call_integration(
                workspace_id=params["action"]["workspace"]["id"],
                integration_id="twitter-v2",
                details={
                    "endpoint": "/2/tweets",
                    "method": "POST",
                    "data": {
                        "text": params["args"]["tweetText"]
                    }
                }
            )
        )
    }
])

# Add the response handler method to the agent
def _handle_tweet_response(self, response):
    self.logger.info(f"Tweet response: {response['output']}")
    if 'message' in response['output']:
        error = json.loads(json.loads(response['output']['message']))
        return f"Error {error['status']}: {error['message']}"
    return response['output']['data']['text']

marketing_manager._handle_tweet_response = _handle_tweet_response

# Start the agent
if __name__ == "__main__":
    try:
        asyncio.run(marketing_manager.start())
    except KeyboardInterrupt:
        print("Agent stopped by user")
    except Exception as e:
        print(f"Error: {e}") 