import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
import openai
import logging

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src import Agent, AgentOptions
from src.capability import Capability

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for API keys
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError('OPENAI_API_KEY environment variable is required')

# Initialize OpenAI client
openai_client = openai.OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

# Define schema classes for capabilities
class SocialMediaPostParams(BaseModel):
    platform: str
    topic: str

class EngagementMetrics(BaseModel):
    likes: int
    shares: int
    comments: int
    impressions: int

class AnalyzeEngagementParams(BaseModel):
    platform: str
    metrics: EngagementMetrics

# Read system prompt
system_prompt_path = Path(__file__).parent.joinpath('system.md')
if not system_prompt_path.exists():
    raise FileNotFoundError("system.md not found in examples directory")
system_prompt = system_prompt_path.read_text()

# Log API key status
openserv_key = os.getenv('OPENSERV_API_KEY')
openai_key = os.getenv('OPENAI_API_KEY')
logger.info(f"OpenServ API Key: {'configured' if openserv_key else 'missing'}")
logger.info(f"OpenAI API Key: {'configured' if openai_key else 'missing'}")

# Initialize the agent (directly like in TypeScript, no custom class)
marketing_manager = Agent(
    AgentOptions(
        system_prompt=system_prompt,
        api_key=openserv_key,
        openai_api_key=openai_key,
        model="gpt-4o"
    )
)

# Define capability implementations
async def create_social_media_post(params, messages):
    """Creates a social media post for the specified platform."""
    try:
        # Extract parameters
        if isinstance(params, dict) and 'args' in params:
            args_data = params['args']
            if isinstance(args_data, str):
                import json
                args_data = json.loads(args_data)
            args = SocialMediaPostParams.model_validate(args_data)
        elif isinstance(params, SocialMediaPostParams):
            args = params
        else:
            args = SocialMediaPostParams.model_validate(params)
        
        platform = args.platform
        topic = args.topic
        
        logger.info(f"Creating social media post for platform: {platform}, topic: {topic}")

        # Create system message for the completion
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a marketing expert. Create a compelling {platform} post about: {topic}

Follow these platform-specific guidelines:
- Twitter: Max 280 characters, casual tone, use hashtags
- LinkedIn: Professional tone, industry insights, call to action
- Facebook: Engaging, conversational, can be longer

Include emojis where appropriate. Focus on driving engagement.

Only generate post for the given platform. Don't generate posts for other platforms."""
                },
                {
                    "role": "user",
                    "content": topic
                }
            ]
        )

        post_content = completion.choices[0].message.content
        logger.info(f"Created post for {platform} about {topic}")
        return post_content
    except Exception as e:
        logger.error(f"Error creating social media post: {str(e)}", exc_info=True)
        return f"Error creating social media post: {str(e)}"

async def analyze_engagement(params, messages=None):
    """Analyze social media engagement metrics."""
    try:
        # Extract parameters
        if isinstance(params, dict) and 'args' in params:
            args_data = params['args']
            if isinstance(args_data, str):
                import json
                args_data = json.loads(args_data)
            args = AnalyzeEngagementParams.model_validate(args_data)
        elif isinstance(params, AnalyzeEngagementParams):
            args = params
        else:
            args = AnalyzeEngagementParams.model_validate(params)
        
        logger.info(f"Processing engagement analysis for platform: {args.platform}")
        
        completion = openai_client.chat.completions.create(
            model='gpt-4o',
            messages=[
                {
                    'role': 'system',
                    'content': """You are a social media analytics expert. Analyze the engagement metrics and provide actionable recommendations.

Consider platform-specific benchmarks:
- Twitter: Engagement rate = (likes + shares + comments) / impressions
- LinkedIn: Engagement rate = (likes + shares + comments) / impressions * 100
- Facebook: Engagement rate = (likes + shares + comments) / impressions * 100

Provide:
1. Current engagement rate
2. Performance assessment (below average, average, above average)
3. Top 3 actionable recommendations to improve engagement
4. Key metrics to focus on for improvement"""
                },
                {
                    'role': 'user',
                    'content': f"Platform: {args.platform}\nMetrics: {args.metrics.model_dump_json()}"
                }
            ]
        )

        analysis = completion.choices[0].message.content
        logger.info(f"Generated engagement analysis for {args.platform}")
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing engagement: {str(e)}", exc_info=True)
        return f"Error analyzing engagement: {str(e)}"

# Add capabilities to the agent (exactly like TypeScript)
marketing_manager.add_capabilities([
    Capability(
        name='createSocialMediaPost',
        description='Creates a social media post for the specified platform',
        schema=SocialMediaPostParams,
        run=create_social_media_post
    ),
    Capability(
        name='analyzeEngagement',
        description='Analyzes social media engagement metrics and provides recommendations',
        schema=AnalyzeEngagementParams,
        run=analyze_engagement
    )
])

# Log the capabilities being added
for capability in marketing_manager.tools:
    logger.info(f"Adding capability: {capability.name} with schema {capability.schema.__name__}")

# Start the agent (exactly like TypeScript)
if __name__ == '__main__':
    marketing_manager.start()
