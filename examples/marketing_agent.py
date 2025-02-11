import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import openai
import logging
import traceback

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src import Agent, AgentOptions
from src.capability import Capability

load_dotenv()

logger = logging.getLogger(__name__)

def get_openai_client():
    """Get OpenAI client with API key validation."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY environment variable is required')
    return openai.OpenAI(api_key=api_key)

class SocialMediaPostParams(BaseModel):
    platform: str = Field(..., description="The social media platform to post to")
    topic: str = Field(..., description="The topic to create a post about")

class EngagementMetrics(BaseModel):
    likes: int = Field(..., ge=0)
    shares: int = Field(..., ge=0)
    comments: int = Field(..., ge=0)
    impressions: int = Field(..., ge=0)

class AnalyzeEngagementParams(BaseModel):
    platform: str = Field(..., description="The social media platform to analyze")
    metrics: EngagementMetrics

async def create_social_media_post(params: Dict, messages: List[Dict[str, str]]) -> str:
    """Creates a social media post for the specified platform."""
    try:
        args = params.get('args', {})
        logger.info(f"Creating social media post with args: {args}")
        
        client = openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Validate platform is one of the supported platforms
        platform = args['platform'].lower()
        if platform not in ['twitter', 'linkedin', 'facebook']:
            raise ValueError(f"Unsupported platform: {platform}")
        
        completion = await client.chat.completions.create(
            model='gpt-4',  # Changed from gpt-4o to gpt-4
            messages=[
                {
                    'role': 'system',
                    'content': f"""You are a marketing expert. Create a compelling {platform} post about: {args['topic']}

Follow these platform-specific guidelines:
- Twitter: Max 280 characters, casual tone, use hashtags
- LinkedIn: Professional tone, industry insights, call to action
- Facebook: Engaging, conversational, can be longer

Include emojis where appropriate. Focus on driving engagement.

Only generate post for the given platform. Don't generate posts for other platforms."""
                },
                {
                    'role': 'user',
                    'content': args['topic']
                }
            ]
        )

        generated_post = completion.choices[0].message.content
        logger.info(f"Generated {platform} post: {generated_post}")
        return generated_post or 'Failed to generate post'
    except Exception as e:
        logger.error(f"Failed to create social media post: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"Failed to generate post: {str(e)}"

async def analyze_engagement(params: Dict, messages: List[Dict[str, str]]) -> str:
    """Analyzes social media engagement metrics and provides recommendations."""
    try:
        args = params.get('args', {})
        logger.info(f"Analyzing engagement with args: {args}")
        
        client = openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        completion = await client.chat.completions.create(
            model='gpt-4',  # Changed from gpt-4o to gpt-4
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
                    'content': str(args)
                }
            ]
        )

        analysis = completion.choices[0].message.content
        logger.info(f"Generated engagement analysis: {analysis}")
        return analysis or 'Failed to analyze engagement'
    except Exception as e:
        logger.error(f"Failed to analyze engagement: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"Failed to analyze engagement: {str(e)}"

async def create_marketing_agent() -> Agent:
    """Create and configure the marketing agent."""
    # Read system prompt from file
    system_prompt_path = Path(__file__).parent / 'system.md'
    if not system_prompt_path.exists():
        raise FileNotFoundError("system.md not found in examples directory")

    marketing_manager = Agent(
        AgentOptions(
            system_prompt=system_prompt_path.read_text(),
            api_key=os.getenv('OPENSERV_API_KEY'),
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
    )

    # Add capabilities
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

    return marketing_manager

if __name__ == '__main__':
    import asyncio
    
    async def main():
        agent = await create_marketing_agent()
        await agent.start()
        
        try:
            # Keep the agent running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await agent.stop()

    asyncio.run(main())
