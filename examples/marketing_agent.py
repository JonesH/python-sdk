import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
from pydantic import BaseModel
import openai
import logging

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src import Agent, AgentOptions
from src.capability import Capability

load_dotenv()

logger = logging.getLogger(__name__)

if not os.getenv('OPENAI_API_KEY'):
    raise ValueError('OPENAI_API_KEY environment variable is required')

openai_client = openai.OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)

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

async def create_social_media_post(params: SocialMediaPostParams, messages: List[Dict[str, str]]) -> str:
    """Creates a social media post for the specified platform."""
    platform = params.platform
    topic = params.topic

    prompt = f"Create a {platform} post about {topic}. Keep it engaging and concise."
    messages.append({"role": "user", "content": prompt})

    completion = openai_client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    return completion.choices[0].message.content

async def analyze_engagement(
    agent: Agent,
    params: Dict[str, Any]
) -> str:
    """Analyze social media engagement metrics."""
    args = params.get('args')
    if not isinstance(args, AnalyzeEngagementParams):
        args = AnalyzeEngagementParams.model_validate(args)

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
                'content': str(args.model_dump())
            }
        ]
    )

    if not completion.choices or not completion.choices[0].message:
        return 'Failed to analyze engagement'

    analysis = completion.choices[0].message.content
    if not analysis:
        return 'Failed to analyze engagement'

    logger.info("Generated engagement analysis for %s: %s", args.platform, analysis)
    return analysis

def create_marketing_agent() -> Agent:
    """Create and configure the marketing agent."""
    system_prompt_path = Path(__file__).parent.joinpath('system.md')
    if not system_prompt_path.exists():
        raise FileNotFoundError("system.md not found in examples directory")

    marketing_manager = Agent(
        AgentOptions(
            system_prompt=system_prompt_path.read_text(),
            api_key=os.getenv('OPENSERV_API_KEY'),
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
    )

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
    agent = create_marketing_agent()
    agent.start()
