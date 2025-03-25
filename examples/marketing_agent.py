#!/usr/bin/env python3
"""
Marketing Agent Example

This example demonstrates a specialized marketing agent with social media capabilities.
"""

import os
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import AsyncOpenAI
import sys

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import from the local SDK
from openserv_sdk import Agent, AgentOptions
from openserv_sdk.logger import logger

# Load environment variables from .env file
load_dotenv()

# Check for required environment variables
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize OpenAI client
openai = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize the agent
marketing_manager = Agent(
    options=AgentOptions(
        system_prompt=Path(__file__).parent.joinpath("system.md").read_text(),
        api_key=os.environ.get("OPENSERV_API_KEY"),
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
)

# Define parameter models for capabilities
class SocialMediaPlatform(str):
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"

class CreateSocialMediaPostParams(BaseModel):
    platform: str = Field(..., description="The social media platform to create a post for")
    topic: str = Field(..., description="The topic to create a post about")

class EngagementMetrics(BaseModel):
    likes: int = Field(..., description="Number of likes")
    shares: int = Field(..., description="Number of shares")
    comments: int = Field(..., description="Number of comments")
    impressions: int = Field(..., description="Number of impressions")

class AnalyzeEngagementParams(BaseModel):
    platform: str = Field(..., description="The social media platform to analyze")
    metrics: EngagementMetrics = Field(..., description="Engagement metrics to analyze")

# Add capabilities to the agent
async def create_social_media_post(agent, params, messages):
    try:
        args = params["args"]
        
        completion = await agent.openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a marketing expert. Create a compelling {args.platform} post about: {args.topic}

Follow these platform-specific guidelines:
- Twitter: Max 280 characters, casual tone, use hashtags
- LinkedIn: Professional tone, industry insights, call to action
- Facebook: Engaging, conversational, can be longer

Include emojis where appropriate. Focus on driving engagement.

Only generate post for the given platform. Don't generate posts for other platforms.

Save the post in markdown format as a file and attach it to the task.
"""
                },
                {
                    "role": "user",
                    "content": args.topic
                }
            ]
        )
        
        generated_post = completion.choices[0].message.content
        logger.info(f"Generated {args.platform} post: {generated_post}")
        
        if not generated_post:
            logger.error("Failed to generate post")
            return "Failed to generate post"
            
        return generated_post
    except Exception as e:
        logger.error(f"Error in create_social_media_post: {str(e)}")
        return f"Error generating post: {str(e)}"

async def analyze_engagement(agent, params, messages):
    try:
        args = params["args"]
        
        completion = await agent.openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """You are a social media analytics expert. Analyze the engagement metrics and provide actionable recommendations.

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
                    "role": "user",
                    "content": str(args.model_dump())
                }
            ]
        )
        
        analysis = completion.choices[0].message.content
        logger.info(f"Generated engagement analysis for {args.platform}: {analysis}")
        
        if not analysis:
            logger.error("Failed to analyze engagement")
            return "Failed to analyze engagement"
            
        return analysis
    except Exception as e:
        logger.error(f"Error in analyze_engagement: {str(e)}")
        return f"Error analyzing engagement: {str(e)}"

marketing_manager.add_capabilities([
    {
        "name": "createSocialMediaPost",
        "description": "Creates a social media post for the specified platform",
        "schema": CreateSocialMediaPostParams,
        "run": create_social_media_post
    },
    {
        "name": "analyzeEngagement",
        "description": "Analyzes social media engagement metrics and provides recommendations",
        "schema": AnalyzeEngagementParams,
        "run": analyze_engagement
    }
])

# Start the agent
if __name__ == "__main__":
    try:
        logger.info("Starting marketing agent...")
        asyncio.run(marketing_manager.start())
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        raise 