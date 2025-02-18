import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from openai import AsyncOpenAI
from pathlib import Path

from openserv_sdk import Agent, AgentOptions, Capability
from openserv_sdk.types import SocialMediaPostParams, AnalyzeEngagementParams, EngagementMetrics

@pytest.fixture
def mock_openai():
    """Mock AsyncOpenAI client for testing."""
    with patch('openai.AsyncOpenAI') as mock:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = AsyncMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='Generated social media post content',
                        role='assistant'
                    )
                )
            ]
        )
        mock.return_value = mock_client
        yield mock

@pytest.fixture
def mock_agent():
    with patch('openserv_sdk.agent.Agent') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance

@pytest.mark.asyncio
async def test_create_social_media_post(mock_openai):
    """Test creating a social media post."""
    params = {
        "args": {
            "platform": "twitter",
            "topic": "coding schools"
        }
    }
    messages = [
        {"role": "user", "content": "Write a tweet about coding schools"}
    ]

    agent = Agent(AgentOptions(
        system_prompt="You are a marketing expert",
        api_key="test-key",
        openai_api_key="test-openai-key"
    ))

    async def test_run(params, messages):
        completion = await mock_openai.return_value.chat.completions.create(
            model='gpt-4',
            messages=[
                {
                    'role': 'system',
                    'content': f"You are a marketing expert. Create a compelling {params['args']['platform']} post about: {params['args']['topic']}"
                },
                {
                    'role': 'user',
                    'content': params['args']['topic']
                }
            ]
        )
        return completion.choices[0].message.content

    capability = Capability(
        name="createSocialMediaPost",
        description="Creates a social media post for the specified platform",
        schema=SocialMediaPostParams,
        run=test_run
    )
    agent.add_capability(capability)

    result = await agent.handle_tool_route(
        "createSocialMediaPost",
        {"args": params["args"], "messages": messages, "action": None}
    )
    assert result == "Generated social media post content"

@pytest.mark.asyncio
async def test_analyze_engagement(mock_openai):
    """Test analyzing engagement metrics."""
    params = {
        "args": {
            "platform": "twitter",
            "metrics": {
                "likes": 100,
                "shares": 50,
                "comments": 25,
                "impressions": 1000
            }
        }
    }
    messages = []

    agent = Agent(AgentOptions(
        system_prompt="You are a marketing expert",
        api_key="test-key",
        openai_api_key="test-openai-key"
    ))

    async def test_run(params, messages):
        completion = await mock_openai.return_value.chat.completions.create(
            model='gpt-4',
            messages=[
                {
                    'role': 'system',
                    'content': "You are a social media analytics expert"
                },
                {
                    'role': 'user',
                    'content': str(params["args"])
                }
            ]
        )
        return completion.choices[0].message.content

    capability = Capability(
        name="analyzeEngagement",
        description="Analyzes social media engagement metrics",
        schema=AnalyzeEngagementParams,
        run=test_run
    )
    agent.add_capability(capability)

    result = await agent.handle_tool_route(
        "analyzeEngagement",
        {"args": params["args"], "messages": messages, "action": None}
    )
    assert result == "Generated social media post content"

@pytest.mark.asyncio
async def test_handle_empty_openai_response(mock_openai):
    """Test handling empty OpenAI response."""
    mock_openai.return_value.chat.completions.create.return_value = AsyncMock(choices=[])
    
    params = {
        "args": {
            "platform": "twitter",
            "topic": "coding schools"
        }
    }
    messages = []

    agent = Agent(AgentOptions(
        system_prompt="You are a marketing expert",
        api_key="test-key",
        openai_api_key="test-openai-key"
    ))

    async def test_run(params, messages):
        completion = await mock_openai.return_value.chat.completions.create(
            model='gpt-4',
            messages=[
                {
                    'role': 'system',
                    'content': f"You are a marketing expert. Create a compelling {params['args']['platform']} post about: {params['args']['topic']}"
                },
                {
                    'role': 'user',
                    'content': params['args']['topic']
                }
            ]
        )
        return completion.choices[0].message.content if completion.choices else "Failed to generate post"

    capability = Capability(
        name="createSocialMediaPost",
        description="Creates a social media post for the specified platform",
        schema=SocialMediaPostParams,
        run=test_run
    )
    agent.add_capability(capability)

    result = await agent.handle_tool_route(
        "createSocialMediaPost",
        {"args": params["args"], "messages": messages, "action": None}
    )
    assert result == "Failed to generate post"

@pytest.mark.asyncio
async def test_handle_missing_message_content(mock_openai):
    """Test handling missing message content in OpenAI response."""
    mock_openai.return_value.chat.completions.create.return_value = AsyncMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=None,
                    role="assistant"
                )
            )
        ]
    )
    
    params = {
        "args": {
            "platform": "twitter",
            "topic": "coding schools"
        }
    }
    messages = []

    agent = Agent(AgentOptions(
        system_prompt="You are a marketing expert",
        api_key="test-key",
        openai_api_key="test-openai-key"
    ))

    async def test_run(params, messages):
        completion = await mock_openai.return_value.chat.completions.create(
            model='gpt-4',
            messages=[
                {
                    'role': 'system',
                    'content': f"You are a marketing expert. Create a compelling {params['args']['platform']} post about: {params['args']['topic']}"
                },
                {
                    'role': 'user',
                    'content': params['args']['topic']
                }
            ]
        )
        return completion.choices[0].message.content or "Failed to generate post"

    capability = Capability(
        name="createSocialMediaPost",
        description="Creates a social media post for the specified platform",
        schema=SocialMediaPostParams,
        run=test_run
    )
    agent.add_capability(capability)

    result = await agent.handle_tool_route(
        "createSocialMediaPost",
        {"args": params["args"], "messages": messages, "action": None}
    )
    assert result == "Failed to generate post"

@pytest.mark.asyncio
async def test_create_social_media_post_api_error(mock_openai):
    """Test handling OpenAI API error in create_social_media_post."""
    mock_openai.return_value.chat.completions.create.side_effect = Exception("API Error")
    
    params = {
        "args": {
            "platform": "twitter",
            "topic": "coding schools"
        }
    }
    messages = []

    agent = Agent(AgentOptions(
        system_prompt="You are a marketing expert",
        api_key="test-key",
        openai_api_key="test-openai-key"
    ))

    async def test_run(params, messages):
        try:
            completion = await mock_openai.return_value.chat.completions.create(
                model='gpt-4',
                messages=[
                    {
                        'role': 'system',
                        'content': f"You are a marketing expert. Create a compelling {params['args']['platform']} post about: {params['args']['topic']}"
                    },
                    {
                        'role': 'user',
                        'content': params['args']['topic']
                    }
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Failed to generate post: {str(e)}"

    capability = Capability(
        name="createSocialMediaPost",
        description="Creates a social media post for the specified platform",
        schema=SocialMediaPostParams,
        run=test_run
    )
    agent.add_capability(capability)

    result = await agent.handle_tool_route(
        "createSocialMediaPost",
        {"args": params["args"], "messages": messages, "action": None}
    )
    assert result == "Failed to generate post: API Error"

@pytest.mark.asyncio
async def test_analyze_engagement_api_error(mock_openai):
    """Test handling OpenAI API error in analyze_engagement."""
    mock_openai.return_value.chat.completions.create.side_effect = Exception("API Error")
    
    params = {
        "args": {
            "platform": "twitter",
            "metrics": {
                "likes": 100,
                "shares": 50,
                "comments": 25,
                "impressions": 1000
            }
        }
    }
    messages = []

    agent = Agent(AgentOptions(
        system_prompt="You are a marketing expert",
        api_key="test-key",
        openai_api_key="test-openai-key"
    ))

    async def test_run(params, messages):
        try:
            completion = await mock_openai.return_value.chat.completions.create(
                model='gpt-4',
                messages=[
                    {
                        'role': 'system',
                        'content': "You are a social media analytics expert"
                    },
                    {
                        'role': 'user',
                        'content': str(params["args"])
                    }
                ]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Failed to analyze engagement: {str(e)}"

    capability = Capability(
        name="analyzeEngagement",
        description="Analyzes social media engagement metrics",
        schema=AnalyzeEngagementParams,
        run=test_run
    )
    agent.add_capability(capability)

    result = await agent.handle_tool_route(
        "analyzeEngagement",
        {"args": params["args"], "messages": messages, "action": None}
    )
    assert result == "Failed to analyze engagement: API Error" 
