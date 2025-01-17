import os
import pytest
from unittest.mock import patch, MagicMock

from src import Agent
from examples.marketing_agent import create_marketing_agent, create_social_media_post, analyze_engagement, SocialMediaPostParams, AnalyzeEngagementParams, EngagementMetrics

@pytest.fixture
def mock_openai():
    with patch('openai.OpenAI') as mock:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='Test response',
                        role='assistant'
                    )
                )
            ]
        )
        mock.return_value = mock_client
        yield mock

@pytest.fixture
def mock_agent():
    with patch('src.agent.Agent') as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance

@pytest.mark.asyncio
async def test_create_social_media_post(mock_openai):
    """Test creating a social media post."""
    params = SocialMediaPostParams(
        platform='twitter',
        topic='coding schools'
    )
    messages = [
        {"role": "user", "content": "Write a tweet about coding schools"}
    ]

    # Run the function
    result = await create_social_media_post(params, messages)

    # Verify OpenAI was called correctly
    mock_openai.assert_called_once()
    call_args = mock_openai.chat.completions.create.call_args[1]
    assert call_args['model'] == 'gpt-4'
    assert len(call_args['messages']) == 2  # Initial message + prompt
    assert call_args['messages'][1]['role'] == 'user'
    assert 'Create a twitter post about coding schools' in call_args['messages'][1]['content']
    
    # Verify result
    assert result == 'Test response'

@pytest.mark.asyncio
async def test_analyze_engagement(mock_openai, mock_agent):
    """Test analyzing engagement metrics."""
    params = {
        'args': AnalyzeEngagementParams(
            platform='twitter',
            metrics=EngagementMetrics(
                likes=100,
                shares=50,
                comments=25,
                impressions=1000
            )
        )
    }

    # Run the function
    result = await analyze_engagement(mock_agent, params)

    # Verify OpenAI was called correctly
    mock_openai.assert_called_once()
    call_args = mock_openai.chat.completions.create.call_args[1]
    assert call_args['model'] == 'gpt-4o'
    assert len(call_args['messages']) == 2  # System prompt + metrics
    assert call_args['messages'][0]['role'] == 'system'
    assert 'social media analytics expert' in call_args['messages'][0]['content']
    assert call_args['messages'][1]['role'] == 'user'
    
    # Verify result
    assert result == 'Test response'

def test_create_marketing_agent():
    """Test creating a marketing agent instance."""
    with patch('pathlib.Path.read_text') as mock_read:
        mock_read.return_value = "Test system prompt"
        
        agent = create_marketing_agent()
        
        # Verify agent was created with correct options
        assert isinstance(agent, Agent)
        assert agent.config.system_prompt == "Test system prompt"
        assert agent.config.api.api_key == os.getenv('OPENSERV_API_KEY')
        assert agent.config.openai.api_key == os.getenv('OPENAI_API_KEY')
        
        # Verify capabilities were added
        tools = agent.tools
        assert len(tools) == 2
        
        # Verify createSocialMediaPost capability
        social_media_cap = next(t for t in tools if t.name == 'createSocialMediaPost')
        assert social_media_cap.description == 'Creates a social media post for the specified platform'
        assert social_media_cap.schema == SocialMediaPostParams
        assert social_media_cap.run == create_social_media_post
        
        # Verify analyzeEngagement capability
        analyze_cap = next(t for t in tools if t.name == 'analyzeEngagement')
        assert analyze_cap.description == 'Analyzes social media engagement metrics and provides recommendations'
        assert analyze_cap.schema == AnalyzeEngagementParams
        assert analyze_cap.run == analyze_engagement

@pytest.mark.asyncio
async def test_handle_empty_openai_response(mock_openai):
    """Test handling empty OpenAI response."""
    mock_openai.chat.completions.create.return_value = MagicMock(choices=[])
    
    params = SocialMediaPostParams(
        platform='twitter',
        topic='coding schools'
    )
    messages = [
        {"role": "user", "content": "Write a tweet"}
    ]

    # Run the function and verify it handles empty response
    result = await create_social_media_post(params, messages)
    assert result == 'Failed to generate post'

@pytest.mark.asyncio
async def test_handle_missing_message_content(mock_openai):
    """Test handling missing message content in OpenAI response."""
    mock_openai.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=None,
                    role='assistant'
                )
            )
        ]
    )
    
    params = SocialMediaPostParams(
        platform='twitter',
        topic='coding schools'
    )
    messages = [
        {"role": "user", "content": "Write a tweet"}
    ]

    # Run the function and verify it handles missing content
    result = await create_social_media_post(params, messages)
    assert result == 'Failed to generate post' 
