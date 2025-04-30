# How to Run Examples

This guide will walk you through setting up and running the example agents in this repository.

## Prerequisites

1. Expose your local server:
During development, OpenServ needs to reach your agent running on your computer. Since your computer doesn't have a public internet address, we'll use a tunneling tool.

What is tunneling? It creates a temporary secure pathway from the internet to your computer, allowing OpenServ to send requests to your agent while you develop it.

Choose one option:
- ngrok (recommended for beginners)
- localtunnel (open source option)

### Quick start with ngrok:
1. Download and install ngrok
2. Open your terminal and run:
```bash
ngrok http 7378  # Use your actual port number if different
```
3. Look for a line like `Forwarding https://abc123.ngrok-free.app -> http://localhost:7378`
4. Copy the https URL (e.g., https://abc123.ngrok-free.app) - you'll need this later

## Python Setup

1. Create and activate a virtual environment:
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

2. Install the required packages:
```bash
# Install the OpenServ SDK and dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade openai
```

## 2. Create an account on OpenServ and set up your developer account
1. Create a developer account on OpenServ
2. Navigate to the Developer menu on the left sidebar
3. Click on Profile to set up your account as a developer on the platform

## 3. Register your agent
To begin developing an agent for OpenServ, you must first register it:

1. Navigate to the Developer sidebar menu
2. Click on Add Agent
3. Add details about your agent:
   - Agent Name: Choose a descriptive name
   - Agent Endpoint: Add the tunneling URL from step 1 as the agent's endpoint URL
   - Capabilities Description: Add your agent capabilities

## 4. Create a Secret (API) Key for your Agent
Note that every agent has its own API Key

1. Navigate to Developer sidebar menu -> Your Agents
2. Open the Details of the agent for which you wish to generate a secret key
3. Click on Create Secret Key
4. Store this key securely as it will be required to authenticate your agent's requests with the OpenServ API

## 5. Set Up Your Environment
Add your secret keys to your environment variables or to an .env file on your project root:

```bash
export OPENSERV_API_KEY=your_api_key_here
export OPENAI_API_KEY=your_openai_api_key_here # Required for testing agent locally with your own LLM API Key
```

## Running the Examples

### Basic Agent Example
This example demonstrates the fundamental capabilities of the OpenServ SDK and how to create a simple agent.

1. Navigate to the examples directory and run the basic agent:
```bash
python3 examples/basic_agent.py
```

2. Create a new project at OpenServ, choose your agent and add the following project prompt:
```
Hi, I am Me. Greet me and say goodbye.
```

### Marketing Agent Example
This example demonstrates a specialized marketing agent with social media capabilities.

1. Navigate to the examples directory and Run the marketing agent:
```bash
python3 examples/marketing_agent.py
```

2. Create a new project at OpenServ, choose your agent and add the following project prompt. When creating the agent, add Twitter integration to your agent. 
```
First, get my Twitter account information. Then create a compelling tweet about our new AI-driven marketing automation platform that increases engagement by 45%. After that, send this tweet to my Twitter account. Finally, analyze these engagement metrics: 25 likes, 12 shares, 8 comments, and 400 impressions to provide recommendations for improving future performance.
```


## Local Testing

You can test your agents locally using the `process()` method before deploying them to the OpenServ platform:

### Using process() for Local Testing

The `process()` method allows you to test your agent's capabilities locally using OpenAI's API without needing to deploy to the OpenServ platform:

```python
from openserv.types import ProcessParams

result = await agent.process(ProcessParams(
    messages=[
        {"role": "user", "content": "Your message here"}
    ]
))

# The response contains the model's reply
response = result["response"]
print(response)
```

This approach is useful for:
1. Rapid development and testing of agent capabilities
2. Debugging your agent's behavior with different inputs
3. Testing how your agent handles different types of requests
4. Verifying that your capabilities work as expected before deployment

**Note:** Using the `process()` method requires a valid `OPENAI_API_KEY` to be set in your environment variables.

## Troubleshooting

If you encounter any issues:

1. Ensure all environment variables are properly set
2. Check that your ngrok tunnel is active and the URL is correct
3. Verify that your agent is properly registered on OpenServ
4. Check the logs for any error messages
5. Make sure you have all required dependencies installed

## Additional Notes

- The examples use different capabilities and integrations to demonstrate various features of the OpenServ SDK
- Each example can be modified and extended based on your specific needs
- Remember to keep your API keys secure and never commit them to version control
- The system prompts and capabilities can be customized by modifying the respective files 