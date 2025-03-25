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
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

2. Install the required packages:
```bash
# Install the OpenServ SDK in editable mode
pip install -e .
```

3. Verify the installation:
```bash
# Check if the packages are installed correctly
pip list | grep -E "openserv-sdk|python-dotenv|openai"
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
   - Capabilities Description: Describe what your agent can do

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
export OPENAI_API_KEY=your_openai_api_key_here
```

## Running the Examples

### Marketing Agent Example
This example demonstrates a specialized marketing agent with social media capabilities.

1. Navigate to the examples directory:
```bash
cd examples
```

2. Run the marketing agent:
```bash
python marketing_agent.py
```

3. Test the agent with the following prompt:
```
Create a social media post for Twitter about the launch of our new AI-powered productivity tool. Then analyze the engagement metrics with 1000 likes, 200 shares, 50 comments, and 5000 impressions.
```

### Custom Agent Example
This example shows how to create a custom agent with specialized response behavior.

1. Navigate to the examples directory:
```bash
cd examples
```

2. Run the custom agent:
```bash
python custom_agent.py
```

3. Test the agent with the following prompt:
```
Hello, can you help me with a task?
```

### Twitter Agent Example
This example demonstrates an agent specifically designed to interact with Twitter.

1. Navigate to the examples directory:
```bash
cd examples
```

2. Run the Twitter agent:
```bash
python twitter_agent.py
```

3. Test the agent with the following prompt:
```
Get my Twitter account information and then send a marketing tweet about our new product launch.
```

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