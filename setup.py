from setuptools import setup, find_packages

setup(
    name="openserv_sdk",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "openai>=1.3.0",
        "pydantic>=2.4.2",
        "httpx>=0.25.0",
        "python-dotenv>=1.0.0",
        "aiohttp>=3.8.0",
        "setuptools>=42.0.0"
    ],
    python_requires=">=3.8",
)
