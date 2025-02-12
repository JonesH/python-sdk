from setuptools import setup, find_packages

setup(
    name="openserv_sdk",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pydantic>=2.0.0",
        "openai>=1.0.0",
        "aiohttp>=3.8.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "httpx>=0.25.0",
        "python-dotenv>=1.0.0"
    ],
    python_requires=">=3.8",
    author="OpenServ Labs",
    author_email="support@openserv.ai",
    description="OpenServ Agent library for building AI agents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/openserv-labs/openserv-sdk-python",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
