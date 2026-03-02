from setuptools import setup, find_packages

setup(
    name="intent-atoms",
    version="0.1.0",
    description="Sub-query level intelligent caching for LLM APIs",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Vinay",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "anthropic>=0.84.0",
        "fastapi>=0.135.0",
        "uvicorn>=0.41.0",
        "pydantic>=2.12.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "openai": ["openai>=2.24.0"],
        "mongodb": ["motor>=3.7.0"],
        "voyage": ["voyageai>=0.3.7"],
        "dev": ["pytest>=9.0.0", "pytest-asyncio>=1.3.0", "httpx>=0.28.0"],
    },
    entry_points={
        "console_scripts": [
            "intent-atoms-server=api.server:main",
        ],
    },
)
