"""
LLM Provider abstractions for Intent Atoms.
Supports Anthropic Claude and OpenAI GPT models.
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Optional

# Cost per 1M tokens (as of 2025)
PRICING = {
    "anthropic": {
        "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
        "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
        "claude-opus-4-6": {"input": 15.00, "output": 75.00},
    },
    "openai": {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    },
}


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(self, prompt: str, system: str = "", max_tokens: int = 1024) -> dict:
        """
        Returns: {"text": str, "input_tokens": int, "output_tokens": int}
        """
        pass

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD."""
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(
        self,
        api_key: str,
        decompose_model: str = "claude-haiku-4-5-20251001",
        compose_model: str = "claude-haiku-4-5-20251001",
        generate_model: str = "claude-sonnet-4-5-20250929",
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.decompose_model = decompose_model
        self.compose_model = compose_model
        self.generate_model = generate_model

    async def complete(self, prompt: str, system: str = "", max_tokens: int = 1024, model: str = None) -> dict:
        model = model or self.generate_model
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        
        response = await self.client.messages.create(**kwargs)
        return {
            "text": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        if not hasattr(self, '_embed_model'):
            from sentence_transformers import SentenceTransformer
            print("Loading embedding model...")
            self._embed_model = SentenceTransformer('all-mpnet-base-v2')
            print("Embedding model loaded!")
        embeddings = self._embed_model.encode(texts, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]

    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str = None) -> float:
        model = model or self.generate_model
        pricing = PRICING["anthropic"].get(model, {"input": 3.0, "output": 15.0})
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    def __init__(
        self,
        api_key: str,
        decompose_model: str = "gpt-4o-mini",
        compose_model: str = "gpt-4o-mini",
        generate_model: str = "gpt-4o",
        embed_model: str = "text-embedding-3-small",
    ):
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")

        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.decompose_model = decompose_model
        self.compose_model = compose_model
        self.generate_model = generate_model
        self.embed_model = embed_model

    async def complete(self, prompt: str, system: str = "", max_tokens: int = 1024, model: str = None) -> dict:
        model = model or self.generate_model
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens
        )
        return {
            "text": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(model=self.embed_model, input=texts)
        return [item.embedding for item in response.data]

    def estimate_cost(self, input_tokens: int, output_tokens: int, model: str = None) -> float:
        model = model or self.generate_model
        pricing = PRICING["openai"].get(model, {"input": 2.5, "output": 10.0})
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


def get_provider(name: str, api_key: str, **kwargs) -> LLMProvider:
    """Factory function to get an LLM provider."""
    providers = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
    }
    if name not in providers:
        raise ValueError(f"Unknown provider: {name}. Choose from: {list(providers.keys())}")
    return providers[name](api_key=api_key, **kwargs)
