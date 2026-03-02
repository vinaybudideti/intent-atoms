"""
Intent Decomposer — Breaks complex queries into atomic, cacheable intents.

This is the key innovation: instead of caching at the query level (brittle, low hit rate)
or at the response level (context-dependent), we cache at the INTENT level.

"How do I deploy a React app with Docker on AWS?" becomes:
  → Atom 1: "Building a React app for production deployment"
  → Atom 2: "Containerizing a web application with Docker"
  → Atom 3: "Deploying Docker containers to AWS"
  
Each atom can be independently cached and reused across different queries.
"""

import json
import time
from typing import Optional
from .models import DecomposedQuery
from .providers import LLMProvider


DECOMPOSE_SYSTEM_PROMPT = """You are an intent decomposition engine. Your job is to break down user queries into atomic, independent intents.

RULES:
1. Each atom must be a SELF-CONTAINED question that can be answered independently.
2. Atoms should be as GENERIC as possible while preserving the original meaning.
   - BAD: "How to deploy my-specific-app.js to AWS" (too specific)
   - GOOD: "How to deploy a Node.js application to AWS" (generic, reusable)
3. Produce 1-5 atoms. Simple queries = 1 atom. Complex queries = more atoms.
4. Each atom needs: intent_text (the question), intent_label (snake_case identifier), domain_tags (topic categories).
5. If the query is already atomic (single clear intent), return it as a single atom.
6. Use CANONICAL phrasing for common intents. Always phrase atoms in this standard format:
   - "How to [verb] [technology/concept]"
   - NOT "Benefits of..." NOT "Best practices for..." NOT "Strategies for..."
   - Example: "How to deploy applications with Docker" (canonical)
   - NOT: "Docker deployment strategies" or "Benefits of Docker for deployment"
7. For deployment topics, always use: "How to deploy [thing] with/to [platform]"
8. For implementation topics, always use: "How to implement [thing] in [language/framework]"
9. For comparison topics, always use: "Comparing [A] vs [B]"
10. For explanation topics, always use: "How [thing] works"
11. DO NOT add atoms that weren't implied by the original query.

RESPOND WITH ONLY valid JSON, no markdown, no preamble:
{
  "atoms": [
    {
      "intent_text": "How to build a React application for production",
      "intent_label": "react_production_build",
      "domain_tags": ["react", "frontend", "deployment"]
    }
  ]
}"""


class IntentDecomposer:
    """Decomposes complex queries into atomic, cacheable intents."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def decompose(self, query: str) -> DecomposedQuery:
        """
        Break a user query into atomic intents.
        Uses the cheapest/fastest model available (Haiku or GPT-4o-mini).
        """
        start = time.time()
        
        # Use the decompose model (cheap + fast)
        model = getattr(self.provider, 'decompose_model', None)
        
        result = await self.provider.complete(
            prompt=f"Decompose this query into atomic intents:\n\n\"{query}\"",
            system=DECOMPOSE_SYSTEM_PROMPT,
            max_tokens=256,
            model=model,
        )

        # Parse the response
        try:
            text = result["text"].strip()
            # Handle potential markdown code blocks
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text)
            atoms = parsed.get("atoms", [])
        except (json.JSONDecodeError, KeyError, IndexError):
            # Fallback: treat the whole query as a single atom
            atoms = [{
                "intent_text": query,
                "intent_label": "full_query",
                "domain_tags": [],
            }]

        elapsed_ms = (time.time() - start) * 1000

        decomposed = DecomposedQuery(
            original_query=query,
            atoms=atoms,
            decomposition_tokens=result["input_tokens"] + result["output_tokens"],
            decomposition_cost=self.provider.estimate_cost(
                result["input_tokens"], result["output_tokens"], model=model
            ),
        )

        return decomposed

    async def decompose_batch(self, queries: list[str]) -> list[DecomposedQuery]:
        """Decompose multiple queries (useful for benchmarking)."""
        import asyncio
        return await asyncio.gather(*[self.decompose(q) for q in queries])
