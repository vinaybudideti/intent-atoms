"""
Response Composer — Stitches cached atom fragments and new generations 
into a single coherent response.

This is where the cost savings happen. Instead of generating a full response
from scratch (expensive), the composer takes pre-generated fragments and 
uses a cheap model to weave them together contextually.

The LLM goes from being a GENERATOR to being a COMPOSER.
"""

import json
import time
from .models import Atom, MatchResult
from .providers import LLMProvider


COMPOSE_SYSTEM_PROMPT = """You are a response composer. You receive a user's original question and a set of answer fragments (atoms). Your job is to weave these fragments into a single, natural, coherent response.

RULES:
1. Use ALL provided fragments — don't drop information.
2. The response should flow naturally, as if written from scratch.
3. Add transitions between fragments so the answer reads smoothly.
4. Adapt the tone to match the original question (technical, casual, etc.).
5. If fragments overlap, merge them without repetition.
6. Keep the response concise — don't add filler.
7. Preserve any code snippets, commands, or examples from fragments exactly as-is.

Write the composed response directly. No preamble, no "Here's your answer:", just the answer."""


GENERATE_SYSTEM_PROMPT = """You are a helpful assistant. Answer the following specific question concisely and accurately. 
Focus ONLY on this exact question — don't address other topics.
Keep your response focused and under 300 words unless the topic requires more detail."""


class ResponseComposer:
    """Composes final responses from cached and newly generated atom fragments."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    async def generate_atom(self, intent_text: str, domain_tags: list[str] = None) -> tuple[str, int, int]:
        """
        Generate a response for a cache-miss atom.
        Returns: (response_text, input_tokens, output_tokens)
        """
        model = getattr(self.provider, 'generate_model', None)
        
        result = await self.provider.complete(
            prompt=intent_text,
            system=GENERATE_SYSTEM_PROMPT,
            max_tokens=400,
            model=model,
        )
        return result["text"], result["input_tokens"], result["output_tokens"]

    async def compose(
        self,
        original_query: str,
        fragments: list[dict],  # [{"intent": str, "response": str, "source": "cache"|"generated"}]
    ) -> tuple[str, int, int]:
        """
        Compose fragments into a final coherent response.
        
        If there's only one fragment, return it directly (no composition needed).
        Uses the cheapest model for composition since it's just arranging text.
        
        Returns: (composed_response, input_tokens, output_tokens)
        """
        # Single fragment — no composition needed
        if len(fragments) == 1:
            return fragments[0]["response"], 0, 0

        # Build the composition prompt
        fragment_text = "\n\n".join(
            f"--- Fragment {i+1}: {f['intent']} ---\n{f['response']}"
            for i, f in enumerate(fragments)
        )

        prompt = f"""Original question: "{original_query}"

Answer fragments to compose:

{fragment_text}

Compose these into a single natural response:"""

        model = getattr(self.provider, 'compose_model', None)
        
        result = await self.provider.complete(
            prompt=prompt,
            system=COMPOSE_SYSTEM_PROMPT,
            max_tokens=800,
            model=model,
        )
        return result["text"], result["input_tokens"], result["output_tokens"]
