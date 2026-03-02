"""
Intent Atoms Engine v2 — Full-query FAISS caching.

Simplified pipeline:
  Query -> Embed (local MPNet) -> FAISS search -> If hit: return cached -> If miss: LLM call, cache

Cache hits cost ZERO LLM tokens. No decomposition, no composition.
"""

import time
from typing import Optional
from .models import QueryResult, CacheStats
from .faiss_store import FAISSStore
from .providers import LLMProvider, get_provider


GENERATE_SYSTEM_PROMPT = """You are a helpful assistant. Provide a clear, comprehensive answer to the user's question. Be concise but thorough."""


class IntentAtomsEngineV2:
    """
    v2 engine with FAISS-based full-query caching.

    Usage:
        engine = IntentAtomsEngineV2(
            llm_provider="anthropic",
            api_key="sk-...",
            persist_dir="./data/faiss_cache",
        )

        result = await engine.query("How do I deploy React with Docker?")
        # First call: LLM generates, caches result
        # Second similar call: returns cached response (ZERO cost)
    """

    def __init__(
        self,
        llm_provider: str = "anthropic",
        api_key: str = "",
        persist_dir: Optional[str] = "./data/faiss_cache",
        similarity_threshold: float = 0.83,
        embedding_dimension: int = 768,
    ):
        self.provider = get_provider(llm_provider, api_key=api_key)

        self.store = FAISSStore(
            dimension=embedding_dimension,
            persist_dir=persist_dir,
        )

        self.similarity_threshold = similarity_threshold
        self._query_count = 0

    async def query(self, user_query: str) -> QueryResult:
        """Process a user query with FAISS caching."""
        total_start = time.time()
        result = QueryResult(original_query=user_query)

        # Step 1: Embed the query
        embed_start = time.time()
        embeddings = await self.provider.embed([user_query])
        query_embedding = embeddings[0]
        result.embedding_time_ms = (time.time() - embed_start) * 1000

        # Step 2: Search FAISS
        search_start = time.time()
        search_results = await self.store.search(
            embedding=query_embedding,
            top_k=1,
            threshold=self.similarity_threshold,
        )
        result.search_time_ms = (time.time() - search_start) * 1000
        result.matching_time_ms = result.search_time_ms

        if search_results:
            # Cache HIT
            cached_entry, similarity = search_results[0]

            result.response = cached_entry.response_text
            result.is_cache_hit = True
            result.matched_query = cached_entry.query_text
            result.similarity_score = similarity

            result.total_atoms = 1
            result.cache_hits = 1
            result.cache_misses = 0
            result.tokens_saved = cached_entry.output_tokens + cached_entry.input_tokens
            result.total_tokens_used = 0
            result.estimated_cost = 0.0
            result.estimated_cost_without_cache = cached_entry.generation_cost

            result.decomposition_time_ms = 0.0
            result.generation_time_ms = 0.0
            result.composition_time_ms = 0.0

            await self.store.log_query({
                "query": user_query,
                "is_cache_hit": True,
                "similarity_score": similarity,
                "matched_query": cached_entry.query_text,
                "tokens_saved": result.tokens_saved,
                "cost_saved": cached_entry.generation_cost,
            })

        else:
            # Cache MISS
            gen_start = time.time()

            model = getattr(self.provider, 'generate_model', None)
            llm_result = await self.provider.complete(
                prompt=user_query,
                system=GENERATE_SYSTEM_PROMPT,
                max_tokens=1024,
                model=model,
            )

            result.generation_time_ms = (time.time() - gen_start) * 1000

            response_text = llm_result["text"]
            input_tokens = llm_result["input_tokens"]
            output_tokens = llm_result["output_tokens"]
            generation_cost = self.provider.estimate_cost(
                input_tokens, output_tokens, model=model
            )

            result.response = response_text
            result.is_cache_hit = False
            result.matched_query = ""
            result.similarity_score = 0.0

            result.total_atoms = 1
            result.cache_hits = 0
            result.cache_misses = 1
            result.tokens_saved = 0
            result.total_tokens_used = input_tokens + output_tokens
            result.estimated_cost = generation_cost
            result.estimated_cost_without_cache = generation_cost

            result.decomposition_time_ms = 0.0
            result.composition_time_ms = 0.0

            # Cache the result
            await self.store.store(
                embedding=query_embedding,
                query_text=user_query,
                response_text=response_text,
                metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "generation_cost": generation_cost,
                    "model_used": model or "",
                },
            )

            await self.store.log_query({
                "query": user_query,
                "is_cache_hit": False,
                "tokens_used": input_tokens + output_tokens,
                "cost": generation_cost,
            })

        result.total_time_ms = (time.time() - total_start) * 1000
        self._query_count += 1
        return result

    async def get_stats(self) -> CacheStats:
        """Get overall cache performance statistics."""
        return await self.store.get_stats()

    async def clear_cache(self) -> None:
        """Clear all cached queries."""
        await self.store.clear()

    async def evict_stale(self, max_age_days: int = 30) -> int:
        """Remove stale cached queries."""
        return await self.store.evict_stale(max_age_days)
