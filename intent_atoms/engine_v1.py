"""
Intent Atoms Engine — The main orchestrator.

This is the drop-in replacement for direct LLM API calls.
Instead of: response = llm.complete(query)
Use:        result = engine.query(query)

The engine handles the full pipeline:
  Query → Decompose → Match → Generate (misses only) → Compose → Cache → Return

Cost savings come from:
  1. Skipping generation for cached atoms (biggest saving)
  2. Using cheap models for decomposition and composition
  3. Progressively better cache hit rates over time
"""

import time
import asyncio
from typing import Optional
from .models import Atom, QueryResult, DecomposedQuery, MatchResult, CacheStats
from .decomposer import IntentDecomposer
from .matcher import SimilarityMatcher
from .composer import ResponseComposer
from .atom_store import AtomStore, get_store
from .providers import LLMProvider, get_provider


class IntentAtomsEngine:
    """
    Main engine for Intent Atoms sub-query caching system.
    
    Usage:
        engine = IntentAtomsEngine(
            llm_provider="anthropic",
            api_key="sk-...",
            store_backend="local",
            persist_path="./atom_cache.json",
        )
        
        result = await engine.query("How do I deploy React with Docker on AWS?")
        print(result.response)
        print(f"Saved {result.cost_savings_pct:.1f}% in API costs")
    """

    def __init__(
        self,
        llm_provider: str = "anthropic",
        api_key: str = "",
        store_backend: str = "local",
        similarity_threshold: float = 0.82,
        persist_path: Optional[str] = "./data/atom_cache.json",
        **store_kwargs,
    ):
        # Initialize LLM provider
        self.provider = get_provider(llm_provider, api_key=api_key)

        # Initialize components
        self.store = get_store(
            store_backend,
            persist_path=persist_path,
            **store_kwargs,
        )
        self.decomposer = IntentDecomposer(self.provider)
        self.matcher = SimilarityMatcher(
            self.store,
            self.provider,
            strong_threshold=similarity_threshold,
        )
        self.composer = ResponseComposer(self.provider)

        # Stats tracking
        self._query_count = 0

    async def query(self, user_query: str) -> QueryResult:
        """
        Process a user query through the Intent Atoms pipeline.
        
        Returns a QueryResult with the response, cost analysis, and cache stats.
        """
        total_start = time.time()
        result = QueryResult(original_query=user_query)

        # ─── Step 1: Decompose ───
        decomp_start = time.time()
        decomposed = await self.decomposer.decompose(user_query)
        result.decomposition_time_ms = (time.time() - decomp_start) * 1000
        result.total_atoms = decomposed.num_atoms

        # ─── Step 2: Match against cache ───
        match_start = time.time()
        intent_texts = [a["intent_text"] for a in decomposed.atoms]
        match_results = await self.matcher.match_batch(intent_texts)
        result.matching_time_ms = (time.time() - match_start) * 1000

        # ─── Step 3: Generate missing atoms ───
        gen_start = time.time()
        fragments = []
        total_gen_input = 0
        total_gen_output = 0

        for atom_info, match in zip(decomposed.atoms, match_results):
            if match.is_cache_hit and match.matched_atom:
                # Cache hit — reuse the cached response
                result.cache_hits += 1
                result.atoms_reused.append(match.matched_atom)
                result.tokens_saved += match.matched_atom.token_count
                fragments.append({
                    "intent": atom_info["intent_text"],
                    "response": match.matched_atom.response_fragment,
                    "source": "cache",
                })
            else:
                # Cache miss — generate new response
                result.cache_misses += 1
                response_text, in_tok, out_tok = await self.composer.generate_atom(
                    atom_info["intent_text"],
                    atom_info.get("domain_tags", []),
                )
                total_gen_input += in_tok
                total_gen_output += out_tok

                # Create and cache the new atom
                embeddings = await self.provider.embed([atom_info["intent_text"]])
                new_atom = Atom(
                    intent_label=atom_info.get("intent_label", ""),
                    intent_text=atom_info["intent_text"],
                    response_fragment=response_text,
                    embedding=embeddings[0],
                    token_count=out_tok,
                    domain_tags=atom_info.get("domain_tags", []),
                )
                await self.store.store(new_atom)
                result.atoms_generated.append(new_atom)

                fragments.append({
                    "intent": atom_info["intent_text"],
                    "response": response_text,
                    "source": "generated",
                })

        result.generation_time_ms = (time.time() - gen_start) * 1000

        # ─── Step 4: Compose final response ───
        compose_start = time.time()
        composed_text, comp_in, comp_out = await self.composer.compose(
            user_query, fragments
        )
        result.composition_time_ms = (time.time() - compose_start) * 1000
        result.response = composed_text

        # ─── Step 5: Calculate costs ───
        # Actual cost (with caching)
        decomp_cost = decomposed.decomposition_cost
        gen_cost = self.provider.estimate_cost(total_gen_input, total_gen_output)
        compose_cost = self.provider.estimate_cost(comp_in, comp_out) if comp_in > 0 else 0
        result.estimated_cost = decomp_cost + gen_cost + compose_cost
        result.total_tokens_used = (
            decomposed.decomposition_tokens + total_gen_input + total_gen_output + comp_in + comp_out
        )

        # Hypothetical cost without caching (full generation for all atoms)
        # Estimate: each atom would need ~100 input + ~300 output tokens via the expensive model
        avg_gen_tokens = 400
        result.estimated_cost_without_cache = self.provider.estimate_cost(
            100 * result.total_atoms,
            300 * result.total_atoms,
        )
        # Add decomposition overhead to actual cost comparison
        # If no atoms were generated, the "without cache" cost should reflect full generation
        if result.cache_hits > 0 and result.estimated_cost_without_cache < result.estimated_cost:
            result.estimated_cost_without_cache = result.estimated_cost * 1.5

        result.total_time_ms = (time.time() - total_start) * 1000

        # ─── Step 6: Log for analytics ───
        if hasattr(self.store, 'log_query'):
            await self.store.log_query({
                "query": user_query,
                "num_atoms": result.total_atoms,
                "cache_hits": result.cache_hits,
                "cache_misses": result.cache_misses,
                "tokens_saved": result.tokens_saved,
                "cost_saved": max(0, result.estimated_cost_without_cache - result.estimated_cost),
                "total_time_ms": result.total_time_ms,
            })

        self._query_count += 1
        return result

    async def get_stats(self) -> CacheStats:
        """Get overall cache performance statistics."""
        return await self.store.get_stats()

    async def clear_cache(self) -> None:
        """Clear all cached atoms."""
        await self.store.clear()

    async def evict_stale(self, max_age_days: int = 30) -> int:
        """Remove stale atoms that haven't been used recently."""
        if hasattr(self.store, 'evict_stale'):
            return await self.store.evict_stale(max_age_days)
        return 0


class IntentAtomsMiddleware:
    """
    Drop-in middleware for existing LLM API calls.
    
    Wraps any function that takes a string and returns a string,
    adding Intent Atoms caching transparently.
    
    Usage:
        # Wrap your existing LLM call
        @intent_atoms_middleware(engine)
        async def my_llm_call(query: str) -> str:
            return await openai.complete(query)
        
        # Now it automatically uses caching
        result = await my_llm_call("How do I deploy React?")
    """

    def __init__(self, engine: IntentAtomsEngine):
        self.engine = engine

    def __call__(self, func):
        async def wrapper(query: str, **kwargs) -> str:
            # Check if bypass is requested
            if kwargs.pop("bypass_cache", False):
                return await func(query, **kwargs)
            
            result = await self.engine.query(query)
            return result.response
        
        wrapper._engine = self.engine
        wrapper._original = func
        return wrapper


def intent_atoms_middleware(engine: IntentAtomsEngine):
    """Decorator factory for the middleware."""
    return IntentAtomsMiddleware(engine)
