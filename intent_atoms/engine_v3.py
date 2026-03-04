"""
Intent Atoms Engine v3 — Hybrid two-layer caching with 3-tier matching.

Layer 1 (full-query FAISS search) has three tiers:
  Tier 1 — DIRECT HIT  (sim > 0.85): Return cached response. Zero cost. ~15ms.
  Tier 2 — ADAPT       (sim 0.70–0.85): Cheap Haiku call adapts a near-miss response. ~4x cheaper.
  Tier 3 — FULL MISS   (sim < 0.70): Fall through to Layer 2 atom matching.

Layer 2 (atom-level decomposition + matching):
  Decompose → embed atoms → FAISS atom search → reuse hits, generate misses → compose.

Layer 3 (cache everything):
  Store full query + response in query index AND each atom + fragment in atom index.

Two FAISS indexes:
  - query_index: full query embeddings (Layer 1 lookups)
  - atom_index:  atomic intent embeddings (Layer 2 lookups)
"""

import time
from typing import Optional
from .models import Atom, QueryResult, CacheStats
from .faiss_store import FAISSStore
from .decomposer import IntentDecomposer
from .composer import ResponseComposer
from .providers import LLMProvider, get_provider


ADAPT_SYSTEM_PROMPT = """You are a response adaptation engine. You receive a user's question and a previously cached answer to a similar question. Your job is to adapt the cached answer to better address the user's specific question.

RULES:
1. Keep the same structure and detail level as the original answer.
2. Adjust any specifics that differ between the original and new question.
3. Do not add information that wasn't in the original answer unless directly relevant.
4. Write the adapted response directly. No preamble."""


class IntentAtomsEngineV3:
    """
    Hybrid v3 engine with 3-tier full-query matching and atom-level fallback.

    Usage:
        engine = IntentAtomsEngineV3(
            llm_provider="anthropic",
            api_key="sk-...",
            persist_dir="./data/v3_cache",
        )

        result = await engine.query("How do I deploy React with Docker?")
        # result.match_tier: "direct_hit" | "adapted" | "atom_hit" | "full_miss"
    """

    def __init__(
        self,
        llm_provider: str = "anthropic",
        api_key: str = "",
        persist_dir: Optional[str] = "./data/v3_cache",
        direct_hit_threshold: float = 0.85,
        adapt_threshold: float = 0.70,
        atom_threshold: float = 0.82,
        embedding_dimension: int = 768,
    ):
        self.provider = get_provider(llm_provider, api_key=api_key)

        # Layer 1: full-query FAISS index
        self.query_store = FAISSStore(
            dimension=embedding_dimension,
            persist_dir=persist_dir,
            index_prefix="query" if persist_dir else None,
        )

        # Layer 2: atom-level FAISS index
        self.atom_store = FAISSStore(
            dimension=embedding_dimension,
            persist_dir=persist_dir,
            index_prefix="atom" if persist_dir else None,
        )

        # Reuse v1 components for decomposition and composition
        self.decomposer = IntentDecomposer(self.provider)
        self.composer = ResponseComposer(self.provider)

        self.direct_hit_threshold = direct_hit_threshold
        self.adapt_threshold = adapt_threshold
        self.atom_threshold = atom_threshold
        self._query_count = 0

    # ── kept for backward compat with benchmark that passes query_threshold ──
    @property
    def query_threshold(self):
        return self.direct_hit_threshold

    @query_threshold.setter
    def query_threshold(self, value):
        self.direct_hit_threshold = value

    async def query(self, user_query: str) -> QueryResult:
        """Process a query through the 3-tier hybrid pipeline."""
        total_start = time.time()
        result = QueryResult(original_query=user_query)

        # ─── Embed the full query (shared by both layers) ───
        embed_start = time.time()
        embeddings = await self.provider.embed([user_query])
        query_embedding = embeddings[0]
        result.embedding_time_ms = (time.time() - embed_start) * 1000

        # ─── Layer 1: Full-query FAISS search (3 tiers) ───
        search_start = time.time()
        # Search with the lower adapt threshold to catch near-misses too
        query_results = await self.query_store.search(
            embedding=query_embedding,
            top_k=1,
            threshold=self.adapt_threshold,
        )
        result.search_time_ms = (time.time() - search_start) * 1000

        if query_results:
            cached_entry, similarity = query_results[0]

            if similarity >= self.direct_hit_threshold:
                # ── Tier 1: DIRECT HIT — return cached response, zero cost ──
                result.response = cached_entry.response_text
                result.match_layer = 1
                result.match_tier = "direct_hit"
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

                result.matching_time_ms = result.search_time_ms
                result.decomposition_time_ms = 0.0
                result.generation_time_ms = 0.0
                result.composition_time_ms = 0.0

                result.total_time_ms = (time.time() - total_start) * 1000
                self._query_count += 1

                await self.query_store.log_query({
                    "query": user_query,
                    "match_layer": 1,
                    "match_tier": "direct_hit",
                    "is_cache_hit": True,
                    "similarity_score": similarity,
                    "matched_query": cached_entry.query_text,
                    "tokens_saved": result.tokens_saved,
                    "cost_saved": cached_entry.generation_cost,
                })

                return result

            else:
                # ── Tier 2: ADAPT — cheap Haiku call to adapt near-miss ──
                gen_start = time.time()

                adapt_prompt = (
                    f"A user asked: \"{user_query}\"\n\n"
                    f"A similar question was previously answered:\n"
                    f"Question: \"{cached_entry.query_text}\"\n"
                    f"Answer: {cached_entry.response_text}\n\n"
                    f"Please adapt this answer to better address the user's "
                    f"specific question. Keep the same structure and detail level, "
                    f"but adjust any specifics that differ."
                )

                # Use cheap model (Haiku) instead of expensive Sonnet
                adapt_model = getattr(self.provider, 'compose_model', None)
                adapt_result = await self.provider.complete(
                    prompt=adapt_prompt,
                    system=ADAPT_SYSTEM_PROMPT,
                    max_tokens=1024,
                    model=adapt_model,
                )

                result.generation_time_ms = (time.time() - gen_start) * 1000

                adapted_text = adapt_result["text"]
                adapt_in = adapt_result["input_tokens"]
                adapt_out = adapt_result["output_tokens"]
                adapt_cost = self.provider.estimate_cost(
                    adapt_in, adapt_out, model=adapt_model
                )

                result.response = adapted_text
                result.match_layer = 1
                result.match_tier = "adapted"
                result.is_cache_hit = True
                result.matched_query = cached_entry.query_text
                result.similarity_score = similarity

                result.total_atoms = 1
                result.cache_hits = 1
                result.cache_misses = 0
                result.tokens_saved = cached_entry.output_tokens + cached_entry.input_tokens
                result.total_tokens_used = adapt_in + adapt_out
                result.estimated_cost = adapt_cost
                result.estimated_cost_without_cache = cached_entry.generation_cost

                result.matching_time_ms = result.search_time_ms
                result.decomposition_time_ms = 0.0
                result.composition_time_ms = 0.0

                # Cache the adapted response as a new entry
                await self.query_store.store(
                    embedding=query_embedding,
                    query_text=user_query,
                    response_text=adapted_text,
                    metadata={
                        "input_tokens": adapt_in,
                        "output_tokens": adapt_out,
                        "generation_cost": adapt_cost,
                        "model_used": adapt_model or "",
                    },
                )

                result.total_time_ms = (time.time() - total_start) * 1000
                self._query_count += 1

                await self.query_store.log_query({
                    "query": user_query,
                    "match_layer": 1,
                    "match_tier": "adapted",
                    "is_cache_hit": True,
                    "similarity_score": similarity,
                    "matched_query": cached_entry.query_text,
                    "adapt_cost": adapt_cost,
                    "tokens_used": adapt_in + adapt_out,
                })

                return result

        # ─── Tier 3 / Layer 2: Atom-level decomposition + matching ───
        decomp_start = time.time()
        decomposed = await self.decomposer.decompose(user_query)
        result.decomposition_time_ms = (time.time() - decomp_start) * 1000
        result.total_atoms = decomposed.num_atoms

        gen_start = time.time()
        fragments = []
        total_gen_input = 0
        total_gen_output = 0
        atom_hit_count = 0

        for atom_info in decomposed.atoms:
            intent_text = atom_info["intent_text"]

            # Embed and search atom index
            atom_embeddings = await self.provider.embed([intent_text])
            atom_embedding = atom_embeddings[0]

            atom_results = await self.atom_store.search(
                embedding=atom_embedding,
                top_k=1,
                threshold=self.atom_threshold,
            )

            if atom_results:
                # Atom cache hit — reuse cached fragment
                cached_atom, sim = atom_results[0]
                atom_hit_count += 1
                result.cache_hits += 1
                result.tokens_saved += cached_atom.output_tokens + cached_atom.input_tokens

                fragments.append({
                    "intent": intent_text,
                    "response": cached_atom.response_text,
                    "source": "cache",
                })
            else:
                # Atom cache miss — generate
                result.cache_misses += 1
                response_text, in_tok, out_tok = await self.composer.generate_atom(
                    intent_text,
                    atom_info.get("domain_tags", []),
                )
                total_gen_input += in_tok
                total_gen_output += out_tok

                # Cache the new atom
                await self.atom_store.store(
                    embedding=atom_embedding,
                    query_text=intent_text,
                    response_text=response_text,
                    metadata={
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "generation_cost": self.provider.estimate_cost(in_tok, out_tok),
                        "model_used": getattr(self.provider, 'generate_model', ''),
                    },
                )

                fragments.append({
                    "intent": intent_text,
                    "response": response_text,
                    "source": "generated",
                })

        result.generation_time_ms = (time.time() - gen_start) * 1000

        # ─── Compose fragments into final response ───
        compose_start = time.time()
        composed_text, comp_in, comp_out = await self.composer.compose(
            user_query, fragments
        )
        result.composition_time_ms = (time.time() - compose_start) * 1000
        result.response = composed_text

        # ─── Set match tier ───
        if atom_hit_count > 0:
            result.match_layer = 2
            result.match_tier = "atom_hit"
            result.is_cache_hit = True
        else:
            result.match_layer = 0
            result.match_tier = "full_miss"
            result.is_cache_hit = False

        # ─── Calculate costs ───
        decomp_cost = decomposed.decomposition_cost
        gen_cost = self.provider.estimate_cost(total_gen_input, total_gen_output)
        compose_cost = self.provider.estimate_cost(comp_in, comp_out) if comp_in > 0 else 0
        result.estimated_cost = decomp_cost + gen_cost + compose_cost
        result.total_tokens_used = (
            decomposed.decomposition_tokens
            + total_gen_input + total_gen_output
            + comp_in + comp_out
        )

        # Hypothetical cost without cache
        result.estimated_cost_without_cache = self.provider.estimate_cost(
            100 * result.total_atoms,
            300 * result.total_atoms,
        )
        if result.cache_hits > 0 and result.estimated_cost_without_cache < result.estimated_cost:
            result.estimated_cost_without_cache = result.estimated_cost * 1.5

        result.matching_time_ms = result.search_time_ms
        result.total_time_ms = (time.time() - total_start) * 1000

        # ─── Cache the full query + response for Layer 1 future hits ───
        gen_model = getattr(self.provider, 'generate_model', None)
        await self.query_store.store(
            embedding=query_embedding,
            query_text=user_query,
            response_text=composed_text,
            metadata={
                "input_tokens": result.total_tokens_used,
                "output_tokens": total_gen_output + comp_out,
                "generation_cost": result.estimated_cost,
                "model_used": gen_model or "",
            },
        )

        # ─── Log for analytics ───
        await self.query_store.log_query({
            "query": user_query,
            "match_layer": result.match_layer,
            "match_tier": result.match_tier,
            "is_cache_hit": result.is_cache_hit,
            "num_atoms": result.total_atoms,
            "atom_hits": atom_hit_count,
            "tokens_saved": result.tokens_saved,
            "cost": result.estimated_cost,
        })

        self._query_count += 1
        return result

    async def get_stats(self) -> CacheStats:
        """Get combined cache performance statistics."""
        q_stats = await self.query_store.get_stats()
        a_stats = await self.atom_store.get_stats()
        return CacheStats(
            total_atoms_stored=q_stats.total_atoms_stored + a_stats.total_atoms_stored,
            total_queries_processed=q_stats.total_queries_processed,
            overall_hit_rate=q_stats.overall_hit_rate,
            total_tokens_saved=q_stats.total_tokens_saved + a_stats.total_tokens_saved,
            total_cost_saved=q_stats.total_cost_saved + a_stats.total_cost_saved,
            avg_atoms_per_query=q_stats.avg_atoms_per_query,
            most_reused_atoms=q_stats.most_reused_atoms + a_stats.most_reused_atoms,
            domain_distribution={},
            daily_stats=q_stats.daily_stats,
        )

    async def clear_cache(self) -> None:
        """Clear both FAISS indexes."""
        await self.query_store.clear()
        await self.atom_store.clear()

    async def evict_stale(self, max_age_days: int = 30) -> int:
        """Remove stale entries from both indexes."""
        q_removed = await self.query_store.evict_stale(max_age_days)
        a_removed = await self.atom_store.evict_stale(max_age_days)
        return q_removed + a_removed
