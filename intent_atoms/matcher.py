"""
Similarity Matcher — Finds cached atoms that match incoming intents.

The matching strategy uses a tiered threshold:
  - Perfect match (>0.92): Direct reuse, no modification needed
  - Strong match (0.82-0.92): Reuse with minor adaptation
  - Weak match (0.72-0.82): Reuse as context, but regenerate
  - Miss (<0.72): Full generation needed

This tiered approach maximizes cache utilization while maintaining quality.
"""

import time
from typing import Optional
from .models import Atom, MatchResult
from .atom_store import AtomStore
from .providers import LLMProvider


class SimilarityMatcher:
    """Matches incoming atomic intents against the cached atom store."""

    def __init__(
        self,
        store: AtomStore,
        provider: LLMProvider,
        perfect_threshold: float = 0.92,
        strong_threshold: float = 0.82,
        weak_threshold: float = 0.72,
    ):
        self.store = store
        self.provider = provider
        self.perfect_threshold = perfect_threshold
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold

    async def match(self, intent_text: str) -> MatchResult:
        """
        Find the best matching cached atom for a given intent.
        """
        # Generate embedding for the incoming intent
        embeddings = await self.provider.embed([intent_text])
        embedding = embeddings[0]

        # Search the atom store
        results = await self.store.search(
            embedding=embedding,
            top_k=3,
            threshold=self.weak_threshold,
        )

        if not results:
            return MatchResult(
                query_intent=intent_text,
                matched_atom=None,
                similarity_score=0.0,
                is_cache_hit=False,
            )

        best_atom, best_score = results[0]

        # Determine if this counts as a cache hit
        is_hit = best_score >= self.strong_threshold

        if is_hit:
            await self.store.update_usage(best_atom.id)

        return MatchResult(
            query_intent=intent_text,
            matched_atom=best_atom,
            similarity_score=best_score,
            is_cache_hit=is_hit,
        )

    async def match_batch(self, intent_texts: list[str]) -> list[MatchResult]:
        """Match multiple intents at once (more efficient for embeddings)."""
        if not intent_texts:
            return []

        # Batch embed all intents
        embeddings = await self.provider.embed(intent_texts)

        results = []
        for intent_text, embedding in zip(intent_texts, embeddings):
            search_results = await self.store.search(
                embedding=embedding,
                top_k=1,
                threshold=self.weak_threshold,
            )

            if search_results:
                best_atom, best_score = search_results[0]
                is_hit = best_score >= self.strong_threshold
                if is_hit:
                    await self.store.update_usage(best_atom.id)
                results.append(MatchResult(
                    query_intent=intent_text,
                    matched_atom=best_atom,
                    similarity_score=best_score,
                    is_cache_hit=is_hit,
                ))
            else:
                results.append(MatchResult(
                    query_intent=intent_text,
                    matched_atom=None,
                    similarity_score=0.0,
                    is_cache_hit=False,
                ))

        return results

    def classify_match(self, score: float) -> str:
        """Classify a similarity score into match tiers."""
        if score >= self.perfect_threshold:
            return "perfect"
        elif score >= self.strong_threshold:
            return "strong"
        elif score >= self.weak_threshold:
            return "weak"
        else:
            return "miss"
