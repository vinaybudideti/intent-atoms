"""
Tests for Intent Atoms engine components.
Run: pytest tests/ -v
"""

import pytest
import asyncio
import math
from intent_atoms.models import Atom, CachedQuery, QueryResult, DecomposedQuery, MatchResult, CacheStats
from intent_atoms.atom_store import LocalAtomStore
from intent_atoms.faiss_store import FAISSStore


# ── Model Tests ──

class TestAtom:
    def test_create_atom(self):
        atom = Atom(
            intent_label="test_intent",
            intent_text="How to test Python code",
            response_fragment="Use pytest...",
            token_count=50,
            domain_tags=["python", "testing"],
        )
        assert atom.intent_label == "test_intent"
        assert atom.usage_count == 0
        assert len(atom.id) == 12

    def test_atom_serialization(self):
        atom = Atom(intent_label="test", intent_text="test q", response_fragment="test a")
        d = atom.to_dict()
        restored = Atom.from_dict(d)
        assert restored.intent_label == atom.intent_label
        assert restored.intent_text == atom.intent_text

    def test_query_result_metrics(self):
        result = QueryResult(
            total_atoms=5,
            cache_hits=3,
            cache_misses=2,
            estimated_cost=0.005,
            estimated_cost_without_cache=0.012,
        )
        assert result.cache_hit_rate == 60.0
        assert abs(result.cost_savings_pct - 58.33) < 0.1


# ── Store Tests ──

class TestLocalAtomStore:
    @pytest.fixture
    def store(self):
        return LocalAtomStore(persist_path=None)

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, store):
        atom = Atom(
            intent_label="test",
            intent_text="test query",
            response_fragment="test response",
            embedding=[0.1] * 256,
        )
        atom_id = await store.store(atom)
        retrieved = await store.get(atom_id)
        assert retrieved is not None
        assert retrieved.intent_label == "test"

    @pytest.mark.asyncio
    async def test_similarity_search(self, store):
        # Store two atoms with different embeddings
        norm = 1.0 / math.sqrt(256)
        
        atom1 = Atom(
            intent_label="python",
            intent_text="Python programming",
            response_fragment="Python is...",
            embedding=[norm] * 256,
        )
        atom2 = Atom(
            intent_label="javascript",
            intent_text="JavaScript programming",
            response_fragment="JavaScript is...",
            embedding=[-norm] * 256,
        )
        await store.store(atom1)
        await store.store(atom2)

        # Search with similar vector to atom1
        results = await store.search([norm] * 256, threshold=0.5)
        assert len(results) >= 1
        assert results[0][0].intent_label == "python"
        assert results[0][1] > 0.9  # High similarity

    @pytest.mark.asyncio
    async def test_usage_tracking(self, store):
        atom = Atom(intent_label="test", embedding=[0.1] * 256)
        await store.store(atom)
        assert atom.usage_count == 0

        await store.update_usage(atom.id)
        updated = await store.get(atom.id)
        assert updated.usage_count == 1

    @pytest.mark.asyncio
    async def test_clear(self, store):
        atom = Atom(intent_label="test", embedding=[0.1] * 256)
        await store.store(atom)
        assert len(store.atoms) == 1

        await store.clear()
        assert len(store.atoms) == 0

    @pytest.mark.asyncio
    async def test_stats(self, store):
        stats = await store.get_stats()
        assert isinstance(stats, CacheStats)
        assert stats.total_atoms_stored == 0


# ── Cosine Similarity Tests ──

class TestCosineSimilarity:
    def test_identical_vectors(self):
        store = LocalAtomStore()
        v = [1.0, 0.0, 0.0]
        assert abs(store._cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        store = LocalAtomStore()
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        assert abs(store._cosine_similarity(v1, v2)) < 1e-6

    def test_opposite_vectors(self):
        store = LocalAtomStore()
        v1 = [1.0, 0.0]
        v2 = [-1.0, 0.0]
        assert abs(store._cosine_similarity(v1, v2) - (-1.0)) < 1e-6


# ── v2: CachedQuery Model Tests ──

class TestCachedQuery:
    def test_create_cached_query(self):
        cq = CachedQuery(
            query_text="How to test?",
            response_text="Use pytest...",
            output_tokens=50,
        )
        assert cq.usage_count == 0
        assert len(cq.id) == 12
        assert cq.query_text == "How to test?"

    def test_serialization(self):
        cq = CachedQuery(
            query_text="test query",
            response_text="test response",
            input_tokens=10,
            output_tokens=50,
        )
        d = cq.to_dict()
        restored = CachedQuery.from_dict(d)
        assert restored.query_text == cq.query_text
        assert restored.response_text == cq.response_text
        assert restored.input_tokens == 10


# ── v2: FAISS Store Tests ──

class TestFAISSStore:
    @pytest.fixture
    def store(self):
        return FAISSStore(dimension=768, persist_dir=None)

    @pytest.mark.asyncio
    async def test_store_and_search_hit(self, store):
        import numpy as np
        rng = np.random.default_rng(42)
        vec = rng.random(768).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        embedding = vec.tolist()

        await store.store(
            embedding=embedding,
            query_text="How to deploy React?",
            response_text="To deploy React...",
            metadata={"output_tokens": 100},
        )

        results = await store.search(embedding, top_k=1, threshold=0.83)
        assert len(results) == 1
        entry, score = results[0]
        assert score > 0.99
        assert entry.query_text == "How to deploy React?"

    @pytest.mark.asyncio
    async def test_search_miss_below_threshold(self, store):
        import numpy as np
        rng = np.random.default_rng(42)

        vec1 = rng.random(768).astype(np.float32)
        vec1 = vec1 / np.linalg.norm(vec1)

        vec2 = rng.random(768).astype(np.float32)
        vec2 = vec2 / np.linalg.norm(vec2)

        await store.store(
            embedding=vec1.tolist(),
            query_text="How to deploy React?",
            response_text="To deploy React...",
        )

        results = await store.search(vec2.tolist(), top_k=1, threshold=0.99)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_clear(self, store):
        import numpy as np
        vec = np.ones(768, dtype=np.float32)
        vec = vec / np.linalg.norm(vec)

        await store.store(
            embedding=vec.tolist(),
            query_text="test",
            response_text="test",
        )
        assert store.index.ntotal == 1
        assert len(store.entries) == 1

        await store.clear()
        assert store.index.ntotal == 0
        assert len(store.entries) == 0

    @pytest.mark.asyncio
    async def test_stats(self, store):
        stats = await store.get_stats()
        assert isinstance(stats, CacheStats)
        assert stats.total_atoms_stored == 0

    @pytest.mark.asyncio
    async def test_query_result_v2_fields(self):
        result = QueryResult(
            total_atoms=1,
            cache_hits=1,
            cache_misses=0,
            estimated_cost=0.0,
            estimated_cost_without_cache=0.01,
            is_cache_hit=True,
            matched_query="How to deploy React?",
            similarity_score=0.91,
        )
        assert result.is_cache_hit is True
        assert result.matched_query == "How to deploy React?"
        assert result.cost_savings_pct == 100.0
