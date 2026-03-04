"""
FAISS Vector Store for Intent Atoms v2.

Uses faiss-cpu with IndexFlatIP (inner product) for fast similarity search
on L2-normalized embeddings from sentence-transformers (all-mpnet-base-v2).

Persistence:
  - FAISS index: faiss.write_index / faiss.read_index (binary)
  - Metadata: JSON file (query text, response text, timestamps, token counts)
"""

import os
import json
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional
from .models import CachedQuery, CacheStats


class FAISSStore:
    """
    FAISS-based vector store for full-query caching.

    self.index (faiss.IndexFlatIP) and self.entries (list[CachedQuery])
    are aligned by position: the i-th vector corresponds to self.entries[i].
    """

    def __init__(
        self,
        dimension: int = 768,
        persist_dir: Optional[str] = None,
        index_prefix: Optional[str] = None,
    ):
        import faiss

        self.dimension = dimension
        self.persist_dir = persist_dir
        self.index = faiss.IndexFlatIP(dimension)
        self.entries: list[CachedQuery] = []
        self.query_log: list[dict] = []

        self._index_path = None
        self._mapping_path = None
        if persist_dir:
            if index_prefix:
                self._index_path = os.path.join(persist_dir, f"{index_prefix}.index")
                self._mapping_path = os.path.join(persist_dir, f"{index_prefix}_meta.json")
            else:
                self._index_path = os.path.join(persist_dir, "faiss_index.bin")
                self._mapping_path = os.path.join(persist_dir, "query_cache.json")
            self._load()

    async def store(
        self,
        embedding: list[float],
        query_text: str,
        response_text: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store a query-response pair with its embedding. Returns the entry id."""
        metadata = metadata or {}

        entry = CachedQuery(
            query_text=query_text,
            response_text=response_text,
            embedding=embedding,
            input_tokens=metadata.get("input_tokens", 0),
            output_tokens=metadata.get("output_tokens", 0),
            generation_cost=metadata.get("generation_cost", 0.0),
            model_used=metadata.get("model_used", ""),
        )

        vec = np.array([embedding], dtype=np.float32)
        self.index.add(vec)
        self.entries.append(entry)

        self._save()
        return entry.id

    async def search(
        self,
        embedding: list[float],
        top_k: int = 1,
        threshold: float = 0.83,
    ) -> list[tuple[CachedQuery, float]]:
        """Search for similar cached queries. Returns (entry, score) pairs above threshold."""
        if self.index.ntotal == 0:
            return []

        vec = np.array([embedding], dtype=np.float32)
        scores, indices = self.index.search(vec, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score >= threshold:
                entry = self.entries[idx]
                entry.usage_count += 1
                entry.last_used = datetime.now(timezone.utc)
                results.append((entry, float(score)))

        if results:
            self._save()

        return results

    async def get_stats(self) -> CacheStats:
        """Return cache statistics."""
        if not self.entries:
            return CacheStats()

        total_tokens_saved = sum(q.get("tokens_saved", 0) for q in self.query_log)
        total_cost_saved = sum(q.get("cost_saved", 0) for q in self.query_log)

        most_reused = sorted(self.entries, key=lambda e: e.usage_count, reverse=True)[:10]

        total_queries = len(self.query_log)
        total_hits = sum(1 for q in self.query_log if q.get("is_cache_hit", False))
        hit_rate = total_hits / max(total_queries, 1)

        return CacheStats(
            total_atoms_stored=len(self.entries),
            total_queries_processed=total_queries,
            overall_hit_rate=hit_rate,
            total_tokens_saved=total_tokens_saved,
            total_cost_saved=total_cost_saved,
            avg_atoms_per_query=1.0,
            most_reused_atoms=[
                {
                    "label": e.query_text[:50],
                    "text": e.query_text,
                    "uses": e.usage_count,
                }
                for e in most_reused
            ],
            domain_distribution={},
            daily_stats=self._compute_daily_stats(),
        )

    async def log_query(self, query_data: dict) -> None:
        """Log a query for analytics."""
        self.query_log.append({
            **query_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._save()

    async def clear(self) -> None:
        """Reset the entire store."""
        import faiss
        self.index = faiss.IndexFlatIP(self.dimension)
        self.entries.clear()
        self.query_log.clear()
        self._save()

    async def evict_stale(self, max_age_days: int = 30) -> int:
        """Remove entries not used within max_age_days. Rebuilds FAISS index."""
        import faiss

        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

        surviving = []
        removed_count = 0
        for entry in self.entries:
            last_used = entry.last_used
            if last_used.tzinfo is None:
                last_used = last_used.replace(tzinfo=timezone.utc)
            if last_used < cutoff and entry.usage_count <= 1:
                removed_count += 1
            else:
                surviving.append(entry)

        if removed_count == 0:
            return 0

        # Rebuild index from surviving entries
        self.index = faiss.IndexFlatIP(self.dimension)
        if surviving:
            vecs = np.array([e.embedding for e in surviving], dtype=np.float32)
            self.index.add(vecs)

        self.entries = surviving
        self._save()
        return removed_count

    def _compute_daily_stats(self) -> list[dict]:
        """Aggregate query_log into daily statistics."""
        daily = {}
        for q in self.query_log:
            day = q.get("timestamp", "")[:10]
            if day not in daily:
                daily[day] = {"date": day, "queries": 0, "hits": 0, "cost_saved": 0}
            daily[day]["queries"] += 1
            if q.get("is_cache_hit", False):
                daily[day]["hits"] += 1
            daily[day]["cost_saved"] += q.get("cost_saved", 0)
        return sorted(daily.values(), key=lambda d: d["date"])

    def _save(self) -> None:
        """Persist FAISS index + JSON mapping to disk."""
        if not self.persist_dir:
            return

        import faiss

        os.makedirs(self.persist_dir, exist_ok=True)
        faiss.write_index(self.index, self._index_path)

        data = {
            "entries": [e.to_dict() for e in self.entries],
            "query_log": self.query_log,
        }
        with open(self._mapping_path, "w") as f:
            json.dump(data, f, default=str)

    def _load(self) -> None:
        """Load FAISS index + JSON mapping from disk."""
        import faiss

        if not self.persist_dir:
            return

        if self._index_path and os.path.exists(self._index_path):
            self.index = faiss.read_index(self._index_path)

        if self._mapping_path and os.path.exists(self._mapping_path):
            try:
                with open(self._mapping_path, "r") as f:
                    data = json.load(f)
                self.entries = [
                    CachedQuery.from_dict(d) for d in data.get("entries", [])
                ]
                self.query_log = data.get("query_log", [])

                # Reconstruct embeddings from FAISS index
                for i, entry in enumerate(self.entries):
                    if i < self.index.ntotal:
                        entry.embedding = self.index.reconstruct(i).tolist()
            except (json.JSONDecodeError, FileNotFoundError):
                pass
