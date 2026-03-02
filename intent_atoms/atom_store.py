"""
Atom Store — Persistent vector storage for intent atoms.

Supports two backends:
  - "local": In-memory with JSON file persistence (for development)
  - "mongodb": MongoDB Atlas with vector search (for production)

The store handles:
  - Storing atoms with their embeddings
  - Vector similarity search for cache matching
  - Usage tracking and TTL-based eviction
  - Statistics and analytics
"""

import json
import os
import time
import math
from datetime import datetime, timedelta, timezone
from typing import Optional
from .models import Atom, CacheStats


class AtomStore:
    """Base class for atom storage backends."""

    async def store(self, atom: Atom) -> str:
        raise NotImplementedError

    async def search(self, embedding: list[float], top_k: int = 5, threshold: float = 0.85) -> list[tuple[Atom, float]]:
        """Returns list of (atom, similarity_score) tuples."""
        raise NotImplementedError

    async def get(self, atom_id: str) -> Optional[Atom]:
        raise NotImplementedError

    async def update_usage(self, atom_id: str) -> None:
        raise NotImplementedError

    async def get_stats(self) -> CacheStats:
        raise NotImplementedError

    async def delete(self, atom_id: str) -> None:
        raise NotImplementedError

    async def clear(self) -> None:
        raise NotImplementedError


class LocalAtomStore(AtomStore):
    """
    In-memory atom store with optional JSON file persistence.
    Perfect for development, testing, and small-scale usage.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self.atoms: dict[str, Atom] = {}
        self.persist_path = persist_path
        self.query_log: list[dict] = []
        
        if persist_path and os.path.exists(persist_path):
            self._load()

    async def store(self, atom: Atom) -> str:
        self.atoms[atom.id] = atom
        self._save()
        return atom.id

    async def search(
        self, embedding: list[float], top_k: int = 5, threshold: float = 0.85
    ) -> list[tuple[Atom, float]]:
        """Brute-force cosine similarity search (fine for <10k atoms)."""
        if not self.atoms:
            return []

        results = []
        for atom in self.atoms.values():
            if not atom.embedding:
                continue
            sim = self._cosine_similarity(embedding, atom.embedding)
            if sim >= threshold:
                results.append((atom, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def get(self, atom_id: str) -> Optional[Atom]:
        return self.atoms.get(atom_id)

    async def update_usage(self, atom_id: str) -> None:
        if atom_id in self.atoms:
            self.atoms[atom_id].usage_count += 1
            self.atoms[atom_id].last_used = datetime.now(timezone.utc)
            self._save()

    async def get_stats(self) -> CacheStats:
        atoms = list(self.atoms.values())
        if not atoms:
            return CacheStats()

        total_usage = sum(a.usage_count for a in atoms)
        domain_dist = {}
        for a in atoms:
            for tag in a.domain_tags:
                domain_dist[tag] = domain_dist.get(tag, 0) + 1

        most_reused = sorted(atoms, key=lambda a: a.usage_count, reverse=True)[:10]

        return CacheStats(
            total_atoms_stored=len(atoms),
            total_queries_processed=len(self.query_log),
            overall_hit_rate=self._compute_hit_rate(),
            total_tokens_saved=sum(q.get("tokens_saved", 0) for q in self.query_log),
            total_cost_saved=sum(q.get("cost_saved", 0) for q in self.query_log),
            avg_atoms_per_query=sum(q.get("num_atoms", 0) for q in self.query_log) / max(len(self.query_log), 1),
            most_reused_atoms=[
                {"label": a.intent_label, "text": a.intent_text, "uses": a.usage_count}
                for a in most_reused
            ],
            domain_distribution=domain_dist,
            daily_stats=self._compute_daily_stats(),
        )

    async def log_query(self, query_data: dict) -> None:
        self.query_log.append({**query_data, "timestamp": datetime.now(timezone.utc).isoformat()})
        self._save()

    async def delete(self, atom_id: str) -> None:
        self.atoms.pop(atom_id, None)
        self._save()

    async def clear(self) -> None:
        self.atoms.clear()
        self.query_log.clear()
        self._save()

    async def evict_stale(self, max_age_days: int = 30, min_usage: int = 1) -> int:
        """Remove atoms that haven't been used recently."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        to_remove = []
        for aid, atom in self.atoms.items():
            last_used = atom.last_used
            if last_used.tzinfo is None:
                last_used = last_used.replace(tzinfo=timezone.utc)
            if last_used < cutoff and atom.usage_count <= min_usage:
                to_remove.append(aid)
        for aid in to_remove:
            del self.atoms[aid]
        self._save()
        return len(to_remove)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x**2 for x in a))
        norm_b = math.sqrt(sum(x**2 for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _compute_hit_rate(self) -> float:
        if not self.query_log:
            return 0.0
        total_atoms = sum(q.get("num_atoms", 0) for q in self.query_log)
        total_hits = sum(q.get("cache_hits", 0) for q in self.query_log)
        return total_hits / max(total_atoms, 1)

    def _compute_daily_stats(self) -> list[dict]:
        daily = {}
        for q in self.query_log:
            day = q.get("timestamp", "")[:10]
            if day not in daily:
                daily[day] = {"date": day, "queries": 0, "hits": 0, "atoms": 0, "cost_saved": 0}
            daily[day]["queries"] += 1
            daily[day]["hits"] += q.get("cache_hits", 0)
            daily[day]["atoms"] += q.get("num_atoms", 0)
            daily[day]["cost_saved"] += q.get("cost_saved", 0)
        return sorted(daily.values(), key=lambda d: d["date"])

    def _save(self):
        if not self.persist_path:
            return
        data = {
            "atoms": {aid: a.to_dict() for aid, a in self.atoms.items()},
            "query_log": self.query_log,
        }
        os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(data, f, default=str)

    def _load(self):
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            self.atoms = {aid: Atom.from_dict(d) for aid, d in data.get("atoms", {}).items()}
            self.query_log = data.get("query_log", [])
        except (json.JSONDecodeError, FileNotFoundError):
            pass


class MongoAtomStore(AtomStore):
    """
    MongoDB Atlas atom store with native vector search.
    Production-ready with automatic indexing and efficient similarity queries.
    
    Requires: MongoDB Atlas M10+ cluster with vector search enabled.
    """

    def __init__(self, connection_string: str, database: str = "intent_atoms", collection: str = "atoms"):
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError:
            raise ImportError("pip install motor")

        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client[database]
        self.collection = self.db[collection]
        self.log_collection = self.db["query_log"]

    async def setup_indexes(self):
        """Create vector search index. Run once after collection creation."""
        # Standard indexes
        await self.collection.create_index("intent_label")
        await self.collection.create_index("last_used")
        await self.collection.create_index("usage_count")
        
        # Note: Vector search index must be created via Atlas UI or API:
        # {
        #   "fields": [{
        #     "type": "vector",
        #     "path": "embedding",
        #     "numDimensions": 256,  # or 1536 for OpenAI
        #     "similarity": "cosine"
        #   }]
        # }

    async def store(self, atom: Atom) -> str:
        doc = atom.to_dict()
        await self.collection.insert_one(doc)
        return atom.id

    async def search(
        self, embedding: list[float], top_k: int = 5, threshold: float = 0.85
    ) -> list[tuple[Atom, float]]:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "atom_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                }
            },
            {
                "$addFields": {
                    "similarity_score": {"$meta": "vectorSearchScore"}
                }
            },
            {
                "$match": {
                    "similarity_score": {"$gte": threshold}
                }
            },
        ]

        results = []
        async for doc in self.collection.aggregate(pipeline):
            score = doc.pop("similarity_score", 0)
            doc.pop("_id", None)
            atom = Atom.from_dict(doc)
            results.append((atom, score))
        return results

    async def get(self, atom_id: str) -> Optional[Atom]:
        doc = await self.collection.find_one({"id": atom_id})
        if doc:
            doc.pop("_id", None)
            return Atom.from_dict(doc)
        return None

    async def update_usage(self, atom_id: str) -> None:
        await self.collection.update_one(
            {"id": atom_id},
            {"$inc": {"usage_count": 1}, "$set": {"last_used": datetime.now(timezone.utc).isoformat()}},
        )

    async def log_query(self, query_data: dict) -> None:
        await self.log_collection.insert_one({**query_data, "timestamp": datetime.now(timezone.utc)})

    async def get_stats(self) -> CacheStats:
        total = await self.collection.count_documents({})
        queries = await self.log_collection.count_documents({})
        
        # Aggregation for stats
        pipeline = [{"$group": {"_id": None, "total_saved": {"$sum": "$tokens_saved"}}}]
        cursor = self.log_collection.aggregate(pipeline)
        agg = await cursor.to_list(1)
        
        return CacheStats(
            total_atoms_stored=total,
            total_queries_processed=queries,
            total_tokens_saved=agg[0]["total_saved"] if agg else 0,
        )

    async def delete(self, atom_id: str) -> None:
        await self.collection.delete_one({"id": atom_id})

    async def clear(self) -> None:
        await self.collection.delete_many({})
        await self.log_collection.delete_many({})


def get_store(backend: str = "local", **kwargs) -> AtomStore:
    """Factory function for atom stores."""
    if backend == "local":
        return LocalAtomStore(persist_path=kwargs.get("persist_path"))
    elif backend == "mongodb":
        return MongoAtomStore(
            connection_string=kwargs["connection_string"],
            database=kwargs.get("database", "intent_atoms"),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose 'local' or 'mongodb'.")
