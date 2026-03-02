"""
Core data models for Intent Atoms.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone
import uuid
import json


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Atom:
    """
    An atomic unit of intent — the smallest meaningful question/answer fragment
    that can be cached and recomposed.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    intent_label: str = ""           # e.g. "react_build_process"
    intent_text: str = ""            # e.g. "How to build a React application for production"
    response_fragment: str = ""      # The generated answer for this atomic intent
    embedding: list[float] = field(default_factory=list)  # Vector embedding
    created_at: datetime = field(default_factory=_utcnow)
    last_used: datetime = field(default_factory=_utcnow)
    usage_count: int = 0
    confidence: float = 1.0          # Quality score of the cached response
    token_count: int = 0             # Tokens in the response fragment
    domain_tags: list[str] = field(default_factory=list)  # e.g. ["devops", "react", "aws"]
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "intent_label": self.intent_label,
            "intent_text": self.intent_text,
            "response_fragment": self.response_fragment,
            "embedding": self.embedding,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "usage_count": self.usage_count,
            "confidence": self.confidence,
            "token_count": self.token_count,
            "domain_tags": self.domain_tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Atom":
        data = data.copy()
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("last_used"), str):
            data["last_used"] = datetime.fromisoformat(data["last_used"])
        return cls(**data)


@dataclass
class CachedQuery:
    """
    A cached full-query entry for v2 engine.
    Caches complete query-response pairs with their embedding for FAISS similarity search.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    query_text: str = ""
    response_text: str = ""
    embedding: list[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utcnow)
    last_used: datetime = field(default_factory=_utcnow)
    usage_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    generation_cost: float = 0.0
    model_used: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "query_text": self.query_text,
            "response_text": self.response_text,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "usage_count": self.usage_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "generation_cost": self.generation_cost,
            "model_used": self.model_used,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CachedQuery":
        data = data.copy()
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("last_used"), str):
            data["last_used"] = datetime.fromisoformat(data["last_used"])
        return cls(**data)


@dataclass
class DecomposedQuery:
    """Result of decomposing a user query into atomic intents."""
    original_query: str = ""
    atoms: list[dict] = field(default_factory=list)  # [{intent_text, intent_label, domain_tags}]
    decomposition_tokens: int = 0  # Tokens used for decomposition
    decomposition_cost: float = 0.0
    
    @property
    def num_atoms(self) -> int:
        return len(self.atoms)


@dataclass
class MatchResult:
    """Result of matching an atomic intent against the store."""
    query_intent: str = ""
    matched_atom: Optional[Atom] = None
    similarity_score: float = 0.0
    is_cache_hit: bool = False
    

@dataclass
class QueryResult:
    """Complete result of processing a query through Intent Atoms."""
    original_query: str = ""
    response: str = ""
    
    # Atom breakdown
    total_atoms: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    atoms_generated: list[Atom] = field(default_factory=list)
    atoms_reused: list[Atom] = field(default_factory=list)
    
    # Cost analysis
    total_tokens_used: int = 0
    tokens_saved: int = 0
    estimated_cost: float = 0.0
    estimated_cost_without_cache: float = 0.0
    
    # Timing
    total_time_ms: float = 0.0
    decomposition_time_ms: float = 0.0
    matching_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    composition_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    search_time_ms: float = 0.0

    # v2 fields
    is_cache_hit: bool = False
    similarity_score: float = 0.0
    matched_query: str = ""
    
    @property
    def cost_savings_pct(self) -> float:
        if self.estimated_cost_without_cache == 0:
            return 0.0
        return (1 - self.estimated_cost / self.estimated_cost_without_cache) * 100

    @property
    def cache_hit_rate(self) -> float:
        if self.total_atoms == 0:
            return 0.0
        return self.cache_hits / self.total_atoms * 100


@dataclass
class CacheStats:
    """Overall cache performance statistics."""
    total_atoms_stored: int = 0
    total_queries_processed: int = 0
    overall_hit_rate: float = 0.0
    total_tokens_saved: int = 0
    total_cost_saved: float = 0.0
    avg_atoms_per_query: float = 0.0
    most_reused_atoms: list[dict] = field(default_factory=list)
    domain_distribution: dict = field(default_factory=dict)
    
    # Time series for dashboard
    daily_stats: list[dict] = field(default_factory=list)
