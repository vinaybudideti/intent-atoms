"""
Intent Atoms — Intelligent semantic caching for LLM APIs.

v1: Atomic intent decomposition (Decompose -> Match -> Generate misses -> Compose)
v2: Full-query FAISS caching (Embed -> Search -> Return or Generate)
"""

__version__ = "0.2.0"
__author__ = "Vinay"

# v2 engine (default)
from .engine_v2 import IntentAtomsEngineV2

# v1 engine (kept for benchmarks)
from .engine_v1 import IntentAtomsEngine

from .models import Atom, CachedQuery, QueryResult, CacheStats

__all__ = [
    "IntentAtomsEngineV2",
    "IntentAtomsEngine",
    "Atom",
    "CachedQuery",
    "QueryResult",
    "CacheStats",
]
