"""
Intent Atoms — Intelligent semantic caching for LLM APIs.

v1: Atomic intent decomposition (Decompose -> Match -> Generate misses -> Compose)
v2: Full-query FAISS caching (Embed -> Search -> Return or Generate)
v3: Hybrid two-layer caching (Full-query FAISS + Atom-level FAISS fallback)
"""

__version__ = "0.3.0"
__author__ = "Vinay"

# v3 engine (default — hybrid)
from .engine_v3 import IntentAtomsEngineV3

# v2 engine
from .engine_v2 import IntentAtomsEngineV2

# v1 engine
from .engine_v1 import IntentAtomsEngine

from .models import Atom, CachedQuery, QueryResult, CacheStats

__all__ = [
    "IntentAtomsEngineV3",
    "IntentAtomsEngineV2",
    "IntentAtomsEngine",
    "Atom",
    "CachedQuery",
    "QueryResult",
    "CacheStats",
]
