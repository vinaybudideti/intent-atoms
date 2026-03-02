"""
Intent Atoms API Server — FastAPI REST API (v2).

Endpoints:
  POST /query          — Process a query through the v2 FAISS engine
  GET  /stats          — Get cache performance stats
  POST /clear          — Clear the query cache
  POST /evict          — Evict stale entries
  GET  /health         — Health check
  GET  /atoms          — List all cached entries (paginated)
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from intent_atoms import IntentAtomsEngineV2


# ── Configuration ──
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
API_KEY = os.getenv("LLM_API_KEY", "")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./data/faiss_cache")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.83"))

engine: Optional[IntentAtomsEngineV2] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine

    engine = IntentAtomsEngineV2(
        llm_provider=LLM_PROVIDER,
        api_key=API_KEY,
        persist_dir=PERSIST_DIR,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )
    print(f"Intent Atoms Engine v2 initialized (provider={LLM_PROVIDER}, FAISS store)")
    yield
    print("Shutting down Intent Atoms Engine v2")


app = FastAPI(
    title="Intent Atoms API",
    description="Full-query semantic caching for LLM APIs with FAISS vector search.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ──

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user query to process")
    bypass_cache: bool = Field(False, description="Skip cache and generate fresh")

class QueryResponse(BaseModel):
    response: str
    total_atoms: int
    cache_hits: int
    cache_misses: int
    cost_savings_pct: float
    estimated_cost: float
    estimated_cost_without_cache: float
    tokens_saved: int
    total_time_ms: float
    decomposition_time_ms: float
    matching_time_ms: float
    generation_time_ms: float
    composition_time_ms: float
    is_cache_hit: bool = False
    similarity_score: float = 0.0
    matched_query: str = ""
    embedding_time_ms: float = 0.0
    search_time_ms: float = 0.0

class EvictRequest(BaseModel):
    max_age_days: int = Field(30, description="Remove entries unused for this many days")


# ── Endpoints ──

@app.get("/health")
async def health():
    return {"status": "healthy", "engine_ready": engine is not None, "version": "v2"}


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    if not engine:
        raise HTTPException(503, "Engine not initialized")

    result = await engine.query(request.query)

    return QueryResponse(
        response=result.response,
        total_atoms=result.total_atoms,
        cache_hits=result.cache_hits,
        cache_misses=result.cache_misses,
        cost_savings_pct=result.cost_savings_pct,
        estimated_cost=result.estimated_cost,
        estimated_cost_without_cache=result.estimated_cost_without_cache,
        tokens_saved=result.tokens_saved,
        total_time_ms=result.total_time_ms,
        decomposition_time_ms=result.decomposition_time_ms,
        matching_time_ms=result.matching_time_ms,
        generation_time_ms=result.generation_time_ms,
        composition_time_ms=result.composition_time_ms,
        is_cache_hit=result.is_cache_hit,
        similarity_score=result.similarity_score,
        matched_query=result.matched_query,
        embedding_time_ms=result.embedding_time_ms,
        search_time_ms=result.search_time_ms,
    )


@app.get("/stats")
async def get_stats():
    if not engine:
        raise HTTPException(503, "Engine not initialized")
    stats = await engine.get_stats()
    return {
        "total_atoms_stored": stats.total_atoms_stored,
        "total_queries_processed": stats.total_queries_processed,
        "overall_hit_rate": stats.overall_hit_rate,
        "total_tokens_saved": stats.total_tokens_saved,
        "total_cost_saved": stats.total_cost_saved,
        "avg_atoms_per_query": stats.avg_atoms_per_query,
        "most_reused_atoms": stats.most_reused_atoms,
        "domain_distribution": stats.domain_distribution,
        "daily_stats": stats.daily_stats,
    }


@app.post("/clear")
async def clear_cache():
    if not engine:
        raise HTTPException(503, "Engine not initialized")
    await engine.clear_cache()
    return {"message": "Cache cleared"}


@app.post("/evict")
async def evict_stale(request: EvictRequest):
    if not engine:
        raise HTTPException(503, "Engine not initialized")
    removed = await engine.evict_stale(request.max_age_days)
    return {"removed": removed}


@app.get("/atoms")
async def list_cached_entries(skip: int = 0, limit: int = 50):
    """List all cached queries (v2)."""
    if not engine:
        raise HTTPException(503, "Engine not initialized")

    store = engine.store

    if hasattr(store, 'entries'):
        all_entries = sorted(store.entries, key=lambda e: e.usage_count, reverse=True)
        page = all_entries[skip:skip + limit]
        return {
            "total": len(all_entries),
            "atoms": [
                {
                    "id": e.id,
                    "intent_text": e.query_text,
                    "intent_label": e.query_text[:40],
                    "usage_count": e.usage_count,
                    "domain_tags": [],
                    "created_at": e.created_at.isoformat(),
                    "last_used": e.last_used.isoformat(),
                    "token_count": e.output_tokens,
                    "generation_cost": e.generation_cost,
                }
                for e in page
            ],
        }

    return {"total": 0, "atoms": []}


# ── Serve Dashboard (built React app) ──
DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard" / "dist"

if DASHBOARD_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(DASHBOARD_DIR / "assets")), name="dashboard-assets")

    @app.get("/dashboard")
    @app.get("/dashboard/{full_path:path}")
    async def serve_dashboard(full_path: str = ""):
        return FileResponse(str(DASHBOARD_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
