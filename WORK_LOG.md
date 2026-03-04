# March 1, 2026 ---------------------------------------------------------------

# Intent Atoms — Work Log

## Project Overview
Sub-query level intelligent semantic caching for LLM APIs. Reduces costs by returning cached responses for semantically similar queries instead of calling the LLM every time.

---

## Phase 1: Initial Setup & Fixes (Completed)

### Environment
- macOS arm64, Python 3.13.2 (Homebrew at `/opt/homebrew/`)
- Created venv: `python3 -m venv venv` (activate with `source venv/bin/activate`)
- All packages installed in venv (zero conflicts)

### Fixes Applied
1. **datetime.utcnow() deprecation** (Python 3.12+) — replaced with `datetime.now(timezone.utc)` in `models.py` and `atom_store.py`
2. **pytest-asyncio strict mode** — created `pytest.ini` with `asyncio_mode = auto`
3. **Timezone-aware comparison** in `atom_store.py` `evict_stale()` — handles mixed naive/aware datetimes
4. **setup.py encoding** — added `encoding="utf-8"` to README open()
5. **dotenv loading** — added `python-dotenv` to `examples/basic_usage.py` and `api/server.py`
6. **Added venv/ to .gitignore**

---

## Phase 2: Dashboard Frontend (Completed)

- Created React+Vite app in `dashboard/` directory
- `dashboard/src/IntentAtomsDashboard.jsx` — main component with simulation + live API modes
- `dashboard/src/api.js` — API utility connecting to backend
- `dashboard/vite.config.js` — proxy `/api` → `localhost:8000` for dev mode
- `api/server.py` serves built dashboard at `/dashboard` route
- Build: `cd dashboard && npm run build`

---

## Phase 3: Sentence-Transformers Integration (Completed, then superseded by v2)

- Replaced fake SHA-256 hash embeddings with real sentence-transformers
- Model was `all-MiniLM-L6-v2` (384-dim) — later switched to `all-mpnet-base-v2` (768-dim)
- Lowered similarity thresholds for real embeddings
- v1 test results: 17.6% hit rate (decompose pipeline limited effectiveness)

---

## Phase 4: v2 Engine — FAISS Full-Query Caching (Completed)

### Architecture Change
```
OLD (v1): Query → Decompose (Haiku) → Match atoms → Generate misses (Sonnet) → Compose (Haiku)
NEW (v2): Query → Embed (local MPNet) → FAISS search → If hit: return cached → If miss: call LLM, cache
```

### Files Created
- `intent_atoms/faiss_store.py` — FAISS IndexFlatIP vector store with JSON+binary persistence
- `intent_atoms/engine_v2.py` — Simplified v2 pipeline (embed → search → return or generate)

### Files Modified
- `intent_atoms/engine.py` → renamed to `intent_atoms/engine_v1.py` (kept for benchmarks)
- `intent_atoms/models.py` — added `CachedQuery` dataclass + v2 fields on `QueryResult` (`is_cache_hit`, `similarity_score`, `matched_query`, `embedding_time_ms`, `search_time_ms`)
- `intent_atoms/providers.py` — switched embedding model to `all-mpnet-base-v2` (768-dim)
- `intent_atoms/__init__.py` — exports both `IntentAtomsEngineV2` (default) and `IntentAtomsEngine` (v1), version bumped to 0.2.0
- `api/server.py` — uses `IntentAtomsEngineV2`, threshold 0.83, new response fields, FAISS persist dir
- `requirements.txt` — added `faiss-cpu>=1.7.4`, `sentence-transformers>=3.0.0`
- `tests/test_engine.py` — added 5 new v2 tests (18 total, all passing)

### v2 Test Results
- **Exact match replay:** 10/10 hits, 100% savings, ~30ms per query
- **Semantic match (paraphrased queries):** 10/10 hits, similarity 0.884–0.983
- **Cache hits cost ZERO LLM tokens**

### Key Config
- Embedding model: `all-mpnet-base-v2` (768 dimensions)
- FAISS index: `IndexFlatIP` (inner product on L2-normalized vectors = cosine similarity)
- Similarity threshold: `0.83`
- Persist dir: `./data/faiss_cache/` (contains `faiss_index.bin` + `query_cache.json`)
- `matched_query` field logs which cached query was hit (for threshold debugging)

---

## Decomposer Changes (v1 only, now superseded)

These changes are in `intent_atoms/decomposer.py` but only matter if using v1 engine:
- Added canonical phrasing rules (rules 6-11) to `DECOMPOSE_SYSTEM_PROMPT`
- Lowered `max_tokens` from 512 to 256
- `intent_atoms/composer.py`: `generate_atom` max_tokens 800→400, `compose` max_tokens 1500→800

---

## How to Run

### Backend
```bash
cd "/Users/vinaykumarreddy/Documents/port folio/project3/intent-atoms"
source venv/bin/activate
python3 -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Frontend (dev mode)
```bash
cd dashboard
npm run dev
```

### Tests
```bash
source venv/bin/activate
python3 -m pytest tests/ -v
```

### API Endpoints
- `GET  /health` — health check (includes `"version": "v2"`)
- `POST /query`  — `{"query": "..."}` → response with cache hit/miss info
- `GET  /stats`  — cache performance statistics
- `POST /clear`  — clear all cached queries
- `POST /evict`  — remove stale entries
- `GET  /atoms`  — list cached entries (paginated)
- `GET  /dashboard` — serve React dashboard

---

## What's NOT Done Yet
- Dashboard frontend not updated for v2 response fields (still works, just doesn't show `matched_query`, `similarity_score` etc.)
- No Vercel/production deployment (needs a server for the Python backend)
- `setup.py` not updated with `faiss-cpu` and `sentence-transformers` in install_requires
- No v1 vs v2 benchmark comparison script
- MongoDB store (`MongoAtomStore`) not adapted for v2

# -------------------------------------------------------------------------------------------------