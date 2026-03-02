# ⚛ Intent Atoms

**Sub-query level intelligent caching for LLM APIs.**  
Reduce your API costs by 40-60% without sacrificing response quality.

---

## The Problem

Every LLM API call costs money. Even if a user asks the same question that was answered 5 minutes ago, the full inference pipeline runs from scratch. Existing solutions like semantic caching match at the **query level** — but real queries are compound:

> "How do I deploy a React app with Docker on AWS?"

This isn't one question. It's **three atomic intents**:
1. Building React for production
2. Containerizing with Docker  
3. Deploying to AWS

## The Solution: Intent Atoms

Instead of caching complete queries (low hit rate) or complete responses (context-dependent), Intent Atoms caches at the **atomic intent level**.

```
Query → Decompose → Match atoms → Generate misses only → Compose → Return
                          ↓
                    [Atom Cache]
                    docker_containerization ✓ (cached)
                    aws_deployment ✓ (cached)  
                    react_build ✗ (generate)
```

When a new query shares atomic intents with previous queries, those fragments are pulled from cache — and only the **novel fragments** are generated. The LLM goes from being a **generator** to being a **composer**.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Intent Atoms Engine                    │
│                                                          │
│  ┌──────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐ │
│  │Decomposer│→ │ Matcher  │→ │Generator │→ │ Composer │ │
│  │(Haiku)   │  │(Vectors) │  │(Sonnet)  │  │(Haiku)   │ │
│  └──────────┘  └────┬─────┘  └──────────┘  └──────────┘ │
│                     │                                     │
│              ┌──────┴──────┐                              │
│              │ Atom Store  │                              │
│              │ (MongoDB /  │                              │
│              │  Local)     │                              │
│              └─────────────┘                              │
└─────────────────────────────────────────────────────────┘
```

**Cost model:**
- Decomposition: Haiku ($0.80/1M tokens) — cheap
- Matching: Vector similarity — free (no LLM call)
- Generation: Sonnet ($3/1M tokens) — only for cache misses
- Composition: Haiku ($0.80/1M tokens) — cheap

The expensive model (Sonnet) is only called for **genuinely new** atomic intents.

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/intent-atoms.git
cd intent-atoms
pip install -r requirements.txt
```

### Basic Usage

```python
import asyncio
from intent_atoms import IntentAtomsEngine

async def main():
    engine = IntentAtomsEngine(
        llm_provider="anthropic",       # or "openai"
        api_key="sk-ant-...",
        store_backend="local",           # or "mongodb"
        persist_path="./data/cache.json",
        similarity_threshold=0.88,
    )

    # First query — all cache misses (atoms get generated + cached)
    result = await engine.query("How do I deploy a React app with Docker on AWS?")
    print(result.response)
    print(f"Atoms: {result.total_atoms} | Hits: {result.cache_hits} | Savings: {result.cost_savings_pct:.1f}%")

    # Second query — shares "Docker" and "AWS" atoms!
    result = await engine.query("How to deploy a Python Flask app with Docker to AWS?")
    print(f"Atoms: {result.total_atoms} | Hits: {result.cache_hits} | Savings: {result.cost_savings_pct:.1f}%")
    # → Docker and AWS atoms reused, only Flask atom generated fresh

asyncio.run(main())
```

### Drop-in Middleware

```python
from intent_atoms import IntentAtomsEngine
from intent_atoms.engine import intent_atoms_middleware

engine = IntentAtomsEngine(llm_provider="anthropic", api_key="sk-...")

@intent_atoms_middleware(engine)
async def my_llm_call(query: str) -> str:
    # Your existing LLM call — this becomes the fallback
    return await your_existing_function(query)

# Now automatically uses Intent Atoms caching
response = await my_llm_call("How do I set up CI/CD?")
```

### API Server

```bash
export LLM_PROVIDER=anthropic
export LLM_API_KEY=sk-ant-...
export STORE_BACKEND=local

uvicorn api.server:app --host 0.0.0.0 --port 8000
```

**Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| POST | `/query` | Process a query through the engine |
| GET | `/stats` | Cache performance statistics |
| GET | `/atoms` | List all cached atoms |
| POST | `/clear` | Clear the cache |
| POST | `/evict` | Remove stale atoms |

### MongoDB Atlas (Production)

```bash
export STORE_BACKEND=mongodb
export MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net"
```

Create a vector search index on your `atoms` collection:
```json
{
  "fields": [{
    "type": "vector",
    "path": "embedding",
    "numDimensions": 256,
    "similarity": "cosine"
  }]
}
```

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | LLM provider (`anthropic` or `openai`) |
| `LLM_API_KEY` | — | Your API key |
| `STORE_BACKEND` | `local` | Storage backend (`local` or `mongodb`) |
| `SIMILARITY_THRESHOLD` | `0.88` | Minimum similarity for cache hit |
| `PERSIST_PATH` | `./data/atom_cache.json` | Local store file path |
| `MONGO_URI` | — | MongoDB connection string |

## How Cost Savings Scale

| Queries Processed | Unique Atom Patterns | Expected Cache Hit Rate | Cost Reduction |
|---|---|---|---|
| 10 | ~15-25 | 10-20% | 5-15% |
| 100 | ~60-100 | 30-45% | 25-35% |
| 1,000 | ~200-400 | 45-60% | 35-50% |
| 10,000+ | ~500-1000 | 55-70% | 45-60% |

The more diverse your query volume, the richer your atom cache becomes, and the higher your savings grow over time.

## Project Structure

```
intent-atoms/
├── intent_atoms/
│   ├── __init__.py          # Package exports
│   ├── models.py            # Data models (Atom, QueryResult, etc.)
│   ├── providers.py         # LLM provider abstractions
│   ├── decomposer.py        # Query → atomic intents
│   ├── matcher.py           # Similarity matching against cache
│   ├── atom_store.py        # Vector storage (local + MongoDB)
│   ├── composer.py          # Fragment → coherent response
│   └── engine.py            # Main orchestrator + middleware
├── api/
│   └── server.py            # FastAPI REST API
├── tests/
│   └── test_engine.py       # Test suite
├── examples/
│   └── basic_usage.py       # Quick start example
├── dashboard.jsx            # React analytics dashboard
├── requirements.txt
├── Dockerfile
└── README.md
```

## Roadmap

- [ ] **v0.2**: Fine-tuned small model for decomposition (eliminate Haiku dependency)
- [ ] **v0.3**: Atom quality scoring — track if reused atoms get positive user feedback
- [ ] **v0.4**: Cross-user shared atom pool with privacy controls
- [ ] **v0.5**: Real-time dashboard with WebSocket updates
- [ ] **v1.0**: Production SDK with OpenAI/Anthropic drop-in replacement clients

## Tech Stack

- **Python 3.11+** with asyncio
- **FastAPI** for the API layer
- **MongoDB Atlas** with vector search for production storage
- **Anthropic Claude** (Haiku for cheap ops, Sonnet for generation)
- **React + Recharts** for the analytics dashboard

## License

MIT

---

Built by [Vinay](https://github.com/yourusername) — reducing LLM costs, one atom at a time. ⚛
