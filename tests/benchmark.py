#!/usr/bin/env python3
"""
Benchmark: V1 (decompose) vs V2 (FAISS) vs V3 (hybrid 3-tier)

Runs 10 queries through all three engines and compares:
  - Cache hit rates and match tiers (V3: direct_hit / adapted / atom_hit / full_miss)
  - API costs
  - Response times
  - Similarity scores

Results are saved to benchmarks/<timestamp>_<mode>_10q/

Usage:
  python tests/benchmark.py --dry-run   # Simulates LLM calls (no API cost)
  python tests/benchmark.py             # Real Anthropic API calls
"""

import argparse
import asyncio
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from intent_atoms import IntentAtomsEngine, IntentAtomsEngineV2, IntentAtomsEngineV3


# ── Tee: capture stdout while still printing ───────────────────────────────

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()
    def isatty(self):
        return False


# ── Benchmark queries ───────────────────────────────────────────────────────

QUERIES = [
    "How do I deploy a React app with Docker on AWS?",
    "What is JWT authentication and how to implement it in Node.js?",
    "Explain SQL vs NoSQL databases",
    "How to set up CI/CD with GitHub Actions?",
    "How to deploy a Python Flask app with Docker on AWS?",
    "Implement JWT auth in a FastAPI Python application",
    "Comparing relational databases versus document stores",
    "Steps to containerize a React frontend and push to AWS",
    "How does Docker networking work internally between containers?",
    "How do I deploy a React app with Docker on AWS?",
]


# ── Dry-run mock ────────────────────────────────────────────────────────────

ATOM_RULES = [
    (["react", "frontend"],    {"intent_text": "How to build a React application for production",       "intent_label": "react_build",     "domain_tags": ["react", "frontend"]}),
    (["docker", "container"],   {"intent_text": "How to containerize a web application with Docker",     "intent_label": "docker_container", "domain_tags": ["docker", "devops"]}),
    (["aws", "cloud"],          {"intent_text": "How to deploy Docker containers to AWS",                "intent_label": "aws_deploy",      "domain_tags": ["aws", "cloud"]}),
    (["jwt", "auth"],           {"intent_text": "How to implement JWT authentication",                   "intent_label": "jwt_auth",        "domain_tags": ["auth", "security"]}),
    (["node", "node.js"],       {"intent_text": "How to build a Node.js backend application",            "intent_label": "nodejs_backend",  "domain_tags": ["nodejs", "backend"]}),
    (["sql", "relational"],     {"intent_text": "How relational SQL databases work",                     "intent_label": "sql_databases",   "domain_tags": ["databases", "sql"]}),
    (["nosql", "document"],     {"intent_text": "How NoSQL document stores work",                        "intent_label": "nosql_databases", "domain_tags": ["databases", "nosql"]}),
    (["ci/cd", "github action"],{"intent_text": "How to set up CI/CD pipelines with GitHub Actions",     "intent_label": "cicd_github",     "domain_tags": ["devops", "ci-cd"]}),
    (["flask", "python"],       {"intent_text": "How to build a Python Flask web application",           "intent_label": "flask_app",       "domain_tags": ["python", "flask"]}),
    (["fastapi"],               {"intent_text": "How to build a FastAPI Python application",             "intent_label": "fastapi_app",     "domain_tags": ["python", "fastapi"]}),
    (["network"],               {"intent_text": "How Docker networking works between containers",        "intent_label": "docker_network",  "domain_tags": ["docker", "networking"]}),
]


def _decompose_query_dry(query: str) -> list[dict]:
    q = query.lower()
    atoms = []
    for keywords, atom in ATOM_RULES:
        if any(kw in q for kw in keywords):
            atoms.append(atom)
    if not atoms:
        atoms.append({"intent_text": query, "intent_label": "full_query", "domain_tags": []})
    return atoms


def patch_provider_dry_run(provider):
    async def mock_complete(prompt, system="", max_tokens=1024, model=None):
        if "decomposition" in system.lower() or "decompose" in system.lower():
            atoms = _decompose_query_dry(prompt)
            response_json = json.dumps({"atoms": atoms})
            return {"text": response_json, "input_tokens": 80, "output_tokens": len(response_json) // 4}

        if "adapt" in system.lower() or "previously answered" in prompt.lower():
            return {
                "text": "[DRY-RUN] Adapted response. The original cached answer has been adjusted.",
                "input_tokens": 200, "output_tokens": 250,
            }

        snippet = prompt[:80].replace("\n", " ")
        return {
            "text": f"[DRY-RUN] Simulated response for: {snippet}...\nPlaceholder content.",
            "input_tokens": 150, "output_tokens": 300,
        }

    provider.complete = mock_complete


# ── Benchmark runner ────────────────────────────────────────────────────────

async def run_engine(engine, engine_name: str, queries: list[str]) -> list[dict]:
    await engine.clear_cache()
    results = []
    n = len(queries)
    print(f"\n{'=' * 70}")
    print(f"  Running {engine_name} — {n} queries")
    print(f"{'=' * 70}")

    for i, q in enumerate(queries, 1):
        r = await engine.query(q)
        tier = getattr(r, "match_tier", "") or ""
        row = {
            "query": q,
            "cache_hits": r.cache_hits,
            "cache_misses": r.cache_misses,
            "total_atoms": r.total_atoms,
            "cost": r.estimated_cost,
            "cost_without_cache": r.estimated_cost_without_cache,
            "time_ms": r.total_time_ms,
            "tokens_used": r.total_tokens_used,
            "tokens_saved": r.tokens_saved,
            "similarity_score": r.similarity_score,
            "matched_query": r.matched_query,
            "is_cache_hit": r.is_cache_hit,
            "match_layer": r.match_layer,
            "match_tier": tier,
        }
        results.append(row)

        TIER_LABELS = {"direct_hit": "T1:direct", "adapted": "T2:adapt", "atom_hit": "L2:atom", "full_miss": ""}
        status = "HIT" if r.cache_hits > 0 else "MISS"
        tier_label = TIER_LABELS.get(tier, "")
        tier_str = f"  {tier_label}" if tier_label else ""
        sim = f"  sim={r.similarity_score:.3f}" if r.similarity_score > 0 else ""
        cost_str = f"${r.estimated_cost:.6f}"
        print(f"  [{i:2d}/{n:2d}] {status:4s}{tier_str:<12s} cost={cost_str:>12s}  time={r.total_time_ms:7.0f}ms{sim}")
        if r.matched_query:
            print(f"          matched: \"{r.matched_query[:60]}\"")

    return results


# ── Results display ─────────────────────────────────────────────────────────

def totals(results: list[dict]) -> dict:
    total_hits = sum(r["cache_hits"] for r in results)
    total_atoms = sum(r["total_atoms"] for r in results)
    total_cost = sum(r["cost"] for r in results)
    total_cost_no_cache = sum(r["cost_without_cache"] for r in results)
    total_time = sum(r["time_ms"] for r in results)
    total_tokens = sum(r["tokens_used"] for r in results)
    total_saved = sum(r["tokens_saved"] for r in results)
    hit_rate = (total_hits / total_atoms * 100) if total_atoms > 0 else 0
    savings_pct = ((1 - total_cost / total_cost_no_cache) * 100) if total_cost_no_cache > 0 else 0
    t1_direct = sum(1 for r in results if r["match_tier"] == "direct_hit")
    t2_adapt = sum(1 for r in results if r["match_tier"] == "adapted")
    l2_atom = sum(1 for r in results if r["match_tier"] == "atom_hit")
    full_miss = sum(1 for r in results if r["match_tier"] == "full_miss" or r["match_tier"] == "")
    return {
        "hits": total_hits, "atoms": total_atoms, "cost": total_cost,
        "cost_no_cache": total_cost_no_cache, "time": total_time,
        "tokens": total_tokens, "saved": total_saved, "hit_rate": hit_rate,
        "savings_pct": savings_pct, "t1_direct": t1_direct, "t2_adapt": t2_adapt,
        "l2_atom": l2_atom, "full_miss": full_miss,
    }


def print_comparison(v1_results, v2_results, v3_results):
    W = 150
    print(f"\n\n{'=' * W}")
    print(f"{'QUERY-BY-QUERY COMPARISON':^{W}}")
    print(f"{'=' * W}")
    header = (
        f"{'#':>2}  {'Query':<40}  "
        f"{'V1 Hit':>6} {'V1 Cost':>10} {'V1 ms':>7}  "
        f"{'V2 Hit':>6} {'V2 Cost':>10} {'V2 ms':>7}  "
        f"{'V3 Hit':>6} {'V3 Cost':>10} {'V3 ms':>7} {'Tier':>10}"
    )
    print(header)
    print("-" * W)

    for i, (v1, v2, v3) in enumerate(zip(v1_results, v2_results, v3_results), 1):
        q_short = v1["query"][:38] + ".." if len(v1["query"]) > 40 else v1["query"]
        v1_hit = f"{v1['cache_hits']}/{v1['total_atoms']}"
        v2_hit = "HIT" if v2["is_cache_hit"] else "MISS"
        v3_hit = "HIT" if v3["is_cache_hit"] else "MISS"
        tier = v3["match_tier"] or "—"
        print(
            f"{i:2d}  {q_short:<40}  "
            f"{v1_hit:>6} ${v1['cost']:>9.6f} {v1['time_ms']:>6.0f}ms  "
            f"{v2_hit:>6} ${v2['cost']:>9.6f} {v2['time_ms']:>6.0f}ms  "
            f"{v3_hit:>6} ${v3['cost']:>9.6f} {v3['time_ms']:>6.0f}ms {tier:>10}"
        )

    t1 = totals(v1_results)
    t2 = totals(v2_results)
    t3 = totals(v3_results)

    SW = 85
    print(f"\n\n{'=' * SW}")
    print(f"{'SUMMARY':^{SW}}")
    print(f"{'=' * SW}")
    print(f"{'Metric':<30}  {'V1 (Decompose)':>16}  {'V2 (FAISS)':>16}  {'V3 (Hybrid)':>16}")
    print("-" * SW)
    print(f"{'Cache hits / total':.<30}  {t1['hits']:>6}/{t1['atoms']:<9} {t2['hits']:>6}/{t2['atoms']:<9} {t3['hits']:>6}/{t3['atoms']:<9}")
    print(f"{'Hit rate':.<30}  {t1['hit_rate']:>15.1f}% {t2['hit_rate']:>15.1f}% {t3['hit_rate']:>15.1f}%")
    print(f"{'Total cost':.<30}  ${t1['cost']:>14.6f}  ${t2['cost']:>14.6f}  ${t3['cost']:>14.6f}")
    print(f"{'Cost without cache':.<30}  ${t1['cost_no_cache']:>14.6f}  ${t2['cost_no_cache']:>14.6f}  ${t3['cost_no_cache']:>14.6f}")
    print(f"{'Cost savings':.<30}  {t1['savings_pct']:>15.1f}% {t2['savings_pct']:>15.1f}% {t3['savings_pct']:>15.1f}%")
    print(f"{'Total time':.<30}  {t1['time']:>14.0f}ms  {t2['time']:>14.0f}ms  {t3['time']:>14.0f}ms")
    print(f"{'Tokens used':.<30}  {t1['tokens']:>16,}  {t2['tokens']:>16,}  {t3['tokens']:>16,}")
    print(f"{'Tokens saved':.<30}  {t1['saved']:>16,}  {t2['saved']:>16,}  {t3['saved']:>16,}")
    print(f"{'=' * SW}")

    print(f"\n  V3 Tier Breakdown:")
    print(f"    Tier 1 — Direct Hit (sim > 0.85):  {t3['t1_direct']} queries  (zero cost)")
    print(f"    Tier 2 — Adapted   (sim 0.70-0.85): {t3['t2_adapt']} queries  (cheap Haiku)")
    print(f"    Layer 2 — Atom Hit  (decompose):    {t3['l2_atom']} queries  (partial cache)")
    print(f"    Full Miss (no cache):               {t3['full_miss']} queries  (full Sonnet)")

    engines = [("V1 (Decompose)", t1), ("V2 (FAISS)", t2), ("V3 (Hybrid)", t3)]
    best_savings = max(engines, key=lambda e: e[1]["savings_pct"])
    best_speed = min(engines, key=lambda e: e[1]["time"])
    print(f"\n  >> Best cost savings: {best_savings[0]} at {best_savings[1]['savings_pct']:.1f}%")
    print(f"  >> Fastest overall:  {best_speed[0]} at {best_speed[1]['time']:.0f}ms")
    print()


# ── Save results ────────────────────────────────────────────────────────────

def save_results(mode: str, v1_results, v2_results, v3_results, output_text: str):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = PROJECT_ROOT / "benchmarks" / f"{ts}_{mode}_10q"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Meta
    meta = {
        "timestamp": ts,
        "mode": mode,
        "num_queries": len(QUERIES),
        "engines": ["v1_decompose", "v2_faiss", "v3_hybrid"],
        "queries": QUERIES,
        "v1_summary": totals(v1_results),
        "v2_summary": totals(v2_results),
        "v3_summary": totals(v3_results),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Per-engine results
    (run_dir / "v1_results.json").write_text(json.dumps(v1_results, indent=2))
    (run_dir / "v2_results.json").write_text(json.dumps(v2_results, indent=2))
    (run_dir / "v3_results.json").write_text(json.dumps(v3_results, indent=2))

    # Full console output
    (run_dir / "output.txt").write_text(output_text)

    print(f"  Results saved to: {run_dir.relative_to(PROJECT_ROOT)}/")
    return run_dir


# ── Main ────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Benchmark V1 vs V2 vs V3 caching engines")
    parser.add_argument("--dry-run", action="store_true", help="Simulate LLM calls with dummy responses (no API cost)")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("LLM_API_KEY", "")

    if not args.dry_run and not api_key:
        print("ERROR: LLM_API_KEY not found in .env — use --dry-run or set the key")
        sys.exit(1)

    mode = "dry-run" if args.dry_run else "live"
    mode_display = "DRY-RUN (simulated LLM)" if args.dry_run else "LIVE (real API calls)"

    # Tee stdout to capture output
    capture = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, capture)

    print(f"\n  Intent Atoms Benchmark — {mode_display}")
    print(f"  Queries: {len(QUERIES)}")
    print(f"  Engines: V1 (Decompose) | V2 (FAISS) | V3 (Hybrid 3-tier)")

    key = api_key or "dry-run-key"

    v1 = IntentAtomsEngine(llm_provider="anthropic", api_key=key, store_backend="local", similarity_threshold=0.82, persist_path=None)
    v2 = IntentAtomsEngineV2(llm_provider="anthropic", api_key=key, persist_dir=None, similarity_threshold=0.83)
    v3 = IntentAtomsEngineV3(llm_provider="anthropic", api_key=key, persist_dir=None, direct_hit_threshold=0.85, adapt_threshold=0.70, atom_threshold=0.82)

    if args.dry_run:
        patch_provider_dry_run(v1.provider)
        patch_provider_dry_run(v2.provider)
        patch_provider_dry_run(v3.provider)

    v1_results = await run_engine(v1, "V1 (Decompose → Match → Generate → Compose)", QUERIES)
    v2_results = await run_engine(v2, "V2 (Embed → FAISS Search → Return/Generate)", QUERIES)
    v3_results = await run_engine(v3, "V3 (Tier 1: Direct | Tier 2: Adapt | Layer 2: Atoms)", QUERIES)

    print_comparison(v1_results, v2_results, v3_results)

    # Restore stdout and save
    sys.stdout = original_stdout
    output_text = capture.getvalue()

    run_dir = save_results(mode, v1_results, v2_results, v3_results, output_text)


if __name__ == "__main__":
    asyncio.run(main())
