#!/usr/bin/env python3
"""
100-Query Benchmark — V3 hybrid engine cache warm-up test.

10 original queries x 10 variations each (1 original + 9 paraphrases) = 100 queries.
Shuffled order to simulate real-world traffic. Proves cache improves over time.

Usage:
  python tests/benchmark_100.py --dry-run   # Simulates LLM calls (no API cost)
  python tests/benchmark_100.py             # Real Anthropic API calls
"""

import argparse
import asyncio
import io
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from intent_atoms import IntentAtomsEngineV3


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


# ── 100 queries: 10 topics x 10 variations ─────────────────────────────────

QUERY_GROUPS = [
    # Group 1: React + Docker + AWS
    [
        "How do I deploy a React app with Docker on AWS?",
        "Deploying React applications using Docker containers to AWS",
        "Steps to containerize a React project and host it on Amazon",
        "React app deployment workflow with Docker and AWS services",
        "How to push a Dockerized React frontend to AWS?",
        "Setting up a React Docker container on Amazon Web Services",
        "Guide to deploying React with Docker on AWS ECS",
        "What's the process for running a React app in Docker on AWS?",
        "Containerize and deploy React to AWS using Docker",
        "Best approach to ship a React application via Docker to AWS",
    ],
    # Group 2: JWT + Node.js
    [
        "What is JWT authentication and how to implement it in Node.js?",
        "Implementing JSON Web Token auth in a Node.js application",
        "How do JWTs work and how to use them in Node.js?",
        "JWT authentication tutorial for Node.js backend",
        "Setting up token-based authentication with JWT in Node",
        "How to secure a Node.js API using JWT tokens?",
        "Explain JWT and show how to add it to a Node.js server",
        "Node.js JWT implementation for user authentication",
        "Building auth with JSON Web Tokens in Node.js Express",
        "How to generate and verify JWTs in a Node application?",
    ],
    # Group 3: SQL vs NoSQL
    [
        "Explain SQL vs NoSQL databases",
        "Comparing SQL and NoSQL database technologies",
        "What are the differences between SQL and NoSQL?",
        "SQL versus NoSQL: when to use each type of database",
        "Relational databases compared to non-relational databases",
        "How do SQL databases differ from NoSQL databases?",
        "Pros and cons of SQL vs NoSQL for application development",
        "When should I use SQL instead of NoSQL?",
        "SQL and NoSQL database comparison for developers",
        "Understanding the trade-offs between SQL and NoSQL systems",
    ],
    # Group 4: CI/CD + GitHub Actions
    [
        "How to set up CI/CD with GitHub Actions?",
        "Setting up continuous integration and deployment with GitHub Actions",
        "GitHub Actions CI/CD pipeline configuration guide",
        "How to automate builds and deployments using GitHub Actions?",
        "Creating a CI/CD workflow with GitHub Actions YAML",
        "Automating testing and deployment with GitHub Actions",
        "Step-by-step guide to GitHub Actions for CI/CD pipelines",
        "How to configure automated builds in GitHub Actions?",
        "Using GitHub Actions to set up continuous deployment",
        "GitHub Actions tutorial for automated testing and releases",
    ],
    # Group 5: Flask + Docker + AWS
    [
        "How to deploy a Python Flask app with Docker on AWS?",
        "Deploying Flask applications in Docker containers on AWS",
        "Containerizing a Flask API and running it on Amazon Web Services",
        "Steps to Dockerize a Flask app and push to AWS",
        "Flask Docker deployment to AWS: complete walkthrough",
        "How to run a Dockerized Python Flask backend on AWS?",
        "Setting up Flask in a Docker container on AWS ECS",
        "Guide to deploying a Flask REST API with Docker on AWS",
        "Packaging a Flask application with Docker for AWS deployment",
        "How to containerize Flask and deploy to Amazon cloud?",
    ],
    # Group 6: JWT + FastAPI
    [
        "Implement JWT auth in a FastAPI Python application",
        "How to add JWT authentication to a FastAPI project?",
        "FastAPI JWT token-based authentication implementation",
        "Building secure auth with JWT in Python FastAPI",
        "Setting up JSON Web Token auth for a FastAPI backend",
        "How to protect FastAPI endpoints with JWT?",
        "FastAPI authentication tutorial using JWT tokens",
        "Implementing token-based security in FastAPI with JWT",
        "How to create login and JWT verification in FastAPI?",
        "Adding JWT auth middleware to a Python FastAPI application",
    ],
    # Group 7: Relational vs document stores
    [
        "Comparing relational databases versus document stores",
        "Relational databases vs document-oriented databases explained",
        "How do relational databases compare to document stores like MongoDB?",
        "SQL relational databases versus NoSQL document databases",
        "When to choose a relational database over a document store",
        "Differences between RDBMS and document-based databases",
        "Comparing PostgreSQL-style databases with MongoDB-style stores",
        "Relational vs document store: which is better for my app?",
        "Understanding relational and document database paradigms",
        "Trade-offs between relational tables and document collections",
    ],
    # Group 8: Containerize React + push to AWS
    [
        "Steps to containerize a React frontend and push to AWS",
        "How to put a React frontend in a Docker container for AWS",
        "Containerizing a React application and deploying to AWS",
        "React frontend Docker container to AWS deployment steps",
        "How to create a Docker image for React and push it to AWS?",
        "Building a Docker container for a React SPA and hosting on AWS",
        "Packaging React frontend in Docker for Amazon deployment",
        "Guide to Dockerizing React and uploading to AWS",
        "Steps for creating a React Docker image and running on AWS",
        "How to build and deploy a containerized React app on AWS?",
    ],
    # Group 9: Docker networking
    [
        "How does Docker networking work internally between containers?",
        "Explain Docker container networking and inter-container communication",
        "How do Docker containers communicate with each other?",
        "Docker networking internals: bridge, host, and overlay networks",
        "Understanding how Docker handles network traffic between containers",
        "How does Docker route traffic between running containers?",
        "Docker container-to-container networking explained",
        "Internal networking mechanisms in Docker for multi-container apps",
        "How does Docker DNS and bridge networking work?",
        "Explaining Docker network drivers and container connectivity",
    ],
    # Group 10: Kubernetes basics
    [
        "What is Kubernetes and how to get started with it?",
        "Kubernetes fundamentals: pods, services, and deployments",
        "Getting started with Kubernetes container orchestration",
        "How to deploy your first application on Kubernetes?",
        "Kubernetes basics for beginners: core concepts explained",
        "Introduction to Kubernetes: what it is and why it matters",
        "How does Kubernetes manage containerized applications?",
        "Starting with Kubernetes: setting up your first cluster",
        "Kubernetes 101: understanding pods, nodes, and clusters",
        "Beginner's guide to deploying apps with Kubernetes",
    ],
]

# Flatten and shuffle with a fixed seed for reproducibility
ALL_QUERIES = [q for group in QUERY_GROUPS for q in group]
random.seed(42)
SHUFFLED_QUERIES = ALL_QUERIES.copy()
random.shuffle(SHUFFLED_QUERIES)


# ── Dry-run mock ────────────────────────────────────────────────────────────

ATOM_RULES = [
    (["react", "frontend", "spa"],     {"intent_text": "How to build a React application for production",       "intent_label": "react_build",     "domain_tags": ["react", "frontend"]}),
    (["docker", "container", "dockerize"], {"intent_text": "How to containerize a web application with Docker",  "intent_label": "docker_container", "domain_tags": ["docker", "devops"]}),
    (["aws", "amazon", "ecs"],         {"intent_text": "How to deploy Docker containers to AWS",                "intent_label": "aws_deploy",      "domain_tags": ["aws", "cloud"]}),
    (["jwt", "json web token"],        {"intent_text": "How to implement JWT authentication",                   "intent_label": "jwt_auth",        "domain_tags": ["auth", "security"]}),
    (["node", "node.js", "express"],   {"intent_text": "How to build a Node.js backend application",            "intent_label": "nodejs_backend",  "domain_tags": ["nodejs", "backend"]}),
    (["sql", "relational", "rdbms", "postgresql"], {"intent_text": "How relational SQL databases work",         "intent_label": "sql_databases",   "domain_tags": ["databases", "sql"]}),
    (["nosql", "document", "mongodb"], {"intent_text": "How NoSQL document stores work",                        "intent_label": "nosql_databases", "domain_tags": ["databases", "nosql"]}),
    (["ci/cd", "github action", "continuous"], {"intent_text": "How to set up CI/CD pipelines with GitHub Actions", "intent_label": "cicd_github", "domain_tags": ["devops", "ci-cd"]}),
    (["flask"],                        {"intent_text": "How to build a Python Flask web application",           "intent_label": "flask_app",       "domain_tags": ["python", "flask"]}),
    (["fastapi"],                      {"intent_text": "How to build a FastAPI Python application",             "intent_label": "fastapi_app",     "domain_tags": ["python", "fastapi"]}),
    (["network", "bridge", "dns"],     {"intent_text": "How Docker networking works between containers",        "intent_label": "docker_network",  "domain_tags": ["docker", "networking"]}),
    (["kubernetes", "k8s", "pod", "cluster"], {"intent_text": "How Kubernetes container orchestration works",   "intent_label": "kubernetes",      "domain_tags": ["kubernetes", "devops"]}),
    (["deploy", "deployment"],         {"intent_text": "How to deploy applications to cloud platforms",         "intent_label": "cloud_deploy",    "domain_tags": ["devops", "cloud"]}),
    (["auth", "authentication", "security", "token"], {"intent_text": "How to implement authentication in web applications", "intent_label": "web_auth", "domain_tags": ["auth", "security"]}),
    (["compar", "versus", "vs", "differ", "trade-off"], {"intent_text": "How to compare and choose between technologies", "intent_label": "tech_comparison", "domain_tags": ["architecture"]}),
]


def _decompose_query_dry(query: str) -> list[dict]:
    q = query.lower()
    atoms = []
    for keywords, atom in ATOM_RULES:
        if any(kw in q for kw in keywords):
            atoms.append(atom)
    if not atoms:
        atoms.append({"intent_text": query, "intent_label": "full_query", "domain_tags": []})
    return atoms[:4]  # Cap at 4 atoms


def patch_provider_dry_run(provider):
    async def mock_complete(prompt, system="", max_tokens=1024, model=None):
        if "decomposition" in system.lower() or "decompose" in system.lower():
            atoms = _decompose_query_dry(prompt)
            response_json = json.dumps({"atoms": atoms})
            return {"text": response_json, "input_tokens": 80, "output_tokens": len(response_json) // 4}

        if "adapt" in system.lower() or "previously answered" in prompt.lower():
            return {
                "text": "[DRY-RUN] Adapted response based on cached similar answer.",
                "input_tokens": 200,
                "output_tokens": 250,
            }

        snippet = prompt[:60].replace("\n", " ")
        return {
            "text": f"[DRY-RUN] Response for: {snippet}... Placeholder content.",
            "input_tokens": 150,
            "output_tokens": 300,
        }

    provider.complete = mock_complete


# ── Benchmark runner ────────────────────────────────────────────────────────

async def run_benchmark(engine, queries: list[str], dry_run: bool) -> list[dict]:
    await engine.clear_cache()

    results = []
    n = len(queries)

    # Cumulative tracking
    cum_hits = 0
    cum_total = 0
    cum_cost = 0.0
    cum_cost_no_cache = 0.0

    print(f"\n{'=' * 80}")
    print(f"  V3 Hybrid Engine — {n} queries {'(DRY-RUN)' if dry_run else '(LIVE)'}")
    print(f"{'=' * 80}")
    print(f"  {'#':>3}  {'Status':<14} {'Cost':>11} {'Time':>8} {'Sim':>6} {'Tier':<12} Query")
    print(f"  {'-' * 76}")

    for i, q in enumerate(queries, 1):
        r = await engine.query(q)

        tier = r.match_tier or "full_miss"
        cum_hits += r.cache_hits
        cum_total += r.total_atoms
        cum_cost += r.estimated_cost
        cum_cost_no_cache += r.estimated_cost_without_cache

        row = {
            "query": q,
            "cache_hits": r.cache_hits,
            "total_atoms": r.total_atoms,
            "cost": r.estimated_cost,
            "cost_without_cache": r.estimated_cost_without_cache,
            "time_ms": r.total_time_ms,
            "similarity_score": r.similarity_score,
            "match_tier": tier,
            "is_cache_hit": r.is_cache_hit,
            "tokens_used": r.total_tokens_used,
            "tokens_saved": r.tokens_saved,
        }
        results.append(row)

        # Per-query line
        status = "HIT" if r.cache_hits > 0 else "MISS"
        sim_str = f"{r.similarity_score:.3f}" if r.similarity_score > 0 else "  —  "
        q_short = q[:40] + ".." if len(q) > 42 else q
        print(f"  {i:3d}  {status:<4} {tier:<9} ${r.estimated_cost:>9.6f} {r.total_time_ms:>6.0f}ms {sim_str:>6} {q_short}")

        # Cumulative summary every 10 queries
        if i % 10 == 0:
            hit_rate = (cum_hits / cum_total * 100) if cum_total > 0 else 0
            savings = ((1 - cum_cost / cum_cost_no_cache) * 100) if cum_cost_no_cache > 0 else 0
            print(f"  {'─' * 76}")
            print(f"  After {i:3d} queries: hit_rate={hit_rate:5.1f}%  cost=${cum_cost:.6f}  savings={savings:5.1f}%")
            print(f"  {'─' * 76}")

    return results


def print_summary(results: list[dict]):
    n = len(results)
    total_cost = sum(r["cost"] for r in results)
    total_cost_nc = sum(r["cost_without_cache"] for r in results)
    total_hits = sum(r["cache_hits"] for r in results)
    total_atoms = sum(r["total_atoms"] for r in results)
    total_time = sum(r["time_ms"] for r in results)
    total_tokens = sum(r["tokens_used"] for r in results)
    total_saved = sum(r["tokens_saved"] for r in results)
    hit_rate = (total_hits / total_atoms * 100) if total_atoms > 0 else 0
    savings = ((1 - total_cost / total_cost_nc) * 100) if total_cost_nc > 0 else 0

    # Tier counts
    tier_counts = {}
    for r in results:
        t = r["match_tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1

    # Cost by decile
    deciles = []
    for start in range(0, n, 10):
        chunk = results[start:start + 10]
        chunk_cost = sum(r["cost"] for r in chunk)
        chunk_hits = sum(r["cache_hits"] for r in chunk)
        chunk_atoms = sum(r["total_atoms"] for r in chunk)
        chunk_rate = (chunk_hits / chunk_atoms * 100) if chunk_atoms > 0 else 0
        deciles.append({"range": f"{start+1}-{start+10}", "cost": chunk_cost, "hit_rate": chunk_rate})

    print(f"\n\n{'=' * 65}")
    print(f"{'FINAL RESULTS — 100-QUERY V3 BENCHMARK':^65}")
    print(f"{'=' * 65}")
    print(f"  Total queries:       {n}")
    print(f"  Cache hits / total:  {total_hits}/{total_atoms}")
    print(f"  Overall hit rate:    {hit_rate:.1f}%")
    print(f"  Total cost:          ${total_cost:.6f}")
    print(f"  Cost without cache:  ${total_cost_nc:.6f}")
    print(f"  Cost savings:        {savings:.1f}%")
    print(f"  Total time:          {total_time:.0f}ms")
    print(f"  Tokens used:         {total_tokens:,}")
    print(f"  Tokens saved:        {total_saved:,}")

    print(f"\n  Tier Breakdown:")
    for tier in ["direct_hit", "adapted", "atom_hit", "full_miss"]:
        count = tier_counts.get(tier, 0)
        pct = count / n * 100
        print(f"    {tier:<12s}  {count:3d} queries  ({pct:5.1f}%)")

    print(f"\n  Cache Warm-Up Curve (cost & hit rate per 10-query block):")
    print(f"  {'Block':>8}  {'Cost':>12}  {'Hit Rate':>10}")
    print(f"  {'-' * 34}")
    for d in deciles:
        print(f"  {d['range']:>8}  ${d['cost']:>10.6f}  {d['hit_rate']:>9.1f}%")

    print(f"{'=' * 65}\n")


# ── Save results ────────────────────────────────────────────────────────────

def save_results(mode: str, results: list[dict], output_text: str):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = PROJECT_ROOT / "benchmarks" / f"{ts}_{mode}_100q"
    run_dir.mkdir(parents=True, exist_ok=True)

    n = len(results)
    total_cost = sum(r["cost"] for r in results)
    total_cost_nc = sum(r["cost_without_cache"] for r in results)
    total_hits = sum(r["cache_hits"] for r in results)
    total_atoms = sum(r["total_atoms"] for r in results)
    hit_rate = (total_hits / total_atoms * 100) if total_atoms > 0 else 0
    savings = ((1 - total_cost / total_cost_nc) * 100) if total_cost_nc > 0 else 0

    tier_counts = {}
    for r in results:
        t = r["match_tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1

    meta = {
        "timestamp": ts,
        "mode": mode,
        "num_queries": n,
        "num_topics": len(QUERY_GROUPS),
        "shuffle_seed": 42,
        "engine": "v3_hybrid",
        "summary": {
            "hit_rate": round(hit_rate, 2),
            "total_cost": round(total_cost, 8),
            "cost_without_cache": round(total_cost_nc, 8),
            "cost_savings_pct": round(savings, 2),
            "total_hits": total_hits,
            "total_atoms": total_atoms,
            "tier_counts": tier_counts,
        },
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "v3_results.json").write_text(json.dumps(results, indent=2))
    (run_dir / "output.txt").write_text(output_text)

    print(f"  Results saved to: {run_dir.relative_to(PROJECT_ROOT)}/")
    return run_dir


# ── Main ────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="100-query V3 benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Simulate LLM calls")
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

    print(f"\n  Intent Atoms 100-Query Benchmark — {mode_display}")
    print(f"  {len(QUERY_GROUPS)} topics x 10 variations = {len(SHUFFLED_QUERIES)} queries (shuffled, seed=42)")

    engine = IntentAtomsEngineV3(
        llm_provider="anthropic",
        api_key=api_key or "dry-run-key",
        persist_dir=None,
        direct_hit_threshold=0.85,
        adapt_threshold=0.70,
        atom_threshold=0.82,
    )

    if args.dry_run:
        patch_provider_dry_run(engine.provider)

    results = await run_benchmark(engine, SHUFFLED_QUERIES, args.dry_run)
    print_summary(results)

    # Restore stdout and save
    sys.stdout = original_stdout
    output_text = capture.getvalue()
    save_results(mode, results, output_text)


if __name__ == "__main__":
    asyncio.run(main())
