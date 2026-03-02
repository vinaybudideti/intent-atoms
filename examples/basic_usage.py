"""
Intent Atoms — Basic Usage Example

Run: python examples/basic_usage.py

Demonstrates:
  1. Engine initialization
  2. First query (all cache misses)
  3. Overlapping query (partial cache hits)
  4. Repeated query (full cache hit)
  5. Stats review
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from intent_atoms import IntentAtomsEngine


async def main():
    # Initialize engine
    engine = IntentAtomsEngine(
        llm_provider=os.getenv("LLM_PROVIDER", "anthropic"),
        api_key=os.getenv("LLM_API_KEY", ""),
        store_backend="local",
        persist_path="./data/example_cache.json",
        similarity_threshold=0.88,
    )

    queries = [
        # Query 1: 3 new atoms → all misses
        "How do I deploy a React app with Docker on AWS?",
        # Query 2: shares Docker + AWS atoms → 2 hits, 1 miss (Flask is new)
        "How to deploy a Python Flask app with Docker to AWS?",
        # Query 3: shares React atom → 1 hit, 1 miss (testing is new)
        "Best practices for testing React applications?",
        # Query 4: shares Docker atom → 1 hit, 1 miss (Kubernetes is new)
        "How to orchestrate Docker containers with Kubernetes?",
        # Query 5: all atoms likely cached by now
        "Deploy a React frontend and Flask backend with Docker on AWS",
    ]

    print("=" * 70)
    print("⚛  Intent Atoms — Demo")
    print("=" * 70)

    for i, query in enumerate(queries):
        print(f"\n{'─' * 70}")
        print(f"Query {i+1}: {query}")
        print(f"{'─' * 70}")

        result = await engine.query(query)

        print(f"\n📊 Results:")
        print(f"   Atoms: {result.total_atoms} total | {result.cache_hits} hits | {result.cache_misses} misses")
        print(f"   Hit rate: {result.cache_hit_rate:.1f}%")
        print(f"   Cost: ${result.estimated_cost:.6f} (vs ${result.estimated_cost_without_cache:.6f} without cache)")
        print(f"   Savings: {result.cost_savings_pct:.1f}%")
        print(f"   Tokens saved: {result.tokens_saved}")
        print(f"   Time: {result.total_time_ms:.0f}ms")
        print(f"\n💬 Response (first 200 chars):")
        print(f"   {result.response[:200]}...")

    # Print overall stats
    stats = await engine.get_stats()
    print(f"\n{'=' * 70}")
    print(f"📈 Overall Stats")
    print(f"{'=' * 70}")
    print(f"   Total atoms cached: {stats.total_atoms_stored}")
    print(f"   Total queries: {stats.total_queries_processed}")
    print(f"   Overall hit rate: {stats.overall_hit_rate:.1%}")
    print(f"   Total tokens saved: {stats.total_tokens_saved}")
    print(f"   Total cost saved: ${stats.total_cost_saved:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
