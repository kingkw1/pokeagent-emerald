# agent/brain/demos/demo_walkthrough_rag.py
"""
Phase 4 demo: Queries the walkthrough knowledge base with realistic
game-state questions and shows the full RAG → LLM → Directive pipeline.

Demonstrates:
  1. Semantic retrieval — does the right walkthrough chunk surface?
  2. Location resolution — does the prose name map to the correct graph key?
  3. Strategic planner (mock LLM) — does the pipeline produce a usable directive?

Usage (from project root):
    python -m agent.brain.demos.demo_walkthrough_rag
"""
import json
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent.brain.walkthrough_db import WalkthroughDB
from agent.brain.location_resolver import resolve_location, resolve_location_key
from agent.brain.strategic_planner import StrategicPlanner


def _header(title: str, char="═", width=72):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def _sub(title: str):
    print(f"\n  ── {title} {'─' * max(1, 55 - len(title))}")


# ──────────────────────────────────────────────────────────────────────
# 1. RAW RETRIEVAL: Show that the DB returns relevant chunks
# ──────────────────────────────────────────────────────────────────────

def demo_retrieval(db: WalkthroughDB):
    _header("PART 1: SEMANTIC RETRIEVAL")

    queries = [
        "I just started the game in Littleroot Town. What do I do first?",
        "How do I get through Route 101?",
        "I'm in Oldale Town. Where should I go next?",
        "Where do I battle my rival?",
        "How do I get to Petalburg City from Oldale Town?",
        "I need to go through Petalburg Woods. What should I expect?",
        "How do I beat the Rustboro Gym?",
    ]

    for q in queries:
        _sub(f"Query: {q}")
        results = db.query(q, n_results=2)

        if not results:
            print("    ❌ No results returned.")
            continue

        for j, r in enumerate(results):
            loc = r["metadata"].get("location", "?")
            part = r["metadata"].get("part", "?")
            dist = f"{r['distance']:.4f}" if r.get("distance") is not None else "N/A"
            preview = r["text"].replace("\n", " ")[:100]
            print(f"    [{j+1}] Part {part} · {loc}  (dist={dist})")
            print(f"        \"{preview}…\"")


# ──────────────────────────────────────────────────────────────────────
# 2. LOCATION RESOLUTION: Show the prose→graph bridge
# ──────────────────────────────────────────────────────────────────────

def demo_resolution():
    _header("PART 2: LOCATION RESOLUTION")

    test_names = [
        "Littleroot Town",
        "Route 101",
        "Oldale Town",
        "route 103",           # case-insensitive
        "Petalburg City",
        "Petalburg Woods",
        "Rustboro City",
        "Rustboro Gym",
        "ROUTE_102",           # direct graph key
        "Rustboro",            # fuzzy
        "Birch's Lab",         # alias
        "Dewford Town",        # future — not yet in graph
        "Mount Chimney",       # unknown
    ]

    print(f"\n  {'Input':<25} {'Resolved Key':<35} {'In Graph?'}")
    print(f"  {'─'*25} {'─'*35} {'─'*10}")

    for name in test_names:
        result = resolve_location(name)
        if result:
            key = result["key"]
            has_portals = "portals" in result
            print(f"  {name:<25} {key:<35} {'✅ yes' if has_portals else '⚠️  key only'}")
        else:
            print(f"  {name:<25} {'—':<35} ❌ no match")


# ──────────────────────────────────────────────────────────────────────
# 3. FULL PIPELINE: StrategicPlanner (mock LLM) → Directive
# ──────────────────────────────────────────────────────────────────────

def demo_strategic_planner(db: WalkthroughDB):
    _header("PART 3: FULL STRATEGIC PLANNER PIPELINE (mock LLM)")

    planner = StrategicPlanner(vlm=None, walkthrough_db=db, verbose=False)

    scenarios = [
        {
            "current_location": "LITTLEROOT_TOWN",
            "badge_count": 0,
            "pokemon_summary": "Treecko Lv.5",
            "last_milestone": "STARTER_CHOSEN",
        },
        {
            "current_location": "OLDALE_TOWN",
            "badge_count": 0,
            "pokemon_summary": "Treecko Lv.8",
            "last_milestone": "OLDALE_TOWN",
        },
        {
            "current_location": "ROUTE_102",
            "badge_count": 0,
            "pokemon_summary": "Treecko Lv.10, Zigzagoon Lv.4",
            "last_milestone": "ROUTE_102",
        },
        {
            "current_location": "PETALBURG_CITY",
            "badge_count": 0,
            "pokemon_summary": "Treecko Lv.12",
            "last_milestone": "PETALBURG_CITY",
        },
        {
            "current_location": "RUSTBORO_CITY",
            "badge_count": 0,
            "pokemon_summary": "Grovyle Lv.16, Lotad Lv.10",
            "last_milestone": "RUSTBORO_CITY",
        },
    ]

    for sc in scenarios:
        _sub(f"Location: {sc['current_location']} | Badges: {sc['badge_count']}")
        result = planner.get_next_directive(**sc)

        print(f"    🎯 Target:      {result.get('target_location')} "
              f"({result.get('target_display_name')})")
        print(f"    📝 Description: {result.get('description')}")
        print(f"    📋 Actions:     {result.get('priority_actions')}")
        if result.get("goal_coords"):
            print(f"    📍 Coords:      {result['goal_coords']}")
        else:
            print(f"    📍 Coords:      (not resolved)")
        print(f"    🔖 Source:      {result.get('source')}")


# ──────────────────────────────────────────────────────────────────────
# 4. QUERY_NEXT_STEPS convenience method
# ──────────────────────────────────────────────────────────────────────

def demo_next_steps(db: WalkthroughDB):
    _header("PART 4: query_next_steps() convenience")

    locations = ["Littleroot Town", "Oldale Town", "Petalburg City", "Rustboro City"]

    for loc in locations:
        _sub(f"query_next_steps(\"{loc}\")")
        results = db.query_next_steps(loc, n_results=2)
        for j, r in enumerate(results):
            meta_loc = r["metadata"].get("location", "?")
            dist = f"{r['distance']:.4f}" if r.get("distance") is not None else "N/A"
            preview = r["text"].replace("\n", " ")[:90]
            print(f"    [{j+1}] {meta_loc}  (dist={dist})")
            print(f"        \"{preview}…\"")


# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  🗺️  WALKTHROUGH RAG DEMO — Phase 4")
    print("  Demonstrates retrieval, resolution, and the strategic planner")
    print("=" * 72)

    db = WalkthroughDB(db_path="./memory_db")
    count = db.count()
    print(f"\n  📦 strategy_guide collection: {count} chunks")

    if count == 0:
        print("\n  ❌ Collection is empty. Build it first:")
        print("     python scripts/build_walkthrough_db.py --offline --rebuild")
        return

    demo_retrieval(db)
    demo_resolution()
    demo_strategic_planner(db)
    demo_next_steps(db)

    _header("DEMO COMPLETE", "═")
    print("  All four stages passed.  The walkthrough RAG pipeline is functional.\n")


if __name__ == "__main__":
    main()
