# agent/brain/demos/inspect_walkthrough.py
"""
Utility: Dumps every chunk stored in the walkthrough ``strategy_guide``
ChromaDB collection — the knowledge base that powers Phase 4's strategic
planner.

Shows each chunk's location heading, part number, section order, battle
flag, character count, and a text preview.

Usage (from project root):
    python -m agent.brain.demos.inspect_walkthrough
    python -m agent.brain.demos.inspect_walkthrough --full        # print full text per chunk
    python -m agent.brain.demos.inspect_walkthrough --location "Route 101"  # filter by location
"""
import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent.brain.walkthrough_db import WalkthroughDB


def _divider(char="─", width=72):
    return char * width


def inspect(full_text: bool = False, location_filter: str | None = None):
    print(_divider("═"))
    print("📖 WALKTHROUGH DB INSPECTOR")
    print(_divider("═"))

    db = WalkthroughDB(db_path="./memory_db")
    total = db.count()

    if total == 0:
        print("❌ strategy_guide collection is empty.")
        print("   Run:  python scripts/build_walkthrough_db.py --offline --rebuild")
        return

    # Fetch everything (ChromaDB get() returns all docs when no filter given)
    data = db.collection.get(include=["documents", "metadatas"])
    docs = data["documents"]
    metas = data["metadatas"]

    # Pair and sort by (part, section_order) for natural reading order
    entries = list(zip(docs, metas))
    entries.sort(key=lambda e: (e[1].get("part", 0), e[1].get("section_order", 0)))

    # Optional location filter
    if location_filter:
        entries = [
            (d, m) for d, m in entries
            if location_filter.lower() in m.get("location", "").lower()
        ]
        print(f"📍 Filter: '{location_filter}' — {len(entries)}/{total} chunks match\n")

    print(f"📚 Total chunks: {total}")
    print(f"📦 Parts present: {sorted(set(m.get('part', '?') for _, m in entries))}")
    print(f"⚔️  Chunks with battles: {sum(1 for _, m in entries if m.get('has_battle'))}")
    print()

    for i, (doc, meta) in enumerate(entries):
        loc = meta.get("location", "???")
        part = meta.get("part", "?")
        order = meta.get("section_order", "?")
        has_battle = "⚔️" if meta.get("has_battle") else "  "
        chars = len(doc)

        print(f"  [{i:>2}] Part {part}, §{order}  {has_battle}  {loc}")
        print(f"       {chars:,} chars", end="")

        if full_text:
            print()
            print(_divider("·"))
            for line in doc.splitlines():
                print(f"       {line}")
            print(_divider("·"))
        else:
            # Show first 120 chars as preview
            preview = doc.replace("\n", " ")[:120]
            print(f"  │ {preview}…" if len(doc) > 120 else f"  │ {preview}")

        print()

    print(_divider("═"))
    print(f"✅ {len(entries)} chunks displayed.")
    print(_divider("═"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect the walkthrough strategy_guide ChromaDB collection."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print the full text of every chunk (instead of a preview).",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="Filter by location name (substring, case-insensitive).",
    )
    args = parser.parse_args()
    inspect(full_text=args.full, location_filter=args.location)
