#!/usr/bin/env python3
"""
Dump the strategy_guide ChromaDB collection to a JSON file.

Usage::

    python scripts/dump_walkthrough_db.py                         # → data/walkthrough_db_dump.json
    python scripts/dump_walkthrough_db.py --out /tmp/my_dump.json
    python scripts/dump_walkthrough_db.py --query "Route 102 navigation"  # semantic search
    python scripts/dump_walkthrough_db.py --location "Route 102"          # filter by location metadata
    python scripts/dump_walkthrough_db.py --supplemental-only             # only SUPPLEMENTAL chunks
    python scripts/dump_walkthrough_db.py --stats                         # summary counts only
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.brain.walkthrough_db import WalkthroughDB

logging.basicConfig(level=logging.WARNING)


def dump_all(db: WalkthroughDB) -> list[dict]:
    """Retrieve every document in the collection."""
    result = db.collection.get(include=["documents", "metadatas"])
    docs = result.get("documents", [])
    metas = result.get("metadatas", [])
    ids = result.get("ids", [])
    return [
        {"id": id_, "metadata": meta, "text": doc}
        for id_, meta, doc in zip(ids, metas, docs)
    ]


def dump_query(db: WalkthroughDB, query: str, n: int = 10) -> list[dict]:
    """Semantic search — return top-n results with distance scores."""
    results = db.query(query, n_results=n)
    return [
        {
            "distance": r["distance"],
            "metadata": r["metadata"],
            "text": r["text"],
        }
        for r in results
    ]


def print_stats(entries: list[dict]) -> None:
    from collections import Counter

    parts = Counter(e["metadata"].get("part") for e in entries)
    supplemental = sum(1 for e in entries if e["metadata"].get("supplemental"))
    has_battle = sum(1 for e in entries if e["metadata"].get("has_battle"))

    print(f"Total chunks : {len(entries)}")
    print(f"Supplemental : {supplemental}")
    print(f"Has battle   : {has_battle}")
    print(f"\nChunks per walkthrough part:")
    for part, count in sorted(parts.items()):
        label = f"Part {part}" if part is not None else "None"
        print(f"  {label:8s}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Dump the strategy_guide ChromaDB collection to JSON."
    )
    parser.add_argument(
        "--out",
        default="data/walkthrough_db_dump.json",
        help="Output file path (default: data/walkthrough_db_dump.json).",
    )
    parser.add_argument(
        "--db-path",
        default="./memory_db",
        help="Path to ChromaDB persistent storage (default: ./memory_db).",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Run a semantic search query instead of dumping all chunks.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of results to return for --query (default: 10).",
    )
    parser.add_argument(
        "--location",
        default=None,
        help="Filter output to chunks whose 'location' metadata matches this value.",
    )
    parser.add_argument(
        "--supplemental-only",
        action="store_true",
        help="Only output chunks with supplemental=True metadata.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print summary statistics instead of writing a file.",
    )
    args = parser.parse_args()

    db = WalkthroughDB(db_path=args.db_path)
    print(f"Collection '{WalkthroughDB.COLLECTION_NAME}': {db.count()} chunks", file=sys.stderr)

    if args.query:
        entries = dump_query(db, args.query, n=args.n)
        print(f"Query: {args.query!r}  →  {len(entries)} results", file=sys.stderr)
    else:
        entries = dump_all(db)

    # Post-filter
    if args.location:
        entries = [e for e in entries if e.get("metadata", {}).get("location") == args.location]
        print(f"Filtered to location={args.location!r}: {len(entries)} chunks", file=sys.stderr)

    if args.supplemental_only:
        entries = [e for e in entries if e.get("metadata", {}).get("supplemental")]
        print(f"Filtered to supplemental only: {len(entries)} chunks", file=sys.stderr)

    if args.stats:
        print_stats(entries)
        return

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(entries)} chunks → {out_path}")


if __name__ == "__main__":
    main()
