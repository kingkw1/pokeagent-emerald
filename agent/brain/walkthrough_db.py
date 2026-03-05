# agent/brain/walkthrough_db.py
"""
Walkthrough Knowledge Base — Phase 4.1

A dedicated ChromaDB collection (``strategy_guide``) that stores chunked
walkthrough text from Bulbapedia.  Separate from the ``episodic_memory``
collection used for recovery planning.

Usage::

    from agent.brain.walkthrough_db import WalkthroughDB

    db = WalkthroughDB()                 # uses default persistent path
    db.add_chunk("Head north through ...", metadata={...})
    results = db.query("I am in Littleroot Town, what do I do next?", n=3)
"""

import logging
import re
import uuid
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

# ============================================================================
# MediaWiki text preprocessing helpers
# ============================================================================

# Match {{Template|...}} constructs (non-greedy, single-line)
_RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
# Match [[File:...|...]] and [[Image:...|...]]
_RE_FILE_LINK = re.compile(r"\[\[(?:File|Image):[^\]]*\]\]", re.IGNORECASE)
# Match remaining [[Link|display]] → keep display text
_RE_WIKI_LINK = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
# Match {| ... |} wiki tables (multiline)
_RE_TABLE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
# Match HTML-style tags
_RE_HTML = re.compile(r"<[^>]+>")
# Match category links [[Category:...]]
_RE_CATEGORY = re.compile(r"\[\[Category:[^\]]*\]\]", re.IGNORECASE)
# Collapse repeated blank lines
_RE_BLANK_LINES = re.compile(r"\n{3,}")


def clean_wikitext(raw: str) -> str:
    """Strip MediaWiki markup, keeping prose and item lists.

    Processing order matters — templates must be removed before wiki-links
    so that nested constructs don't survive partially.
    """
    text = raw
    # Remove category links first (they look like wiki-links)
    text = _RE_CATEGORY.sub("", text)
    # Remove file/image links
    text = _RE_FILE_LINK.sub("", text)
    # Remove templates (may need multiple passes for nested templates)
    for _ in range(5):
        new_text = _RE_TEMPLATE.sub("", text)
        if new_text == text:
            break
        text = new_text
    # Remove wiki tables
    text = _RE_TABLE.sub("", text)
    # Remove HTML tags
    text = _RE_HTML.sub("", text)
    # Resolve wiki-links: [[Route 101|the route]] → the route
    text = _RE_WIKI_LINK.sub(r"\1", text)
    # Remove bold/italic markers
    text = text.replace("'''", "").replace("''", "")
    # Collapse blank lines
    text = _RE_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def chunk_wikitext(raw: str, part_number: int) -> List[Dict[str, Any]]:
    """Split raw wikitext into chunks at ``==Heading==`` boundaries.

    Each chunk gets metadata::

        {
            "location":      "Route 101",     # from the heading
            "part":          1,                # walkthrough part number
            "section_order": 3,                # position within the part
            "has_battle":    True,             # heuristic: trainer/battle in text
        }

    Returns a list of ``{"text": ..., "metadata": {...}}`` dicts.
    """
    # Split on level-2 headings (==Heading==) keeping the heading text
    heading_pattern = re.compile(r"^==\s*([^=]+?)\s*==\s*$", re.MULTILINE)
    parts = heading_pattern.split(raw)

    chunks: List[Dict[str, Any]] = []
    # parts[0] is text before the first heading (usually empty / intro)
    section_order = 0

    if parts[0].strip():
        intro_text = clean_wikitext(parts[0])
        if len(intro_text) > 50:
            chunks.append({
                "text": intro_text,
                "metadata": {
                    "location": "Introduction",
                    "part": part_number,
                    "section_order": section_order,
                    "has_battle": False,
                },
            })
        section_order += 1

    # Iterate heading/body pairs
    i = 1
    while i < len(parts) - 1:
        heading = parts[i].strip()
        body_raw = parts[i + 1]
        body = clean_wikitext(body_raw)

        if len(body) < 20:
            # Skip trivially short sections
            i += 2
            section_order += 1
            continue

        # Heuristic: detect battles
        battle_keywords = [
            "battle", "trainer", "fight", "defeat", "gym leader",
            "wild pokémon", "wild pokemon", "lv.", "level ",
        ]
        has_battle = any(kw in body.lower() for kw in battle_keywords)

        # Also absorb level-3 sub-headings (===Sub===) into this chunk
        # They are already part of the body text after splitting on ==
        chunks.append({
            "text": f"{heading}\n\n{body}",
            "metadata": {
                "location": heading,
                "part": part_number,
                "section_order": section_order,
                "has_battle": has_battle,
            },
        })

        i += 2
        section_order += 1

    return chunks


# ============================================================================
# Supplemental navigation chunks for early-game gaps
# ============================================================================
# The Bulbapedia walkthrough focuses on items/encounters for early routes
# and omits explicit "go here next" guidance.  These hand-written chunks
# cover the critical Route 101 → Oldale → Route 103 → rival battle →
# Route 102 → Petalburg progression so the RAG retriever returns relevant
# context even at the start of the game.
# ============================================================================

# ============================================================================
# WalkthroughDB class
# ============================================================================

class WalkthroughDB:
    """ChromaDB-backed walkthrough knowledge base (``strategy_guide`` collection).

    Uses the same embedding model (``all-MiniLM-L6-v2``) as ``EpisodicMemory``
    for consistency and to avoid loading a second model.
    """

    COLLECTION_NAME = "strategy_guide"

    def __init__(self, db_path: str = "./memory_db"):
        logger.info(f"[WalkthroughDB] Initializing at {db_path}...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )
        logger.info(
            f"[WalkthroughDB] Online. Chunks in DB: {self.collection.count()}"
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_chunk(self, text: str, metadata: Dict[str, Any]) -> str:
        """Add a single walkthrough chunk with metadata. Returns the doc ID."""
        doc_id = str(uuid.uuid4())
        # ChromaDB metadata values must be str, int, float, or bool
        safe_meta = {
            k: v for k, v in metadata.items()
            if isinstance(v, (str, int, float, bool))
        }
        self.collection.add(
            documents=[text],
            metadatas=[safe_meta],
            ids=[doc_id],
        )
        return doc_id

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Batch-add chunks from ``chunk_wikitext()`` output. Returns count added."""
        added = 0
        for chunk in chunks:
            self.add_chunk(chunk["text"], chunk["metadata"])
            added += 1
        return added

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        n_results: int = 3,
        location_filter: Optional[str] = None,
        min_section_order: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over walkthrough chunks.

        Args:
            query_text: Natural-language query.
            n_results: Max results to return.
            location_filter: If set, restrict to chunks whose ``location``
                metadata matches this value (exact match).
            min_section_order: If set, only return chunks with
                ``section_order >= min_section_order`` (for "what comes next").

        Returns:
            List of ``{"text": str, "metadata": dict, "distance": float}``.
        """
        count = self.collection.count()
        if count == 0:
            return []

        n = min(n_results * 3, count)  # over-fetch for post-filtering

        where_filter = None
        if location_filter:
            where_filter = {"location": location_filter}

        kwargs: Dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": n,
        }
        if where_filter:
            kwargs["where"] = where_filter

        try:
            results = self.collection.query(**kwargs)
        except Exception as exc:
            logger.warning(f"[WalkthroughDB] Query failed: {exc}")
            return []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = (
            results["distances"][0]
            if results.get("distances")
            else [None] * len(documents)
        )

        entries = [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

        # Post-filter: section_order
        if min_section_order is not None:
            entries = [
                e for e in entries
                if e["metadata"].get("section_order", 0) >= min_section_order
            ]

        return entries[:n_results]

    def query_next_steps(
        self,
        current_location: str,
        n_results: int = 3,
    ) -> List[Dict[str, Any]]:
        """Convenience: retrieve chunks relevant to "what do I do in/after {location}"."""
        query = (
            f"What should the player do in or after {current_location}? "
            f"Where should they go next?"
        )
        return self.query(query, n_results=n_results)

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Number of chunks in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete and recreate the collection (for rebuilds)."""
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )
        logger.info("[WalkthroughDB] Collection cleared.")

    def peek(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return first *n* entries for inspection."""
        result = self.collection.peek(limit=n)
        docs = result.get("documents", [])
        metas = result.get("metadatas", [])
        return [
            {"text": d, "metadata": m}
            for d, m in zip(docs, metas)
        ]
