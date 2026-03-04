# agent/brain/memory.py
import chromadb
import uuid
import time
import logging
from typing import Any, Dict, Optional

from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


def extract_spatial_metadata(state_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract ``(x, y, location)`` from the emulator's ``state_data`` dict.

    Returns a flat dict suitable for merging into ChromaDB metadata:

    .. code-block:: python

        {"pos_x": 5, "pos_y": 12, "location": "ROUTE_102"}

    Missing or ``None`` fields are omitted so ChromaDB never stores
    ``None`` values (which it rejects).
    """
    spatial: Dict[str, Any] = {}

    player = state_data.get("player", {}) if state_data else {}
    position = player.get("position") or {}

    if isinstance(position, dict):
        x = position.get("x")
        y = position.get("y")
    elif isinstance(position, (list, tuple)) and len(position) >= 2:
        x, y = position[0], position[1]
    else:
        x, y = None, None

    if x is not None:
        spatial["pos_x"] = int(x)
    if y is not None:
        spatial["pos_y"] = int(y)

    location = player.get("location")
    if location:
        spatial["location"] = str(location)

    return spatial


class EpisodicMemory:
    """
    Long-term semantic memory using ChromaDB.
    Allows the agent to log events and retrieve relevant context via vector search.
    """
    def __init__(self, db_path="./memory_db"):
        logger.info(f"[Memory] Initializing VectorDB at {db_path}...")
        
        # 1. Initialize Client (Persistent storage)
        self.client = chromadb.PersistentClient(path=db_path)
        
        # 2. Initialize Embedding Function
        # We use 'all-MiniLM-L6-v2': Fast, local, and good for short sentences.
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # 3. Get or Create Collection
        self.collection = self.client.get_or_create_collection(
            name="game_history",
            embedding_function=self.embedding_fn
        )
        logger.info(f"[Memory] System Online. Items in DB: {self.collection.count()}")

    def log_event(self, text: str, metadata: dict = None,
                  state_data: dict = None):
        """Writes a significant event to the VectorDB.

        Args:
            text: Human-readable event description.
            metadata: Caller-supplied metadata dict (copied, not mutated).
            state_data: Optional emulator ``state_data`` dict. When provided
                the player's ``(x, y, location)`` are automatically merged
                into *metadata* (Phase 3 — Spatial Awareness).
        """
        # Copy to avoid mutating the caller's dict
        metadata = dict(metadata) if metadata else {}

        # ── Phase 3: Spatial Awareness ──
        if state_data is not None:
            spatial = extract_spatial_metadata(state_data)
            # Caller-supplied keys win; spatial fills gaps only.
            for k, v in spatial.items():
                metadata.setdefault(k, v)

        # Ensure timestamp exists
        if "timestamp" not in metadata:
            metadata["timestamp"] = time.time()

        # Add a UUID to ensure every log is unique
        doc_id = str(uuid.uuid4())

        try:
            self.collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )
        except Exception as e:
            logger.error(f"Failed to log event: {e}")

    def retrieve_raw(self, query: str, n_results: int = 3) -> list[dict]:
        """
        Performs a Semantic Search and returns structured results.
        Each dict contains 'text', 'metadata', and 'distance'.
        """
        count = self.collection.count()
        if count == 0:
            return []

        n_results = min(n_results, count)

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0] if results.get('distances') else [None] * len(documents)

        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

    def retrieve_relevant(self, query: str, n_results: int = 3, max_distance: float | None = None) -> str:
        """
        Performs a Semantic Search.
        Returns a formatted string of the top N relevant memories.

        Args:
            max_distance: If set, drops results whose cosine distance exceeds
                          this threshold (lower = more similar). Typical useful
                          range for MiniLM is 0.8–1.3.
        """
        entries = self.retrieve_raw(query, n_results)

        if max_distance is not None:
            entries = [e for e in entries if e["distance"] is not None and e["distance"] <= max_distance]

        if not entries:
            return "No relevant memories found."

        # Format as a bulleted list for the LLM
        context_string = "\n".join([f"- {e['text']}" for e in entries])
        return context_string

    # ── Phase 3: Spatial queries ────────────────────────────────────
    def retrieve_near(
        self,
        query: str,
        *,
        location: Optional[str] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        radius: int = 10,
        n_results: int = 5,
        max_distance: Optional[float] = None,
    ) -> list[dict]:
        """Retrieve memories filtered or ranked by spatial proximity.

        Combines semantic search with optional metadata filtering:

        * **location** — hard-filter via ChromaDB ``where`` clause so only
          memories from that map are considered.
        * **x / y / radius** — post-filter: after the semantic query,
          discard results whose ``pos_x``/``pos_y`` lie outside the L∞
          (Chebyshev) radius.  Memories without spatial metadata are kept.

        Returns the same ``[{"text", "metadata", "distance"}, ...]``
        format as ``retrieve_raw``.
        """
        where_filter = None
        if location:
            where_filter = {"location": location}

        count = self.collection.count()
        if count == 0:
            return []

        # Fetch more candidates than requested to allow post-filtering
        fetch_n = min(n_results * 3, count)

        kwargs: dict = {
            "query_texts": [query],
            "n_results": fetch_n,
        }
        if where_filter:
            kwargs["where"] = where_filter

        try:
            results = self.collection.query(**kwargs)
        except Exception as e:
            logger.warning(f"[Memory] Spatial query failed: {e}")
            return []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0] if results.get("distances") else [None] * len(documents)

        entries = [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]

        # Optional cosine-distance ceiling
        if max_distance is not None:
            entries = [
                e for e in entries
                if e["distance"] is None or e["distance"] <= max_distance
            ]

        # Optional coordinate-radius post-filter (L∞ / Chebyshev)
        if x is not None and y is not None:
            def _within_radius(entry: dict) -> bool:
                meta = entry.get("metadata", {})
                mx = meta.get("pos_x")
                my = meta.get("pos_y")
                if mx is None or my is None:
                    return True  # Keep memories without coords
                return abs(mx - x) <= radius and abs(my - y) <= radius
            entries = [e for e in entries if _within_radius(e)]

        return entries[:n_results]

    def clear_memory(self):
        """Debug tool to wipe history (Used for fresh runs/demos)"""
        try:
            self.client.delete_collection("game_history")
            self.collection = self.client.get_or_create_collection(
                name="game_history",
                embedding_function=self.embedding_fn
            )
            logger.info("Database Wiped.")
        except Exception as e:
            logger.error(f"Error clearing DB: {e}")