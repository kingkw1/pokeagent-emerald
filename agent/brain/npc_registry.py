# agent/brain/npc_registry.py
"""
NPC Identity Registry — Adaptive Discovery (Phase 4.4d)

Replaces hardcoded graphics_id constants with a learned registry that maps
semantic NPC roles to their runtime identifiers.  The agent discovers NPC
identities through interaction and stores them for future lookups.

Three-phase resolution:
  A) OBSERVE — on dialogue start, record (location, local_id) → identity
  B) RECALL  — resolve "rival" / "nurse" etc. from the registry
  C) INFER   — cold start heuristic: nearest non-player NPC
"""
from __future__ import annotations

import json
import logging
import os
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Role aliases ──────────────────────────────────────────────────
# Maps common directive phrases to canonical role strings.
# Handles plurals, alternate spellings, and descriptive phrases.
ROLE_ALIASES: Dict[str, str] = {
    # Rival
    "rival": "rival",
    "may": "rival",
    "brendan": "rival",
    "the rival": "rival",
    # Nurse Joy (any Pokémon Center)
    "nurse": "nurse",
    "nurse joy": "nurse",
    "pokemon center nurse": "nurse",
    # Norman / Petalburg Gym
    "norman": "gym_leader_norman",
    "dad": "gym_leader_norman",
    "father": "gym_leader_norman",
    # Roxanne / Rustboro Gym
    "roxanne": "gym_leader_roxanne",
    # Prof. Birch
    "birch": "professor_birch",
    "professor birch": "professor_birch",
    "prof. birch": "professor_birch",
    "prof birch": "professor_birch",
    # Wally
    "wally": "wally",
    # Mr. Briney
    "mr. briney": "mr_briney",
    "briney": "mr_briney",
}


def _canon_role(role: str) -> str:
    """Normalise a role string to its canonical form."""
    key = role.strip().lower()
    return ROLE_ALIASES.get(key, key)


def _canon_name(name: str) -> str:
    """Normalise an NPC name for matching (strip, lower, collapse spaces)."""
    return " ".join(name.strip().lower().split())


class NpcRegistry:
    """
    Persistent registry mapping ``(location, local_id)`` to NPC identity.

    Storage is a flat JSON file so the agent can "discover fresh" by simply
    deleting the file.  An optional ``EpisodicMemory`` reference is used to
    log discoveries for semantic retrieval backup.

    Key format:  ``"<location>:<local_id>"``  e.g. ``"ROUTE_103:2"``
    Value:       ``{name, role, graphics_id, first_seen_step}``
    """

    def __init__(
        self,
        json_path: str = "./memory_db/npc_registry.json",
        episodic_memory=None,
    ):
        self.json_path = json_path
        self.episodic_memory = episodic_memory
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────

    def _load(self):
        """Load registry from JSON file if it exists."""
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r") as f:
                    self._data = json.load(f)
                logger.info(f"[NPC REGISTRY] Loaded {len(self._data)} entries from {self.json_path}")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"[NPC REGISTRY] Failed to load {self.json_path}: {e}")
                self._data = {}
        else:
            self._data = {}

    def _save(self):
        """Persist the registry to JSON."""
        try:
            os.makedirs(os.path.dirname(self.json_path) or ".", exist_ok=True)
            with open(self.json_path, "w") as f:
                json.dump(self._data, f, indent=2)
        except OSError as e:
            logger.error(f"[NPC REGISTRY] Failed to save: {e}")

    @staticmethod
    def _key(location: str, local_id: int) -> str:
        return f"{location}:{local_id}"

    # ── Core API ──────────────────────────────────────────────────

    def register_npc(
        self,
        location: str,
        local_id: int,
        *,
        name: Optional[str] = None,
        role: Optional[str] = None,
        graphics_id: Optional[int] = None,
        step: int = 0,
    ) -> None:
        """
        Upsert an NPC identity into the registry.

        Called by the observation hook when dialogue starts after interacting
        with an NPC.  Fields that are ``None`` will NOT overwrite existing
        non-None values (merge semantics).
        """
        key = self._key(location, local_id)
        existing = self._data.get(key, {})

        entry = {
            "location": location,
            "local_id": local_id,
            "name": name or existing.get("name"),
            "role": _canon_role(role) if role else existing.get("role"),
            "graphics_id": graphics_id if graphics_id is not None else existing.get("graphics_id"),
            "first_seen_step": existing.get("first_seen_step", step),
        }

        is_new = key not in self._data
        self._data[key] = entry
        self._save()

        action = "Registered" if is_new else "Updated"
        logger.info(f"[NPC REGISTRY] {action}: {key} → {entry}")

        # Log to EpisodicMemory for semantic retrieval backup
        if self.episodic_memory and name:
            self.episodic_memory.log_event(
                f"Discovered NPC: {name} (role={entry.get('role')}) "
                f"at {location} local_id={local_id} graphics_id={graphics_id}",
                metadata={
                    "type": "npc_discovery",
                    "npc_name": _canon_name(name),
                    "npc_role": entry.get("role", ""),
                    "location": location,
                    "local_id": local_id,
                    "graphics_id": graphics_id or 0,
                },
            )

    def lookup_by_role(
        self,
        role: str,
        location: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Find a registered NPC by semantic role.

        If ``location`` is given, prefer entries from that map.  Otherwise
        return the first match across all locations.

        Returns ``{name, role, graphics_id, local_id, location}`` or ``None``.
        """
        canon = _canon_role(role)

        # Priority: same location first
        if location:
            for entry in self._data.values():
                if entry.get("role") == canon and entry.get("location") == location:
                    return dict(entry)

        # Fallback: any location
        for entry in self._data.values():
            if entry.get("role") == canon:
                return dict(entry)

        return None

    def lookup_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find a registered NPC by name (case-insensitive)."""
        target = _canon_name(name)
        for entry in self._data.values():
            entry_name = entry.get("name")
            if entry_name and _canon_name(entry_name) == target:
                return dict(entry)
        return None

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Return a copy of the full registry."""
        return dict(self._data)

    def clear(self) -> None:
        """Wipe all entries (for fresh-start tests)."""
        self._data = {}
        self._save()
        logger.info("[NPC REGISTRY] Cleared all entries")

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    # ── Cold-Start Heuristic (Phase C) ────────────────────────────

    @staticmethod
    def infer_nearest_npc(
        active_npcs: List[Dict[str, Any]],
        player_x: int,
        player_y: int,
        *,
        max_distance: int = 12,
    ) -> Optional[Dict[str, Any]]:
        """
        Cold-start fallback: find the nearest non-player, visible NPC.

        Used when the registry has no match for the requested role and the
        agent needs *something* to walk toward.

        Returns the NPC dict from ``active_npcs`` or ``None``.
        """
        best = None
        best_dist = float("inf")

        for npc in active_npcs:
            if npc.get("is_player"):
                continue
            if npc.get("invisible") or npc.get("off_screen"):
                continue

            dx = npc.get("current_x", 0) - player_x
            dy = npc.get("current_y", 0) - player_y
            dist = abs(dx) + abs(dy)  # Manhattan distance

            if dist < best_dist and dist <= max_distance:
                best_dist = dist
                best = npc

        if best:
            logger.info(
                f"[NPC REGISTRY] Cold-start infer: nearest NPC at "
                f"({best.get('current_x')}, {best.get('current_y')}) "
                f"dist={best_dist}, gfx={best.get('graphics_id')}"
            )
        return best
