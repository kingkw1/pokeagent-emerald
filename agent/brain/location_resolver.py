# agent/brain/location_resolver.py
"""
Location Name Resolver — Phase 4.4

Bridges walkthrough prose (e.g., "Littleroot Town") to the navigation
system's ``LOCATION_GRAPH`` keys (e.g., ``"LITTLEROOT_TOWN"``).

The LLM should only output *location names*, never coordinates.  This
module resolves those names to exact graph keys + coordinate data so
the A* pathfinder can consume them.
"""

import logging
from difflib import get_close_matches
from typing import Any, Dict, List, Optional

from agent.location_graph import LOCATION_GRAPH

logger = logging.getLogger(__name__)

# ============================================================================
# ALIAS TABLE — maps human-readable walkthrough names to LOCATION_GRAPH keys.
# Extend this as walkthrough coverage grows beyond Rustboro.
# ============================================================================

LOCATION_ALIASES: Dict[str, str] = {
    # Littleroot / Route 101 / Oldale
    "Littleroot Town": "LITTLEROOT_TOWN",
    "Player's House": "PLAYERS_HOUSE_1F",
    "Player's House 1F": "PLAYERS_HOUSE_1F",
    "Player's House 2F": "PLAYERS_HOUSE_2F",
    "May's House": "MAYS_HOUSE_1F",
    "Rival's House": "MAYS_HOUSE_1F",
    "Professor Birch's Lab": "PROFESSOR_BIRCHS_LAB",
    "Birch's Lab": "PROFESSOR_BIRCHS_LAB",
    "Birch Lab": "PROFESSOR_BIRCHS_LAB",
    "Route 101": "ROUTE_101",
    "Oldale Town": "OLDALE_TOWN",

    # Route 103 (Rival Battle)
    "Route 103": "ROUTE_103",

    # Route 102 / Petalburg
    "Route 102": "ROUTE_102",
    "Petalburg City": "PETALBURG_CITY",
    "Petalburg Gym": "PETALBURG_CITY_GYM",
    "Petalburg City Gym": "PETALBURG_CITY_GYM",
    "Norman's Gym": "PETALBURG_CITY_GYM",

    # Route 104 / Woods / Rustboro
    "Route 104": "ROUTE_104_SOUTH",
    "Route 104 South": "ROUTE_104_SOUTH",
    "Route 104 (South)": "ROUTE_104_SOUTH",
    "Route 104 North": "ROUTE_104_NORTH",
    "Route 104 (North)": "ROUTE_104_NORTH",
    "Petalburg Woods": "PETALBURG_WOODS",
    "Rustboro City": "RUSTBORO_CITY",
    "Rustboro Gym": "RUSTBORO_CITY_GYM",
    "Rustboro City Gym": "RUSTBORO_CITY_GYM",
    "Roxanne's Gym": "RUSTBORO_CITY_GYM",

    # Pokemon Centers
    "Oldale Pokemon Center": "OLDALE_TOWN_POKEMON_CENTER_1F",
    "Oldale Town Pokemon Center": "OLDALE_TOWN_POKEMON_CENTER_1F",
    "Petalburg Pokemon Center": "PETALBURG_CITY_POKEMON_CENTER_1F",
    "Petalburg City Pokemon Center": "PETALBURG_CITY_POKEMON_CENTER_1F",
    "Rustboro Pokemon Center": "RUSTBORO_CITY_POKEMON_CENTER_1F",
    "Rustboro City Pokemon Center": "RUSTBORO_CITY_POKEMON_CENTER_1F",

    # Marts
    "Oldale Mart": "OLDALE_TOWN_MART",
    "Oldale Town Mart": "OLDALE_TOWN_MART",

    # Future locations (stubs — will be populated as LOCATION_GRAPH grows)
    "Dewford Town": "DEWFORD_TOWN",
    "Route 105": "ROUTE_105",
    "Route 106": "ROUTE_106",
    "Route 109": "ROUTE_109",
    "Route 110": "ROUTE_110",
    "Slateport City": "SLATEPORT_CITY",
    "Mauville City": "MAUVILLE_CITY",
    "Route 111": "ROUTE_111",
    "Route 112": "ROUTE_112",
    "Route 117": "ROUTE_117",
    "Verdanturf Town": "VERDANTURF_TOWN",
    "Fallarbor Town": "FALLARBOR_TOWN",
    "Route 113": "ROUTE_113",
    "Route 114": "ROUTE_114",
    "Meteor Falls": "METEOR_FALLS",
    "Lavaridge Town": "LAVARIDGE_TOWN",
    "Fortree City": "FORTREE_CITY",
    "Route 119": "ROUTE_119",
    "Route 120": "ROUTE_120",
    "Lilycove City": "LILYCOVE_CITY",
    "Mossdeep City": "MOSSDEEP_CITY",
    "Sootopolis City": "SOOTOPOLIS_CITY",
    "Ever Grande City": "EVER_GRANDE_CITY",
    "Victory Road": "VICTORY_ROAD",
}

# Build reverse lookup: LOCATION_GRAPH key → canonical display name
_KEY_TO_DISPLAY: Dict[str, str] = {}
for _alias, _key in LOCATION_ALIASES.items():
    # Prefer the shortest alias per key (usually the canonical name)
    if _key not in _KEY_TO_DISPLAY or len(_alias) < len(_KEY_TO_DISPLAY[_key]):
        _KEY_TO_DISPLAY[_key] = _alias


def resolve_location(name: str) -> Optional[Dict[str, Any]]:
    """Map a prose location name to a ``LOCATION_GRAPH`` entry with coordinates.

    Resolution order:
    1. **Exact alias match** — O(1) dict lookup.
    2. **Case-insensitive alias match** — handles "route 101" vs "Route 101".
    3. **Direct graph key** — e.g., ``"ROUTE_101"`` passed through directly.
    4. **Fuzzy match** — ``difflib.get_close_matches`` with cutoff 0.65.

    Returns:
        ``{"key": "ROUTE_101", "display_name": "Route 101", ...}`` or
        ``None`` if no match is found.
    """
    if not name or not isinstance(name, str):
        return None

    name_stripped = name.strip()

    # 1. Exact alias match
    key = LOCATION_ALIASES.get(name_stripped)

    # 2. Case-insensitive
    if not key:
        name_lower = name_stripped.lower()
        for alias, alias_key in LOCATION_ALIASES.items():
            if alias.lower() == name_lower:
                key = alias_key
                break

    # 3. Direct graph key (e.g., "ROUTE_101")
    if not key and name_stripped.upper().replace(" ", "_") in LOCATION_GRAPH:
        key = name_stripped.upper().replace(" ", "_")

    # 4. Fuzzy match
    if not key:
        candidates = list(LOCATION_ALIASES.keys())
        matches = get_close_matches(name_stripped, candidates, n=1, cutoff=0.65)
        if matches:
            key = LOCATION_ALIASES[matches[0]]
            logger.info(
                f"[LocationResolver] Fuzzy resolved '{name_stripped}' → "
                f"'{matches[0]}' → {key}"
            )

    # Validate against graph
    if key and key in LOCATION_GRAPH:
        entry = LOCATION_GRAPH[key]
        return {"key": key, **entry}

    # Key exists in aliases but not yet in the graph (future location)
    if key:
        logger.warning(
            f"[LocationResolver] '{name_stripped}' resolved to key '{key}' "
            f"but that key is not in LOCATION_GRAPH yet."
        )
        return None

    logger.warning(f"[LocationResolver] Could not resolve location: '{name_stripped}'")
    return None


def resolve_location_key(name: str) -> Optional[str]:
    """Convenience: return just the ``LOCATION_GRAPH`` key string, or ``None``."""
    result = resolve_location(name)
    return result["key"] if result else None


def get_display_name(graph_key: str) -> str:
    """Return the human-readable display name for a ``LOCATION_GRAPH`` key.

    Falls back to the key itself if no alias entry exists.
    """
    # Check the graph's own display_name first
    entry = LOCATION_GRAPH.get(graph_key, {})
    if entry.get("display_name"):
        return entry["display_name"]
    return _KEY_TO_DISPLAY.get(graph_key, graph_key)


def list_known_locations() -> List[str]:
    """Return all alias names that currently resolve to a valid graph key."""
    return sorted(
        alias for alias, key in LOCATION_ALIASES.items()
        if key in LOCATION_GRAPH
    )
