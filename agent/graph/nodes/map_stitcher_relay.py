"""
agent/graph/nodes/map_stitcher_relay — MapStitcherRelay factory.

Routing contract:
  - Receives: AgentState where ``context == "healing_needed"`` (party HP low).
  - Returns: updated AgentState with:
      ``goal_coords``   — tile coordinates of the nearest PokeCenter entrance
      ``goal_location`` — location_graph key of the PokeCenter (if known)
      ``context``       — ``"navigation"``
      ``last_action``   — ``"HEAL_ROUTE"``

PokéCenter lookup (priority order):
  1. ``location_graph`` coordinate lookup — deterministic, zero VLM cost.
     Uses ``find_nearest_pokemon_center()`` BFS + ``get_entrance_coords()``.
     Covers all cities/routes already in the graph (Oldale Town, Petalburg
     City, Rustboro City, and any location that BFS-connects to them).
  2. VLM overhead-map fallback — for cities not yet mapped in location_graph.
     Queries the stitched overhead image (or raw GBA frame) for the PC pixel
     location and converts to tile coordinates.

GBA screen constants: 240 × 160 px, 16 px per tile, player at (120, 80).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from agent.graph.state import AgentState
from agent.location_graph import find_nearest_pokemon_center, get_entrance_coords
from utils.map_stitcher_singleton import get_instance

logger = logging.getLogger(__name__)

_SCREEN_CENTER_X: int = 120   # px — horizontal centre of the GBA screen
_SCREEN_CENTER_Y: int = 80    # px — vertical centre of the GBA screen
_TILE_SIZE_PX: int = 16        # pixels per tile

_VLM_PROMPT: str = (
    "This is the player's current known map. "
    "Mark the pixel location of the nearest Pokemon Center entrance. "
    'Respond as JSON: {"center_x": int, "center_y": int}'
)


def make_map_stitcher_relay_node(vlm: Any) -> Callable[[AgentState], AgentState]:
    """Factory that binds a VLM client into a MapStitcherRelay node.

    Args:
        vlm: VLM client with a ``get_query(image, prompt)`` method that
             returns a dict with ``center_x`` and ``center_y`` keys.
             Only called when the location_graph lookup fails.

    Returns:
        A LangGraph-compatible node callable.
    """

    def map_stitcher_relay_node(state: AgentState) -> AgentState:
        state_data: Dict[str, Any] = state.get("state_data") or {}
        player_data: Dict[str, Any] = state_data.get("player", {})

        # Normalise location string: "Petalburg City" → "PETALBURG_CITY"
        raw_location: str = player_data.get("location", "")
        current_location: str = raw_location.upper().replace(" ", "_")

        # ------------------------------------------------------------------ #
        # Primary path: location_graph coordinate lookup                       #
        # ------------------------------------------------------------------ #
        pc_key: Optional[str] = find_nearest_pokemon_center(current_location)
        if pc_key:
            # Derive city key from PC key:
            # "PETALBURG_CITY_POKEMON_CENTER_1F" → "PETALBURG_CITY"
            city: str = pc_key.replace("_POKEMON_CENTER_1F", "")
            coords = get_entrance_coords(city, pc_key)
            if coords:
                logger.info(
                    "[MAP_STITCHER_RELAY] location_graph lookup: %s → %s %s",
                    current_location,
                    pc_key,
                    coords,
                )
                print(
                    f"💊 [MAP_STITCHER_RELAY] location_graph: {current_location} "
                    f"→ {pc_key} @ {coords}"
                )
                return {
                    **state,
                    "goal_coords": coords,
                    "goal_location": pc_key,
                    "context": "navigation",
                    "last_action": "HEAL_ROUTE",
                }

        # ------------------------------------------------------------------ #
        # Fallback: VLM on stitched overhead map                               #
        # (used for cities not yet added to location_graph)                    #
        # ------------------------------------------------------------------ #
        logger.info(
            "[MAP_STITCHER_RELAY] '%s' not resolved in location_graph — "
            "trying VLM fallback",
            current_location,
        )

        map_image = None
        try:
            stitcher = get_instance()
            if hasattr(stitcher, "get_overhead_image"):
                map_image = stitcher.get_overhead_image()
        except Exception as exc:
            logger.debug("[MAP_STITCHER_RELAY] Could not get overhead image: %s", exc)

        if map_image is None:
            # Fall back to the raw GBA frame captured this step.
            map_image = state.get("frame")

        if map_image is None:
            logger.warning(
                "[MAP_STITCHER_RELAY] No map image available — skipping relay."
            )
            return state

        try:
            response: Dict[str, Any] = vlm.get_query(map_image, _VLM_PROMPT)
        except Exception as exc:
            logger.warning("[MAP_STITCHER_RELAY] VLM query failed: %s", exc)
            return state

        if (
            not isinstance(response, dict)
            or "center_x" not in response
            or "center_y" not in response
        ):
            logger.warning(
                "[MAP_STITCHER_RELAY] Unexpected VLM response format: %s", response
            )
            return state

        pos: Dict[str, Any] = player_data.get("position") or {}
        player_x: int = int(pos.get("x", 0))
        player_y: int = int(pos.get("y", 0))

        tile_x: int = (
            player_x + (int(response["center_x"]) - _SCREEN_CENTER_X) // _TILE_SIZE_PX
        )
        tile_y: int = (
            player_y + (int(response["center_y"]) - _SCREEN_CENTER_Y) // _TILE_SIZE_PX
        )

        logger.info(
            "[MAP_STITCHER_RELAY] VLM PokeCenter: player=(%s, %s) "
            "pixel=(%s, %s) tile=(%s, %s)",
            player_x,
            player_y,
            response["center_x"],
            response["center_y"],
            tile_x,
            tile_y,
        )

        return {
            **state,
            "goal_coords": (tile_x, tile_y),
            "context": "navigation",
            "last_action": "HEAL_ROUTE",
        }

    return map_stitcher_relay_node
