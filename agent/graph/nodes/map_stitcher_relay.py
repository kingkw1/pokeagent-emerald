"""
agent/graph/nodes/map_stitcher_relay — MapStitcherRelay factory.

Routing contract:
  - Receives: AgentState where the agent needs to locate the nearest
    PokeCenter (typically triggered when party HP is critically low).
  - Returns: updated AgentState with:
      ``goal_coords``  — tile coordinates of the nearest PokeCenter entrance
      ``context``      — ``"navigation"``
      ``last_action``  — ``"HEAL_ROUTE"``

The node asks the VLM (e.g. Gemini Flash) to identify the PokeCenter pixel
location on the best available map image, then converts that pixel answer to
tile coordinates using the formula::

    tile_x = player_x + (pixel_x - SCREEN_CENTER_X) // TILE_SIZE_PX
    tile_y = player_y + (pixel_y - SCREEN_CENTER_Y) // TILE_SIZE_PX

GBA screen constants: 240 × 160 px, 16 px per tile, player at (120, 80).

Map image sourcing (in priority order):
  1. ``stitcher.get_overhead_image()`` — PIL Image of the stitched overhead map
     (requires MapStitcher to have that method; it is the intended target).
  2. ``state["frame"]`` — the raw GBA frame as a fallback when no stitched
     overhead is available yet.
  3. If neither is available the state is returned unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from agent.graph.state import AgentState
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

    Returns:
        A LangGraph-compatible node callable.
    """

    def map_stitcher_relay_node(state: AgentState) -> AgentState:
        # --- Obtain a map image -----------------------------------------------
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

        # --- Query the VLM for PokeCenter location ----------------------------
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

        # --- Convert pixel offset to tile coordinates -------------------------
        player_data: Dict[str, Any] = (state.get("state_data") or {}).get("player", {})
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
            "[MAP_STITCHER_RELAY] PokeCenter goal: player=(%s, %s) "
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
