"""
agent/graph/nodes/nav_bot — NavBot specialist node.

Thin LangGraph wrapper over ``agent/pathfinding.py``.  No new pathfinding
logic lives here — this module only provides the AgentState ↔ pathfinding
interface required by the dispatch graph.

Routing contract:
  - Receives: AgentState with ``goal_coords``, ``state_data``,
    optionally ``npc_coords`` and ``should_interact``.
  - Returns: updated AgentState with ``last_action`` and ``last_buttons``
    populated.
  - ``last_action = "PASS"``  when ``goal_coords`` is None or empty.
  - ``last_action = "NAVIGATE"`` otherwise.
"""

from __future__ import annotations

import logging
from typing import Optional

from agent.graph.state import AgentState
from agent.pathfinding import pathfind_to_goal

logger = logging.getLogger(__name__)

_DIRECTION_TO_BUTTON = {
    "north": "UP", "south": "DOWN", "east": "RIGHT", "west": "LEFT",
    "up": "UP",   "down": "DOWN",  "right": "RIGHT", "left": "LEFT",
}


def nav_bot_node(state: AgentState) -> AgentState:
    """Navigate toward ``goal_coords`` using multi-tier pathfinding.

    Calls ``pathfind_to_goal()`` (which internally refreshes NPC obstacles via
    ``update_npc_obstacles()``) and optionally appends an A-press when the
    player is adjacent to ``npc_coords`` and ``should_interact`` is True.

    Args:
        state: Current AgentState.

    Returns:
        Updated AgentState with ``last_action`` and ``last_buttons`` set.
    """
    goal_coords: Optional[tuple] = state.get("goal_coords")
    goal_location: Optional[str] = state.get("goal_location") or ""

    # Handle CROSS_BOUNDARY directives: no goal_coords, but a direction to press.
    if not goal_coords and goal_location.startswith("CROSS_BOUNDARY:"):
        direction = goal_location.split(":", 1)[1].strip().lower()
        button = _DIRECTION_TO_BUTTON.get(direction, "LEFT")
        print(f"[NAVBOT] step={state.get('step_count')}  CROSS_BOUNDARY → {button}")
        return {**state, "last_action": "NAVIGATE", "last_buttons": [button]}

    if not goal_coords:
        logger.debug("[NAVBOT] No goal_coords — passing.")
        return {**state, "last_action": "PASS", "last_buttons": []}

    goal_x, goal_y = int(goal_coords[0]), int(goal_coords[1])
    state_data: dict = state.get("state_data") or {}
    npc_coords: Optional[tuple] = state.get("npc_coords")
    should_interact: bool = state.get("should_interact", False)

    player_pos = state_data.get("player", {}).get("position", {})
    pos_x = player_pos.get("x", "?")
    pos_y = player_pos.get("y", "?")
    print(f"[NAVBOT] step={state.get('step_count')}  pos=({pos_x}, {pos_y})  goal=({goal_x}, {goal_y})")

    # pathfind_to_goal calls update_npc_obstacles internally.
    # Pass npc_coords so the target NPC's tile stays walkable.
    buttons: list = pathfind_to_goal(
        state_data,
        goal_x,
        goal_y,
        npc_coords=npc_coords,
    ) or []

    # Append A-press when the player is adjacent to the target NPC.
    if should_interact and npc_coords:
        px = player_pos.get("x")
        py = player_pos.get("y")
        nx, ny = int(npc_coords[0]), int(npc_coords[1])
        if px is not None and py is not None:
            dist = abs(int(px) - nx) + abs(int(py) - ny)
            if dist <= 1:
                logger.debug("[NAVBOT] Adjacent to NPC at (%s, %s) — appending A.", nx, ny)
                buttons = list(buttons) + ["A"]

    return {**state, "last_action": "NAVIGATE", "last_buttons": buttons}
