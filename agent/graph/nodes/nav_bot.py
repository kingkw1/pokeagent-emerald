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
    goal_desc = state.get("goal_description") or ""
    milestone_label = state.get("active_milestone") or ""
    desc_suffix = f"  ← {goal_desc}" if goal_desc else ""
    ms_suffix = f"  [{milestone_label}]" if milestone_label else ""
    _nav_location = state_data.get('player', {}).get('location', '?')
    print(f"[NAVBOT] step={state.get('step_count')}  pos=({pos_x}, {pos_y})  goal=({goal_x}, {goal_y})  loc='{_nav_location}'{ms_suffix}{desc_suffix}")

    # pathfind_to_goal calls update_npc_obstacles internally.
    # Pass npc_coords so the target NPC's tile stays walkable.
    buttons: list = pathfind_to_goal(
        state_data,
        goal_x,
        goal_y,
        npc_coords=npc_coords,
    ) or []

    # Fallback: when all A* tiers fail, push one step toward the goal so the
    # agent keeps moving and expands the explored map.  This prevents the
    # agent from returning an empty action list (which causes the client to
    # skip the frame, leaving the agent permanently stuck when the map hasn't
    # been explored in the goal direction yet).
    if not buttons:
        dx = goal_x - (pos_x if isinstance(pos_x, int) else int(pos_x))
        dy = goal_y - (pos_y if isinstance(pos_y, int) else int(pos_y))
        if abs(dy) >= abs(dx):
            fallback_dir = "DOWN" if dy > 0 else "UP"
        else:
            fallback_dir = "RIGHT" if dx > 0 else "LEFT"
        print(f"[NAVBOT] ⚠️ pathfinding failed — directional fallback: {fallback_dir}")
        buttons = [fallback_dir]

    # When adjacent to the target NPC, face it then press A.
    # Prepending the face-direction is safe: pressing a direction button when
    # already facing that way does nothing except confirm the facing, then A
    # interacts correctly.
    if should_interact and npc_coords:
        px = player_pos.get("x")
        py = player_pos.get("y")
        nx, ny = int(npc_coords[0]), int(npc_coords[1])
        if px is not None and py is not None:
            dist = abs(int(px) - nx) + abs(int(py) - ny)
            if dist <= 1:
                dx = nx - int(px)
                dy = ny - int(py)
                if dy < 0:
                    face = "UP"
                elif dy > 0:
                    face = "DOWN"
                elif dx > 0:
                    face = "RIGHT"
                else:
                    face = "LEFT"
                logger.debug("[NAVBOT] Adjacent to NPC at (%s, %s) — facing %s then A.", nx, ny, face)
                # REPLACE movement buttons: already adjacent, just face and A.
                # Appending would double-move (e.g. RIGHT + RIGHT + A) when
                # pathfinding already queued a step toward the NPC's tile.
                buttons = [face, "A"]

    return {**state, "last_action": "NAVIGATE", "last_buttons": buttons}
