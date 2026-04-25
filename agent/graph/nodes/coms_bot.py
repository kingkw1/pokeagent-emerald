"""
agent/graph/nodes/coms_bot — ComsBot (dialogue / opening-sequence) specialist node.

Thin LangGraph wrapper over ``agent/opener_bot.py``.  No new FSM logic lives
here — this module only provides the AgentState ↔ OpenerBot interface required
by the dispatch graph.

Routing contract:
  - Receives: AgentState where ``state_data`` indicates active dialogue or an
    opening-sequence trigger for OpenerBot.
  - Returns: updated AgentState with ``last_action = "DIALOGUE"`` and
    ``last_buttons`` set to the button sequence that advances the conversation.

ComsBot uses a two-tier delegation pattern:

1. **OpenerBot** handles scripted FSM sequences (title screen, naming, rival
   encounter, obtaining starter, etc.).  When ``OpenerBot.should_handle()``
   returns True the returned value may be:
     - ``list[str]``         → use directly as buttons
     - ``NavigationGoal``    → set ``goal_coords`` for NavBot, return ``[]``
     - ``ForceDialogueGoal`` → press A to dismiss misclassified dialogue
     - ``None``              → fall through to normal A press

2. **Normal NPC dialogue** → wait for script-idle then press A.

Script-idle guard
-----------------
``wait_for_script_idle()`` is called before every A press for standard
overworld NPC dialogue.  It is skipped when:
  - Location is in ``_SKIP_SCRIPT_IDLE_LOCATIONS`` (intro uses GBA callbacks,
    not ``sGlobalScriptContext``, so the endpoint is meaningless there).
  - No game-server connection is available (exception caught silently).

RAM fallback
------------
``state_data["game"]["game_state"] == "dialog"`` is read directly from GBA RAM
and remains reliable even when VLM perception has timed out.  The router
already uses this field to route into ComsBot, so no additional check is
needed inside the node itself.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.graph.state import AgentState
from agent.opener_bot import (
    ForceDialogueGoal,
    NavigationGoal,
    get_opener_bot,
    wait_for_script_idle,
)

logger = logging.getLogger(__name__)

# Locations where wait_for_script_idle must be skipped — the GBA intro uses
# native C callbacks instead of sGlobalScriptContext, so the endpoint is
# not meaningful there.
_SKIP_SCRIPT_IDLE_LOCATIONS = frozenset({"TITLE_SEQUENCE", "MOVING_VAN"})


def coms_bot_node(state: AgentState) -> AgentState:
    """Advance dialogue or delegate to OpenerBot for scripted sequences.

    Args:
        state: Current AgentState with active dialogue or opener trigger.

    Returns:
        Updated AgentState with ``last_action = "DIALOGUE"`` and
        ``last_buttons`` populated.
    """
    opener = get_opener_bot()
    state_data: Dict[str, Any] = state.get("state_data") or {}
    perception: Dict[str, Any] = state.get("perception") or {}
    # OpenerBot expects the VLM-extracted visual_data sub-dict, not the full
    # perception output.
    visual_data: Dict[str, Any] = perception.get("visual_data") or {}

    buttons: List[str] = ["A"]
    new_goal_coords: Optional[tuple] = state.get("goal_coords")
    new_npc_coords: Optional[tuple] = state.get("npc_coords")
    new_should_interact: bool = state.get("should_interact", False)

    if opener.should_handle(state_data, visual_data):
        result = opener.get_action(state_data, visual_data)
        logger.debug("[COMSBOT] OpenerBot result type: %s", type(result).__name__)

        if isinstance(result, ForceDialogueGoal):
            logger.debug("[COMSBOT] ForceDialogueGoal (%s) — pressing A.", result.reason)
            buttons = ["A"]
        elif isinstance(result, NavigationGoal):
            logger.debug(
                "[COMSBOT] NavigationGoal → (%s, %s) @ %s",
                result.x,
                result.y,
                result.map_location,
            )
            new_goal_coords = (result.x, result.y)
            if result.should_interact is not None:
                new_should_interact = result.should_interact
            buttons = []
        elif isinstance(result, list):
            buttons = result
        else:
            # None or unknown type → fall through to A press
            buttons = ["A"]
    else:
        # Normal dialogue — wait for script idle then press A.
        location: str = state_data.get("player", {}).get("location", "")
        game: dict = state_data.get("game", {})
        # Skip the script-idle wait when RAM confirms we are not in dialogue
        # (covers save-state residual script-context values like mode=155 that
        # would otherwise cause a 2-second timeout on every overworld step).
        ram_in_dialog: bool = game.get("in_dialog", False) or (
            game.get("game_state", "") in ("dialog", "dialogue")
        )
        skip_idle = location in _SKIP_SCRIPT_IDLE_LOCATIONS or not ram_in_dialog
        if not skip_idle:
            try:
                wait_for_script_idle()
            except Exception as exc:
                logger.debug("[COMSBOT] wait_for_script_idle skipped: %s", exc)
        buttons = ["A"]

    logger.debug("[COMSBOT] step=%s  buttons=%s", state.get("step_count"), buttons)

    return {
        **state,
        "goal_coords": new_goal_coords,
        "npc_coords": new_npc_coords,
        "should_interact": new_should_interact,
        "last_action": "DIALOGUE",
        "last_buttons": buttons,
    }
