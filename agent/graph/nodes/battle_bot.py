"""
agent/graph/nodes/battle_bot — BattleBot specialist node.

Thin LangGraph wrapper over ``agent/battle_bot.py``.  No new battle logic
lives here — this module only provides the AgentState ↔ BattleBot interface
required by the dispatch graph.

Routing contract:
  - Receives: AgentState where ``state_data`` indicates an active battle.
  - Returns: updated AgentState with ``last_action = "BATTLE"`` and
    ``last_buttons`` set to the GBA button sequence for the chosen action.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.battle_bot import get_battle_bot
from agent.graph.state import AgentState

logger = logging.getLogger(__name__)

# Maps BattleBot symbolic decisions to GBA button sequences.
# Derived from agent/action.py battle-action handling.
_DECISION_TO_BUTTONS: Dict[str, List[str]] = {
    "ADVANCE_BATTLE_DIALOGUE": ["B", "B"],
    "RECOVER_FROM_RUN_FAILURE": ["B"],
    "SELECT_RUN": ["A"],
    "VLM_SELECT_RUN": ["DOWN", "RIGHT", "A"],
    "PRESS_RIGHT": ["RIGHT"],
    "SELECT_FIGHT": ["A"],
    "USE_MOVE_ABSORB": ["B", "UP", "LEFT", "A", "DOWN", "A"],
    "USE_MOVE_POUND": ["B", "UP", "LEFT", "A", "UP", "A"],
    "PRESS_B": ["B"],
    "PRESS_A_ONLY": ["A"],
    "RUN_FROM_BATTLE": ["A"],
}


def battle_bot_node(state: AgentState) -> AgentState:
    """Delegate battle decisions to BattleBot and convert to button presses.

    Args:
        state: Current AgentState, must have ``state_data`` with active battle.

    Returns:
        Updated AgentState with ``last_action = "BATTLE"`` and
        ``last_buttons`` populated.
    """
    bot = get_battle_bot()
    state_data: Dict[str, Any] = dict(state.get("state_data") or {})

    # Inject latest visual observation so BattleBot can access VLM-extracted
    # fields (mirrors the injection in agent/action.py lines 113–116).
    perception: Dict[str, Any] = state.get("perception") or {}
    latest_obs = perception.get("latest_observation") or perception
    if latest_obs:
        state_data["latest_observation"] = latest_obs

    decision: Optional[str] = bot.get_action(state_data)
    logger.debug("[BATTLEBOT] step=%s  decision=%s", state.get("step_count"), decision)

    buttons: List[str] = (
        _DECISION_TO_BUTTONS.get(decision, ["A"]) if decision else ["A"]
    )

    return {**state, "last_action": "BATTLE", "last_buttons": buttons}
