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
from typing import Any, Callable, Dict, List, Optional

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


def _format_party_hp(state_data: dict) -> str:
    """Return a compact party HP string like 'Treecko 45/50, Wingull 0/32'."""
    party = (state_data.get("player") or {}).get("party") or state_data.get("party") or []
    if not party:
        return "(no party data)"
    parts = []
    for mon in party:
        name = mon.get("species_name") or mon.get("name") or mon.get("species", "?")
        hp = mon.get("current_hp", mon.get("hp", "?"))
        max_hp = mon.get("max_hp", "?")
        parts.append(f"{name} {hp}/{max_hp}")
    return ", ".join(parts)


def make_battle_bot_node(
    episodic_memory=None,
) -> Callable[[AgentState], AgentState]:
    """Return a ``battle_bot_node`` function wired to the given dependencies.

    Args:
        episodic_memory: Optional ``EpisodicMemory`` instance.  When provided,
                         battle start and outcome events are logged to ChromaDB
                         so the HTN Supervisor can use them as completion evidence.
    """
    _prev_in_battle: list[bool] = [False]  # mutable cell avoids nonlocal on Python 3.10

    def battle_bot_node(state: AgentState) -> AgentState:
        bot = get_battle_bot()
        state_data: Dict[str, Any] = dict(state.get("state_data") or {})

        # Inject latest visual observation so BattleBot can access VLM-extracted
        # fields (mirrors the injection in agent/action.py lines 113–116).
        perception: Dict[str, Any] = state.get("perception") or {}
        latest_obs = perception.get("latest_observation") or perception
        if latest_obs:
            state_data["latest_observation"] = latest_obs

        game: dict = state_data.get("game") or {}
        in_battle: bool = bool(game.get("in_battle", False))
        location: str = (state_data.get("player") or {}).get("location", "UNKNOWN")

        # ── Phase 5.3: Battle transition logging ────────────────────────
        if not _prev_in_battle[0] and in_battle:
            logger.debug("[BATTLEBOT] Battle started at %s", location)
            if episodic_memory:
                try:
                    episodic_memory.log_event(
                        f"Battle started at {location}.",
                        metadata={
                            "type": "battle_start",
                            "location": location,
                            "map_id": game.get("map_id", 0),
                        },
                        state_data=state_data,
                    )
                except Exception as exc:
                    logger.warning("[BATTLEBOT] Failed to log battle_start: %s", exc)

        # NOTE: battle_outcome logging lives in handoff_detector.make_handoff_detector_node.
        # This node is only dispatched when in_battle=True, so the True→False transition
        # is never visible here — by then the router has already sent the step to nav_bot.

        _prev_in_battle[0] = in_battle
        # ────────────────────────────────────────────────────────────────

        decision: Optional[str] = bot.get_action(state_data)
        logger.debug("[BATTLEBOT] step=%s  decision=%s", state.get("step_count"), decision)

        buttons: List[str] = (
            _DECISION_TO_BUTTONS.get(decision, ["A"]) if decision else ["A"]
        )

        return {**state, "last_action": "BATTLE", "last_buttons": buttons}

    return battle_bot_node


# Default instance with no episodic memory — used by legacy tests and any
# callers that import battle_bot_node directly.  Production code should use
# make_battle_bot_node(episodic_memory=...) via build_graph() instead.
battle_bot_node = make_battle_bot_node(episodic_memory=None)
