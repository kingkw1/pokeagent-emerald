"""
agent/graph/router — routing_condition for the LangGraph dispatch graph.

The router reads AgentState and returns the name of the next node to execute.
This is used as the conditional edge function on the dispatch node.

Routing priority (highest → lowest):
  1. healing_needed  → map_stitcher_relay
  2. is_in_battle    → battle_bot
  3. is_in_dialog    → coms_bot
  4. default         → nav_bot
"""

from __future__ import annotations

from agent.graph.state import AgentState


def routing_condition(state: AgentState) -> str:
    """Return the name of the next node to route to.

    Called by LangGraph as a conditional edge function on the dispatch node.

    Args:
        state: Current AgentState.

    Returns:
        One of: ``"map_stitcher_relay"``, ``"battle_bot"``,
        ``"coms_bot"``, ``"nav_bot"``.
    """
    context = state.get("context", "navigation")
    state_data = state.get("state_data") or {}
    game = state_data.get("game", {})

    # 1. Healing override takes highest priority
    if context == "healing_needed":
        return "map_stitcher_relay"

    # 2. Battle state
    is_in_battle = game.get("is_in_battle", False) or (
        game.get("game_state", "") == "battle"
    )
    if is_in_battle:
        return "battle_bot"

    # 3. Dialogue state
    is_in_dialog = game.get("is_in_dialog", False) or (
        game.get("game_state", "") in ("dialog", "dialogue")
    )
    if is_in_dialog:
        return "coms_bot"

    # 4. Default: navigation
    return "nav_bot"
