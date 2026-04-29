"""
agent/graph/nodes/handoff_detector — Lightweight, zero-LLM gatekeeper node.

Sits between every specialist node (nav_bot, battle_bot, coms_bot) and
verification_node.  Sets ``supervisor_pending = True`` when a meaningful state
transition has occurred so that the Executive Supervisor knows to re-plan.

What counts as "significant":
  • Any change in node type (battle_bot → nav_bot, coms_bot → nav_bot, etc.)
  • The very first step of a run (no previous node recorded)
  • The goal stack becoming empty (all goals complete)
  • A nav-stall: the player position has not changed for
    _NAV_STALL_THRESHOLD consecutive nav_bot steps, indicating a goal-level
    block that tile-level recovery (stuck_handler.py) has not cleared.

What is NOT significant:
  • Repeated entries to the same node (nav_bot → nav_bot while moving).
    These are the common case and must NOT wake the Supervisor.

Module-level state:
  The nav-stall tracker uses two module-level globals, following the same
  pattern as stuck_handler.py.  Tests must reset them via:
      import agent.graph.nodes.handoff_detector as hd
      hd._consecutive_nav_stall_steps = 0
      hd._last_nav_position = None
"""

from __future__ import annotations

import logging
from typing import Optional

from agent.graph.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transition table
# ---------------------------------------------------------------------------

_SIGNIFICANT_TRANSITIONS: frozenset[tuple[str, str]] = frozenset({
    ("battle_bot",         "nav_bot"),   # battle ended → resume navigation
    ("battle_bot",         "coms_bot"),  # mid-battle dialogue
    ("coms_bot",           "nav_bot"),   # dialogue finished → resume navigation
    ("nav_bot",            "coms_bot"),  # NPC triggered mid-navigation
    ("nav_bot",            "battle_bot"),# wild encounter / trainer spotted
    ("map_stitcher_relay", "nav_bot"),   # healing path resolved → navigate
})

# ---------------------------------------------------------------------------
# Nav-stall detection
# ---------------------------------------------------------------------------
# nav_bot → nav_bot re-entries are ignored by _SIGNIFICANT_TRANSITIONS, which
# creates a blind spot: if the agent is stuck in a goal-level loop the
# Supervisor never wakes.
#
# stuck_handler.py handles tile-level oscillation autonomously (dynamic tile
# blocking, TTL=200 steps) but its internal counter resets to 0 after each
# block attempt, so it never accumulates — reading it externally would always
# return a small value.
#
# Instead we track position epochs here.  15 consecutive nav_bot steps at the
# same (x, y, map_location) tuple means tile-level recovery has failed to make
# progress → escalate to the Supervisor for goal-level replanning.

_NAV_STALL_THRESHOLD: int = 15

_consecutive_nav_stall_steps: int = 0
_last_nav_position: Optional[tuple] = None

# Tracks whether the goal stack was non-empty on the *previous* step, so we can
# detect the transition from non-empty → empty ("stack just exhausted") rather
# than firing every step that the stack happens to be empty (e.g. Phase 1 before
# the Supervisor is implemented).
_prev_goal_stack_was_populated: bool = False

# ---------------------------------------------------------------------------
# Action label → node name mapping
# ---------------------------------------------------------------------------

_ACTION_TO_NODE: dict[str, str] = {
    "NAVIGATE": "nav_bot",
    "BATTLE":   "battle_bot",
    "DIALOGUE": "coms_bot",
}


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def handoff_detector_node(state: AgentState) -> AgentState:
    """Detect meaningful state transitions and set supervisor_pending.

    Reads:
        state["last_action"]      — label written by the specialist that just ran
        state["last_node_fired"]  — label written by *this* node on the previous step
        state["goal_stack"]       — HTN goal stack (list of dicts)
        state["state_data"]       — full game state (for position extraction)

    Writes:
        state["last_node_fired"]    — updated to current_node_name
        state["supervisor_pending"] — True when a significant event is detected
    """
    global _consecutive_nav_stall_steps, _last_nav_position, _prev_goal_stack_was_populated

    current_action: str  = state.get("last_action", "") or ""
    previous_node:  str  = state.get("last_node_fired", "") or ""
    goal_stack:     list = state.get("goal_stack", []) or []

    current_node_name  = _ACTION_TO_NODE.get(current_action, current_action)
    previous_node_name = _ACTION_TO_NODE.get(previous_node, previous_node)

    transition = (previous_node_name, current_node_name)

    # Fire when the stack *transitions* from non-empty → empty (i.e. all goals
    # were just completed).  Do NOT fire every step the stack is empty — that
    # would keep pending=True throughout all of Phase 1 before the Supervisor
    # has ever populated the stack.
    stack_just_exhausted = (not goal_stack) and _prev_goal_stack_was_populated
    _prev_goal_stack_was_populated = bool(goal_stack)

    is_significant: bool = (
        transition in _SIGNIFICANT_TRANSITIONS
        or not previous_node_name   # first step of this run
        or stack_just_exhausted     # stack just drained — need new plan
    )

    # ----------------------------------------------------------------
    # Nav-stall check
    # ----------------------------------------------------------------
    if current_node_name == "nav_bot":
        player = (state.get("state_data") or {}).get("player", {})
        pos    = player.get("position") or {}
        nav_pos: tuple = (pos.get("x"), pos.get("y"), player.get("location"))

        if nav_pos == _last_nav_position:
            _consecutive_nav_stall_steps += 1
        else:
            _consecutive_nav_stall_steps = 0

        _last_nav_position = nav_pos

        if _consecutive_nav_stall_steps >= _NAV_STALL_THRESHOLD:
            logger.warning(
                "[HANDOFF] Nav stall detected: %d consecutive steps at %s "
                "— waking Supervisor for goal-level replanning",
                _consecutive_nav_stall_steps,
                nav_pos,
            )
            is_significant = True
            _consecutive_nav_stall_steps = 0  # reset; prevent re-firing every step
    else:
        # Leaving nav_bot — reset stall tracker
        _consecutive_nav_stall_steps = 0
        _last_nav_position = None

    if is_significant:
        logger.debug(
            "[HANDOFF] Significant: %s → %s  (pending=True  stall_steps=%d)",
            previous_node_name or "(none)",
            current_node_name  or "(none)",
            _consecutive_nav_stall_steps,
        )
    
    return {
        **state,
        "last_node_fired":   current_node_name,
        "supervisor_pending": is_significant,
    }


# ---------------------------------------------------------------------------
# Factory (Phase 5.3 — battle outcome logging)
# ---------------------------------------------------------------------------

def make_handoff_detector_node(episodic_memory=None):
    """Return a ``handoff_detector_node`` optionally wired to episodic memory.

    When ``episodic_memory`` is provided, the returned node logs a
    ``battle_outcome`` event to ChromaDB whenever it detects a
    ``battle_bot → <other>`` transition.  This is the correct place for that
    log because ``battle_bot_node`` is only dispatched while ``in_battle=True``
    — by the time ``in_battle`` flips to ``False`` the router has already sent
    the step to ``nav_bot``, so the ``True → False`` branch in ``battle_bot_node``
    can never fire.

    Args:
        episodic_memory: Optional ``EpisodicMemory`` instance.

    Returns:
        A node function compatible with LangGraph's ``add_node``.
    """
    if episodic_memory is None:
        return handoff_detector_node  # no overhead for the common/test case

    from agent.graph.nodes.battle_bot import _format_party_hp

    def _node_with_memory(state: AgentState) -> AgentState:
        # Detect battle→non-battle before delegating so we use the unmodified state.
        current_action: str = state.get("last_action", "") or ""
        previous_node:  str = state.get("last_node_fired", "") or ""
        current_node_name  = _ACTION_TO_NODE.get(current_action, current_action)
        previous_node_name = _ACTION_TO_NODE.get(previous_node, previous_node)

        if previous_node_name == "battle_bot" and current_node_name != "battle_bot":
            state_data = state.get("state_data") or {}
            game       = state_data.get("game") or {}
            location   = (state_data.get("player") or {}).get("location", "UNKNOWN")
            party_hp   = _format_party_hp(state_data)
            try:
                episodic_memory.log_event(
                    f"Battle ended at {location}. Party HP: {party_hp}",
                    metadata={
                        "type":     "battle_outcome",
                        "location": location,
                        "map_id":   game.get("map_id", 0),
                        "party_hp": party_hp,
                    },
                )
                logger.debug("[HANDOFF] Logged battle_outcome at %s  party=%s", location, party_hp)
            except Exception as exc:
                logger.warning("[HANDOFF] Failed to log battle_outcome: %s", exc)

        return handoff_detector_node(state)

    return _node_with_memory
