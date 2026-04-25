"""
agent/graph/nodes/verification — VerificationNode factory.

Routing contract:
  - Receives: any AgentState after a specialist node has acted.
  - Returns: updated AgentState, possibly with ``milestone_index`` incremented.

This is the *Pull* half of the Hybrid Push/Pull milestone system.  After every
action the node re-reads RAM via
``ObjectiveManager.check_storyline_milestones()`` and advances the progression
counter when the current milestone is satisfied.

Usage
-----
::

    from agent.graph.nodes.verification import make_verification_node

    node = make_verification_node(obj_manager)
    graph.add_node("verification", node)
"""

from __future__ import annotations

import logging
from typing import Callable

from agent.graph.state import AgentState
from agent.objective_manager import MILESTONE_PROGRESSION, ObjectiveManager

logger = logging.getLogger(__name__)


def make_verification_node(
    obj_manager: ObjectiveManager,
) -> Callable[[AgentState], AgentState]:
    """Factory that binds an ObjectiveManager instance into a verification node.

    Args:
        obj_manager: Shared ObjectiveManager that tracks ``completed_goals``.

    Returns:
        A LangGraph-compatible node callable.
    """

    def verification_node(state: AgentState) -> AgentState:
        idx: int = state.get("milestone_index", 0)

        if idx >= len(MILESTONE_PROGRESSION):
            logger.debug(
                "[VERIFICATION] milestone_index=%s is out of range (%s milestones) — no-op.",
                idx,
                len(MILESTONE_PROGRESSION),
            )
            return state

        milestone = MILESTONE_PROGRESSION[idx]
        state_data: dict = state.get("state_data") or {}

        # Drive RAM-based auto-completion for objectives tied to emulator flags.
        # check_storyline_milestones also calls mark_goal_complete internally
        # when it detects completion, so we do not need to call it separately.
        try:
            obj_manager.check_storyline_milestones(state_data)
        except Exception as exc:
            logger.warning("[VERIFICATION] check_storyline_milestones error: %s", exc)

        # Check whether the current milestone is now recorded as complete.
        milestone_id: str = milestone["milestone"]
        if obj_manager.completed_goals.get(milestone_id, False):
            logger.info(
                "[VERIFICATION] Milestone '%s' complete — advancing index %s → %s.",
                milestone_id,
                idx,
                idx + 1,
            )
            return {**state, "milestone_index": idx + 1}

        return state

    return verification_node
