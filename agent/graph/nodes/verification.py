"""
agent/graph/nodes/verification — VerificationNode factory.

Routing contract:
  - Receives: any AgentState after a specialist node has acted.
  - Returns: updated AgentState, possibly with ``milestone_index`` incremented.

This is the *Pull* half of the Hybrid Push/Pull milestone system.  After every
action the node re-reads RAM via
``ObjectiveManager.check_storyline_milestones()`` and advances the progression
counter when the current milestone is satisfied.

Phase 5.1 — completion_type gating
-----------------------------------
Each milestone in ``MILESTONE_PROGRESSION`` now carries a ``completion_type``
field:

* ``"location"`` / ``"battle"`` — existing behaviour: ROM flag in
  ``state_data["milestones"]`` drives completion (via
  ``check_storyline_milestones``).
* ``"dialogue"`` — ROM flag is treated as a scene *trigger* only
  (``check_storyline_milestones`` skips auto-completion for these).
  Advancement only happens when ``state["dialogue_completed"] == True``,
  which is set by ``Agent.step()`` on the dialogue→navigation transition
  after ``TransitionEvaluator`` confirms the expected keywords were spoken.

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
        milestone_id: str = milestone["milestone"]
        completion_type: str = milestone.get("completion_type", "location")
        state_data: dict = state.get("state_data") or {}

        # ------------------------------------------------------------------
        # Phase 5.1: dialogue-type milestones
        # ------------------------------------------------------------------
        if completion_type == "dialogue":
            if state.get("dialogue_completed"):
                logger.info(
                    "[VERIFICATION] Dialogue milestone '%s' confirmed complete "
                    "(dialogue_completed=True) — advancing index %s → %s.",
                    milestone_id,
                    idx,
                    idx + 1,
                )
                obj_manager.mark_goal_complete(
                    milestone_id,
                    description=milestone.get("description", ""),
                    state_data=state_data,
                )
                # Reset dialogue_completed so the flag is not re-consumed next step.
                return {**state, "milestone_index": idx + 1, "dialogue_completed": False}

            logger.debug(
                "[VERIFICATION] Dialogue milestone '%s' waiting for dialogue_completed flag.",
                milestone_id,
            )
            return state

        # ------------------------------------------------------------------
        # location / battle milestones — existing ROM-flag behaviour
        # ------------------------------------------------------------------
        try:
            obj_manager.check_storyline_milestones(state_data)
        except Exception as exc:
            logger.warning("[VERIFICATION] check_storyline_milestones error: %s", exc)

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

