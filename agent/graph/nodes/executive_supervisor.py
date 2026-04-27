"""
agent/graph/nodes/executive_supervisor — HTN Executive Supervisor node.

The Supervisor is the cognitive controller that maintains and updates the
HTN goal stack.  It fires whenever ``handoff_detector`` sets
``supervisor_pending=True`` (i.e. on a significant node-type transition,
first step, or goal-stack exhaustion).

Phase 2 status:
  • ``_bootstrap_stack`` is a stub that returns ``[]`` so the supervisor
    is a transparent no-op until Phase 4 populates it.
  • ``_call_supervisor_llm`` is a stub that always returns CONTINUE until
    Phase 3 adds the real prompt template and JSON schema.
  • ``_expand_strategic_goal`` is a stub returning ``[]``.
  • ``_apply_immediate_directive`` is fully implemented and used when the
    factory is called with ``use_htn=True``.

Stack operations (applied when stack is non-empty):
  POP      — remove Stack[0]; if parent becomes strategic with no children,
             auto-expand with _expand_strategic_goal.
  PUSH     — prepend new sub-goals (capped at _STACK_DEPTH_CAP).
  REPLACE  — swap Stack[0] with a new goal.
  CONTINUE — no stack change (supervisor just refreshes reasoning).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from agent.graph.goal_stack import (
    GoalNode,
    stack_peek,
    stack_pop,
    stack_push,
    stack_replace,
    stack_summary,
)
from agent.graph.state import AgentState

logger = logging.getLogger(__name__)

# Maximum stack depth.  If a PUSH would exceed this, demote to CONTINUE
# and log a WARNING so the runaway-stack case is visible in llm_logs.
_STACK_DEPTH_CAP = 8


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def make_executive_supervisor_node(
    vlm,
    episodic_memory,
    walkthrough_db=None,
    use_htn: bool = False,
) -> Callable[[AgentState], AgentState]:
    """Return the ``executive_supervisor_node`` function wired to the given dependencies.

    Args:
        vlm:              ``VLM`` instance used to call the supervisor LLM.
                          May be ``None`` in tests / offline mode.
        episodic_memory:  ``EpisodicMemory`` instance for context retrieval.
                          May be ``None``.
        walkthrough_db:   ``WalkthroughDB`` instance for bootstrap RAG.
                          May be ``None`` (Phase 2 stub ignores it anyway).
        use_htn:          When ``True`` the node copies the top-of-stack
                          directive into AgentState navigation fields so the
                          FSM specialist nodes pick it up.  Default ``False``
                          keeps the legacy FSM driving navigation.
    """

    def executive_supervisor_node(state: AgentState) -> AgentState:
        step = state.get("step_count", 0)
        stack_raw = state.get("goal_stack", [])
        stack: list[GoalNode] = [GoalNode.from_dict(g) for g in stack_raw]
        state_data: dict = state.get("state_data") or {}

        # ----------------------------------------------------------------
        # 1. Bootstrap: if the stack is empty, try to populate it.
        # ----------------------------------------------------------------
        if not stack:
            stack = _bootstrap_stack(state_data, walkthrough_db, vlm)
            if not stack:
                # Stub returns [] — behave as no-op.
                # This print confirms the node is wired and reached.
                # It disappears in Phase 4 once _bootstrap_stack is real.
                print(f"[SUPERVISOR] step={step}  Bootstrap stub — stack empty, no-op (Phase 4 needed for real HTN).")
                logger.debug(
                    "[SUPERVISOR] step=%s  Bootstrap returned empty stack — no-op.", step
                )
                return {**state, "supervisor_pending": False}

            logger.info(
                "[SUPERVISOR] step=%s  BOOTSTRAP  stack=%s", step, stack_summary(stack)
            )
            print(f"[SUPERVISOR] step={step}  BOOTSTRAP")
            print(f"[SUPERVISOR] Stack: {stack_summary(stack)}")
            return {
                **state,
                "goal_stack": [g.to_dict() for g in stack],
                "supervisor_pending": False,
                "supervisor_last_operation": "BOOTSTRAP",
                "supervisor_last_reasoning": "Initial goal stack populated.",
            }

        # ----------------------------------------------------------------
        # 2. Gather context for the LLM.
        # ----------------------------------------------------------------
        current_goal: Optional[GoalNode] = stack_peek(stack)
        episodic_ctx: str = _query_episodic_memory(episodic_memory, current_goal, state_data)
        game_summary: str = _build_game_summary(state_data, state)
        stack_repr: str = stack_summary(stack)

        # ----------------------------------------------------------------
        # 3. Ask the LLM what to do with the stack.
        # ----------------------------------------------------------------
        operation_payload: dict = _call_supervisor_llm(
            vlm, current_goal, episodic_ctx, game_summary, stack_repr
        )

        op: str = (operation_payload.get("operation") or "CONTINUE").upper()
        reason: str = operation_payload.get("reasoning") or ""
        new_goals: list = operation_payload.get("new_goals") or []

        # ----------------------------------------------------------------
        # 4. Apply the stack operation.
        # ----------------------------------------------------------------
        if op == "POP":
            popped, stack = stack_pop(stack)
            logger.info(
                "[SUPERVISOR] step=%s  POP '%s' — %s",
                step,
                popped.goal_id if popped else "?",
                reason,
            )
            # If the new top-of-stack is a strategic goal with no children,
            # auto-expand it into tactical/immediate sub-goals.
            parent = stack_peek(stack)
            if parent and parent.goal_type == "strategic" and not _has_children(stack, parent):
                new_sub_goals = _expand_strategic_goal(parent, state_data, walkthrough_db, vlm)
                for g in reversed(new_sub_goals):
                    stack = stack_push(stack, g)

        elif op == "PUSH":
            if len(stack) >= _STACK_DEPTH_CAP:
                logger.warning(
                    "[SUPERVISOR] step=%s  PUSH rejected: depth %d >= cap %d "
                    "(reasoning: %s) — demoting to CONTINUE to prevent runaway "
                    "stack growth. Review llm_logs for repeated PUSH operations "
                    "on the same goal.",
                    step,
                    len(stack),
                    _STACK_DEPTH_CAP,
                    reason,
                )
                op = "CONTINUE"
            else:
                for g_dict in new_goals:
                    node = GoalNode.from_dict({**g_dict, "push_reason": reason})
                    stack = stack_push(stack, node)
                logger.info(
                    "[SUPERVISOR] step=%s  PUSH %d goal(s) — %s",
                    step,
                    len(new_goals),
                    reason,
                )

        elif op == "REPLACE":
            if new_goals:
                node = GoalNode.from_dict({**new_goals[0], "push_reason": reason})
                stack = stack_replace(stack, node)
            logger.info("[SUPERVISOR] step=%s  REPLACE Stack[0] — %s", step, reason)

        else:  # CONTINUE (or any unrecognised value)
            op = "CONTINUE"
            logger.debug("[SUPERVISOR] step=%s  CONTINUE — %s", step, reason)

        # ----------------------------------------------------------------
        # 5. Optionally translate the top directive into AgentState fields.
        # ----------------------------------------------------------------
        if use_htn:
            new_state: dict = _apply_immediate_directive(state, stack)
        else:
            new_state = dict(state)

        new_state.update(
            {
                "goal_stack": [g.to_dict() for g in stack],
                "supervisor_pending": False,
                "supervisor_last_operation": op,
                "supervisor_last_reasoning": reason[:500],
            }
        )
        return new_state

    return executive_supervisor_node


# ---------------------------------------------------------------------------
# Stack operation helpers
# ---------------------------------------------------------------------------

def _has_children(stack: list[GoalNode], parent: GoalNode) -> bool:
    """Return True if any entry in *stack* has ``parent_id`` matching *parent.goal_id*."""
    return any(g.parent_id == parent.goal_id for g in stack)


def _apply_immediate_directive(state: AgentState, stack: list[GoalNode]) -> dict:
    """Copy the top-of-stack directive fields into AgentState navigation fields.

    Only called when ``use_htn=True``.  Reads directly from the directive
    dict (JSON-serialisable; no Directive dataclass import needed here) so
    that supervisor-generated goal directives are forwarded to the FSM
    specialist nodes.
    """
    immediate = stack_peek(stack)
    if not immediate or not immediate.directive:
        return dict(state)

    d: dict = immediate.directive
    patch: dict = {}

    if d.get("goal_coords"):
        patch["goal_coords"] = d["goal_coords"]
    if d.get("goal_location"):
        patch["goal_location"] = d["goal_location"]
    if d.get("npc_coords"):
        patch["npc_coords"] = d["npc_coords"]
    if d.get("should_interact") is not None:
        patch["should_interact"] = d["should_interact"]
    if d.get("description"):
        patch["goal_description"] = d["description"]
    if immediate.goal_id:
        patch["active_milestone"] = immediate.goal_id

    return {**state, **patch}


# ---------------------------------------------------------------------------
# Stubs (Phase 2) — replaced in later phases
# ---------------------------------------------------------------------------

def _bootstrap_stack(state_data: dict, walkthrough_db, vlm) -> list[GoalNode]:
    """Phase 2–3 stub.  Returns [] so the supervisor is a transparent no-op.

    Phase 4 replaces this with a real RAG + LLM implementation that seeds
    the stack with strategic → tactical → immediate goals derived from the
    walkthrough knowledge base and current game state.
    """
    return []


def _expand_strategic_goal(
    parent: GoalNode,
    state_data: dict,
    walkthrough_db,
    vlm,
) -> list[GoalNode]:
    """Phase 4+ stub.  Returns [] until RAG-bootstrap is implemented."""
    return []


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

def _query_episodic_memory(
    episodic_memory,
    current_goal: Optional[GoalNode],
    state_data: dict,
) -> str:
    """Retrieve relevant episodic memories for the current goal.

    Returns a formatted string suitable for inclusion in the supervisor
    prompt.  Returns a safe placeholder string on any failure.
    """
    if episodic_memory is None:
        return "(No episodic memory available.)"
    if current_goal is None:
        return "(No current goal.)"
    try:
        query = f"{current_goal.description} {current_goal.completion_condition or ''}".strip()
        return episodic_memory.retrieve_relevant(query, n_results=3, max_distance=1.2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SUPERVISOR] Episodic memory query failed: %s", exc)
        return "(Episodic memory query failed.)"


def _build_game_summary(state_data: dict, state: dict) -> str:
    """Build a compact one-line game-state summary for the supervisor prompt."""
    player: dict = state_data.get("player") or {}
    game: dict = state_data.get("game") or {}
    pos: dict = player.get("position") or {}
    party: list = state_data.get("party") or []

    location: str = player.get("location") or "Unknown"
    x = pos.get("x", "?")
    y = pos.get("y", "?")
    in_battle: bool = bool(game.get("in_battle", False))

    # Support both int (simple badge count) and dict (per-badge flags)
    badges_raw = game.get("badges", 0)
    if isinstance(badges_raw, int):
        badges: int = badges_raw
    elif isinstance(badges_raw, dict):
        badges = sum(1 for v in badges_raw.values() if v)
    else:
        badges = 0

    hp_parts: list[str] = []
    for p in party[:3]:
        name = p.get("name", "?")
        hp = p.get("current_hp", "?")
        max_hp = p.get("max_hp", "?")
        hp_parts.append(f"{name} {hp}/{max_hp}")
    hp_summary = ", ".join(hp_parts) if hp_parts else "No party"

    return (
        f"Location: {location}  Pos: ({x},{y})  "
        f"Badges: {badges}  InBattle: {in_battle}  "
        f"Party: {hp_summary}"
    )


def _call_supervisor_llm(
    vlm,
    current_goal: Optional[GoalNode],
    episodic_ctx: str,
    game_summary: str,
    stack_repr: str,
) -> dict:
    """Call the VLM supervisor and parse its stack-operation decision.

    Phase 2 stub: always returns CONTINUE so the node is a transparent
    no-op.  Phase 3 replaces this body with the full prompt template,
    ``vlm.get_text_query()`` call, and JSON schema parsing.

    Returns a dict with at minimum:
        ``{"operation": str, "reasoning": str, "new_goals": list}``
    """
    return {
        "operation": "CONTINUE",
        "reasoning": "Phase 2 stub — LLM prompt not yet implemented.",
        "new_goals": [],
    }
