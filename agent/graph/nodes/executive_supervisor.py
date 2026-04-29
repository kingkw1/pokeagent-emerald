"""
agent/graph/nodes/executive_supervisor — HTN Executive Supervisor node.

The Supervisor is the cognitive controller that maintains and updates the
HTN goal stack.  It fires whenever ``handoff_detector`` sets
``supervisor_pending=True`` (i.e. on a significant node-type transition,
first step, or goal-stack exhaustion).

Phase 4 status:
  • ``_bootstrap_stack`` is fully implemented: reads milestones from
    ``state_data``, queries the walkthrough RAG, calls the LLM with
    ``_HTN_SYSTEM_PROMPT``, validates the returned HTN, falls back to
    ``_milestone_fallback_stack`` on any error or empty DB.
  • ``_expand_strategic_goal`` is fully implemented: queries RAG with the
    parent goal description and asks the LLM for 2-3 tactical sub-goals.
  • ``_call_supervisor_llm`` uses the real prompts (Phase 3).

Stack operations (applied when stack is non-empty):
  POP      — remove Stack[0]; if parent becomes strategic with no children,
             auto-expand with _expand_strategic_goal.
  PUSH     — prepend new sub-goals (capped at _STACK_DEPTH_CAP).
  REPLACE  — swap Stack[0] with a new goal.
  CONTINUE — no stack change (supervisor just refreshes reasoning).
"""

from __future__ import annotations

import json
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
from agent.objective_manager import MILESTONE_PROGRESSION

logger = logging.getLogger(__name__)

# Maximum stack depth.  If a PUSH would exceed this, demote to CONTINUE
# and log a WARNING so the runaway-stack case is visible in llm_logs.
_STACK_DEPTH_CAP = 8

# ---------------------------------------------------------------------------
# HTN Generation system prompt (Phase 4)
# ---------------------------------------------------------------------------

_HTN_SYSTEM_PROMPT = """\
You are generating the initial goal hierarchy for a Pokémon Emerald AI agent.

Given walkthrough context and current game state, generate a NESTED TASK NETWORK
with exactly this structure:
  - 1 strategic goal  (high-level quest objective, type="strategic")
  - 2-4 tactical goals (mid-level steps to complete the strategic goal, type="tactical")
  - 1 immediate goal  (first concrete action, type="immediate", MUST include directive)

The goals must be ordered from most-immediate (first in array) to most-strategic (last).
Stack[0] (first in array) is what the agent does RIGHT NOW.

OUTPUT FORMAT:
{
  "goals": [
    {
      "goal_id": "traverse_route_104_south",
      "description": "Walk north through Route 104 South",
      "goal_type": "immediate",
      "parent_id": "reach_rustboro_city",
      "completion_condition": "Player location changes to PETALBURG_WOODS or ROUTE_104_NORTH",
      "directive": {
        "action": "NAVIGATE",
        "goal_coords": null,
        "goal_location": "PETALBURG_WOODS",
        "should_interact": false,
        "npc_coords": null,
        "description": "Head north through Route 104 South toward Petalburg Woods"
      },
      "metadata": {}
    },
    {
      "goal_id": "earn_stone_badge",
      "description": "Defeat Gym Leader Roxanne to earn the Stone Badge",
      "goal_type": "strategic",
      "parent_id": null,
      "completion_condition": "Player has 1 badge",
      "directive": null,
      "metadata": {"required_badge_count": 1}
    }
  ]
}

RULES:
1. The immediate goal MUST have a directive block.
2. Use only LOCATION_GRAPH keys for goal_location (e.g. ROUTE_104_SOUTH,
   PETALBURG_WOODS, RUSTBORO_CITY, RUSTBORO_CITY_GYM). Not prose names.
3. Set goal_coords to null if unsure — nav_bot resolves paths automatically.
4. Completion conditions must be observable from game state fields.
5. Return ONLY JSON.
"""


# ---------------------------------------------------------------------------
# Prompt templates (Phase 3)
# ---------------------------------------------------------------------------

SUPERVISOR_SYSTEM_PROMPT = """\
You are the Executive Supervisor for an autonomous Pokémon Emerald AI agent.
You receive:
  1. The current Goal Stack (a nested task hierarchy, Stack[0] is the most immediate goal).
  2. A summary of recent dialogue transcript from episodic memory.
  3. A summary of recent battle outcomes from episodic memory.
  4. The current in-game state (location, HP, badges, battle outcome).

Your job is to decide ONE stack operation:
  - POP       : The immediate goal (Stack[0]) was completed. Remove it.
  - CONTINUE  : The immediate goal is NOT yet complete (e.g. an interruption just ended).
  - PUSH      : A new urgent sub-goal has appeared that must be done first.
                Provide the new goal(s) in "new_goals".
  - REPLACE   : The immediate goal is impossible as stated; swap it for a new approach.
                Provide the replacement in "new_goals[0]".

OUTPUT FORMAT — respond with ONLY a JSON object matching this schema:
{
  "operation":  "POP" | "CONTINUE" | "PUSH" | "REPLACE",
  "reasoning":  "<one sentence chain-of-thought>",
  "new_goals":  [                              // required for PUSH or REPLACE
    {
      "goal_id":              "<snake_case_id>",
      "description":          "<what to do>",
      "goal_type":            "strategic" | "tactical" | "immediate",
      "parent_id":            "<id of parent goal or null>",
      "completion_condition": "<observable condition that means this goal is done>",
      "directive": {                           // required for goal_type="immediate"
        "action":       "NAVIGATE" | "INTERACT" | "DIALOGUE" | "CROSS_BOUNDARY",
        "goal_coords":  [x, y, "LOCATION_KEY"] | null,
        "goal_location": "LOCATION_KEY"        | null,
        "should_interact": true | false,
        "npc_coords":   [x, y]                 | null,
        "description":  "<short nav label>"
      },
      "metadata": {}
    }
  ]
}

RULES:
1. Only issue PUSH for goals that are URGENT and BLOCKING (e.g. HP critical, NPC
   blocking path). Do not PUSH for routine sub-steps — the walkthrough RAG handles those.
2. A "POP" is valid ONLY when the completion_condition of Stack[0] is observably met
   in the current game state summary. When in doubt, use CONTINUE.
3. goal_type="immediate" goals MUST include a "directive" block with enough fields
   for the plan controller to act.
4. Never output coordinates you are unsure of — use goal_location only and set
   goal_coords to null. The nav_bot will resolve path automatically.
5. Return ONLY the JSON. No prose before or after.
"""

SUPERVISOR_USER_TEMPLATE = """\
=== CURRENT GOAL STACK ===
{stack_repr}

=== IMMEDIATE GOAL (Stack[0]) ===
Goal ID   : {goal_id}
Type      : {goal_type}
Objective : {goal_description}
Completion: {completion_condition}

=== RECENT DIALOGUE TRANSCRIPT (from episodic memory) ===
{dialogue_context}

=== RECENT BATTLE OUTCOMES (from episodic memory) ===
{battle_context}

=== CURRENT GAME STATE ===
Location  : {current_location}
Position  : ({pos_x}, {pos_y})
Party HP  : {party_hp_summary}
Badges    : {badge_count}
In Battle : {in_battle}
Last Node : {last_node_fired}
Handoff   : {previous_node} → {current_node}
Step      : {step_count}

What stack operation should be performed?
"""


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
        boot_time: float = state.get("_boot_timestamp", 0.0)
        print(f"[SUPERVISOR] boot_timestamp={boot_time:.3f}")

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
        dialogue_ctx: str = _query_dialogue_context(episodic_memory, current_goal, boot_time)
        battle_ctx: str = _query_battle_outcomes(episodic_memory, boot_time)
        logger.debug("[SUPERVISOR] dialogue_ctx: %s", (dialogue_ctx[:100] + "...") if len(dialogue_ctx) > 100 else dialogue_ctx or "(none)")
        logger.debug("[SUPERVISOR] battle_ctx: %s", (battle_ctx[:100] + "...") if len(battle_ctx) > 100 else battle_ctx or "(none)")
        print(f"[SUPERVISOR] boot_timestamp={boot_time:.3f}")
        print(f"[SUPERVISOR] dialogue_ctx: {dialogue_ctx[:80] if dialogue_ctx else '(none)'}")
        print(f"[SUPERVISOR] battle_ctx: {battle_ctx[:80] if battle_ctx else '(none)'}")
        game_summary: dict = _build_game_summary(state_data, state)
        stack_repr: str = stack_summary(stack)

        # ----------------------------------------------------------------
        # 3. Ask the LLM what to do with the stack.
        # ----------------------------------------------------------------
        operation_payload: dict = _call_supervisor_llm(
            vlm, current_goal, dialogue_ctx, battle_ctx, game_summary, stack_repr
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
# Bootstrap and expansion helpers (Phase 4)
# ---------------------------------------------------------------------------

def _bootstrap_stack(state_data: dict, walkthrough_db, vlm) -> list[GoalNode]:
    """Build the initial goal stack from milestones + walkthrough RAG.

    Algorithm:
    1. Read completed milestones from state_data["milestones"].
    2. Determine narrative position: last completed milestone name.
    3. Query walkthrough RAG with progress summary.
    4. Ask LLM to generate a 3-level HTN (strategic/tactical/immediate).
    5. Validate at least one immediate goal with a directive exists.
    6. Fallback to _milestone_fallback_stack on any error or empty DB.
    """
    milestones = state_data.get("milestones", {})
    last_completed = _get_last_completed_milestone(milestones)
    badge_count = _count_badges(state_data)
    location = _get_current_location(state_data)
    location_natural = location.replace("_", " ").title()

    rag_query = _build_rag_query(location, last_completed, location_natural)
    chunks = walkthrough_db.query(rag_query, n_results=5) if walkthrough_db else []
    context_text = "\n\n".join(c["text"] for c in chunks) if chunks else ""

    logger.info(
        "[SUPERVISOR] BOOTSTRAP  last_completed=%s  location=%s  badges=%s  "
        "rag_chunks=%d",
        last_completed, location, badge_count, len(chunks),
    )
    print(f"[SUPERVISOR] last_completed={last_completed}")
    print(f"[SUPERVISOR] RAG query: {rag_query!r}")
    print(f"[SUPERVISOR] RAG returned {len(chunks)} chunks")
    if chunks:
        snippet = chunks[0]["text"][:120].replace("\n", " ")
        print(f"[SUPERVISOR] strategy_ctx (chunk 1): {snippet}...")

    if not context_text or not vlm:
        logger.warning("[SUPERVISOR] Bootstrap: no RAG context or VLM — using milestone fallback.")
        return _milestone_fallback_stack(milestones, state_data)

    htn_prompt = _build_htn_generation_prompt(
        context_text, location, last_completed, badge_count,
        completed_milestones=_infer_completed_milestones(location, milestones),
    )
    try:
        raw = vlm.get_json_query(_HTN_SYSTEM_PROMPT, htn_prompt, module_name="HTNBootstrap", timeout=60)
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        goals_data = json.loads(raw)
        stack = [GoalNode.from_dict(g) for g in goals_data["goals"]]
        assert any(g.goal_type == "immediate" and g.directive for g in stack), \
            "No immediate goal with directive in HTN output"
        return stack
    except Exception as e:
        logger.warning("[SUPERVISOR] HTN generation failed: %s — using milestone fallback", e)
        return _milestone_fallback_stack(milestones, state_data)


def _expand_strategic_goal(
    parent: GoalNode,
    state_data: dict,
    walkthrough_db,
    vlm,
) -> list[GoalNode]:
    """Query walkthrough RAG and LLM to generate the next tactical sub-goals
    for a strategic goal that has had all its children popped."""
    query = f"{parent.description}. What are the next concrete steps?"
    chunks = walkthrough_db.query(query, n_results=4) if walkthrough_db else []
    context = "\n\n".join(c["text"] for c in chunks) if chunks else ""

    if not context or not vlm:
        return []

    prompt = (
        f"Strategic goal: {parent.description}\n"
        f"Completion condition: {parent.completion_condition}\n\n"
        f"Walkthrough context:\n{context}\n\n"
        f"Generate 2-3 tactical sub-goals to make progress on this strategic goal. "
        f"Each must have goal_type='tactical' and parent_id='{parent.goal_id}'. "
        f"The first sub-goal in the array is the most immediate. "
        f"Return JSON in the same schema: {{\"goals\": [...]}}."
    )
    try:
        raw = vlm.get_json_query(_HTN_SYSTEM_PROMPT, prompt, module_name="HTNExpand", timeout=60)
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        goals_data = json.loads(raw)
        return [GoalNode.from_dict(g) for g in goals_data.get("goals", [])]
    except Exception as e:
        logger.warning("[SUPERVISOR] expand_strategic_goal failed: %s", e)
        return []


def _build_htn_generation_prompt(
    context_text: str,
    location: str,
    last_completed: str,
    badge_count: int,
    completed_milestones: list | None = None,
) -> str:
    """Build the user-side HTN generation prompt from RAG context + game state."""
    completed_section = ""
    if completed_milestones:
        completed_section = (
            f"Already completed (DO NOT include these as goals):\n"
            + "\n".join(f"  - {m}" for m in completed_milestones)
            + "\n\n"
        )
    return (
        f"=== WALKTHROUGH CONTEXT ===\n{context_text}\n\n"
        f"=== CURRENT GAME STATE ===\n"
        f"Location     : {location}\n"
        f"Badge count  : {badge_count}\n"
        f"Last milestone: {last_completed}\n\n"
        f"{completed_section}"
        f"Generate a 3-level goal hierarchy (strategic → tactical → immediate) "
        f"to guide the agent from its current position toward the next gym badge. "
        f"Only include goals that are NOT yet completed. "
        f"The immediate goal MUST describe the very next action from the current location. "
        f"The immediate goal MUST include a directive block. "
        f"Return ONLY JSON."
    )


def _get_last_completed_milestone(milestones: dict) -> str:
    """Return the last completed milestone name from MILESTONE_PROGRESSION order.

    Iterates MILESTONE_PROGRESSION in reverse so the most-advanced completed
    milestone is returned.  Falls back to "GAME_RUNNING" when nothing is done.
    """
    for entry in reversed(MILESTONE_PROGRESSION):
        name = entry["milestone"]
        if milestones.get(name):
            return name
    return "GAME_RUNNING"


def _get_effective_progress_index(location: str, last_completed: str) -> int:
    """Return the effective progress index in MILESTONE_PROGRESSION.

    Takes the max of:
      (a) the index of the milestone whose target_location matches the player's
          physical location (normalised to UPPER_UNDERSCORE format), and
      (b) the index of the last tracked completed milestone.

    This handles save states where physical progress exceeds tracked milestones
    (e.g. player is on ROUTE_102 but only OLDALE_TOWN is in the milestones JSON
    because ROUTE_103 / RIVAL_BATTLE_1 / RECEIVED_POKEDEX were never written).
    """
    # Game state uses "ROUTE 102" (spaces); MILESTONE_PROGRESSION uses "ROUTE_102".
    normalized_location = location.replace(" ", "_").upper()
    loc_idx = next(
        (i for i, e in enumerate(MILESTONE_PROGRESSION)
         if e.get("target_location") == normalized_location),
        -1,
    )
    last_idx = next(
        (i for i, e in enumerate(MILESTONE_PROGRESSION)
         if e["milestone"] == last_completed),
        -1,
    )
    return max(loc_idx, last_idx)


def _infer_completed_milestones(location: str, milestones: dict) -> list[str]:
    """Return the full inferred-completed milestone list for the HTN prompt.

    Combines:
    1. Milestones explicitly tracked in the JSON.
    2. All milestones up to and including the player's physical location index
       (because if you're on Route 102 you must have done the Route 103 battle).
    """
    last_completed = _get_last_completed_milestone(milestones)
    effective_idx = _get_effective_progress_index(location, last_completed)
    inferred = {
        entry["milestone"]
        for i, entry in enumerate(MILESTONE_PROGRESSION)
        if i <= effective_idx
    }
    # Also include anything explicitly in the JSON
    tracked = {name for name, val in milestones.items() if val}
    return sorted(inferred | tracked, key=lambda m: next(
        (i for i, e in enumerate(MILESTONE_PROGRESSION) if e["milestone"] == m), 999
    ))


def _build_rag_query(location: str, last_completed: str, location_natural: str) -> str:
    """Build a content-rich RAG query for the current game position.

    Anchors on the player's physical location (not just the tracked milestone)
    to handle gaps between milestone tracking and actual game progress.
    Uses the next milestone descriptions to surface semantically relevant chunks.
    """
    effective_idx = _get_effective_progress_index(location, last_completed)

    # Find the next non-current target location (our destination)
    normalized_location = location.replace(" ", "_").upper()
    next_location_natural = None
    for entry in MILESTONE_PROGRESSION[effective_idx + 1:]:
        tl = entry.get("target_location")
        if tl and tl != normalized_location:
            next_location_natural = tl.replace("_", " ").title()
            break

    # Collect descriptions for the next 4 milestones to enrich the query
    next_milestone_descs = [
        e["description"]
        for e in MILESTONE_PROGRESSION[effective_idx + 1: effective_idx + 5]
        if e.get("description")
    ]
    desc_text = ". ".join(next_milestone_descs)

    if next_location_natural:
        return (
            f"Travel from {location_natural} to {next_location_natural}. "
            f"Visit {next_location_natural} gym or key location. {desc_text}."
        )
    return f"What to do at {location_natural}. {desc_text}."


def _milestone_fallback_stack(milestones: dict, state_data: dict) -> list[GoalNode]:
    """Build a shallow stack from MILESTONE_PROGRESSION when RAG/LLM is unavailable.

    Finds the first incomplete milestone *after* the last completed one and
    creates a single immediate goal pointing to its target location.  This
    prevents picking a milestone that was never tracked in the JSON but was
    already completed in the save state (e.g. INTRO_CUTSCENE_COMPLETE absent
    from boundary_test_milestones.json even though the intro is long done).

    Falls back to scanning from index 0 only when no completed milestone is
    found (fresh game with empty milestones dict).  Returns an empty list when
    MILESTONE_PROGRESSION is fully exhausted.
    """
    last_completed = _get_last_completed_milestone(milestones)
    # Find the index of the last completed milestone so we start scanning after it.
    last_idx = -1
    for i, entry in enumerate(MILESTONE_PROGRESSION):
        if entry["milestone"] == last_completed:
            last_idx = i
            break

    for entry in MILESTONE_PROGRESSION[last_idx + 1:]:
        name = entry["milestone"]
        if milestones.get(name):
            continue
        target_location = entry.get("target_location")
        description = entry.get("description", name)
        directive = None
        if target_location:
            directive = {
                "action": "NAVIGATE",
                "goal_coords": None,
                "goal_location": target_location,
                "should_interact": False,
                "npc_coords": None,
                "description": f"Navigate to {target_location}",
            }
        goal = GoalNode(
            goal_id=name.lower(),
            description=description,
            goal_type="immediate",
            parent_id=None,
            completion_condition=f"milestone {name} completed",
            directive=directive,
            metadata={"milestone": name},
        )
        return [goal]
    return []


def _count_badges(state_data: dict) -> int:
    """Return the number of badges earned from state_data."""
    game = state_data.get("game") or {}
    badges_raw = game.get("badges", 0)
    if isinstance(badges_raw, int):
        return badges_raw
    if isinstance(badges_raw, dict):
        return sum(1 for v in badges_raw.values() if v)
    return 0


def _get_current_location(state_data: dict) -> str:
    """Return the player's current location string from state_data."""
    player = state_data.get("player") or {}
    return player.get("location") or "Unknown"


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

def _query_dialogue_context(
    episodic_memory,
    current_goal: Optional[GoalNode],
    boot_time: float = 0.0,
    n: int = 5,
) -> str:
    """Return post-boot dialogue_transcript records relevant to the current goal.

    Filters ChromaDB by ``type == dialogue_transcript`` and
    ``timestamp >= boot_time`` so pre-run records never appear in the
    Supervisor's context.  Returns ``""`` (empty string) when nothing is
    found — callers render this as ``"(none)"`` in the prompt template.
    """
    if episodic_memory is None or current_goal is None:
        return ""
    try:
        collection = episodic_memory.collection
        count = collection.count()
        if count == 0:
            return ""
        query = (
            f"NPC dialogue relevant to: '{current_goal.description}'. "
            f"Looking for keywords in: {current_goal.completion_condition}"
        )
        results = collection.query(
            query_texts=[query],
            n_results=min(n, count),
            where={
                "$and": [
                    {"type": {"$eq": "dialogue_transcript"}},
                    {"timestamp": {"$gte": boot_time}},
                ]
            },
            include=["documents", "metadatas"],
        )
        docs = results.get("documents", [[]])[0]
        return "\n".join(docs) if docs else ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SUPERVISOR] Dialogue context query failed: %s", exc)
        return ""


def _query_battle_outcomes(
    episodic_memory,
    boot_time: float = 0.0,
    n: int = 3,
) -> str:
    """Return post-boot battle_outcome records from episodic memory.

    Filters ChromaDB by ``type == battle_outcome`` and
    ``timestamp >= boot_time``.  Returns ``""`` when nothing is found.
    """
    if episodic_memory is None:
        return ""
    try:
        collection = episodic_memory.collection
        count = collection.count()
        if count == 0:
            return ""
        results = collection.query(
            query_texts=["recent battle outcome party HP won lost"],
            n_results=min(n, count),
            where={
                "$and": [
                    {"type": {"$eq": "battle_outcome"}},
                    {"timestamp": {"$gte": boot_time}},
                ]
            },
            include=["documents", "metadatas"],
        )
        docs = results.get("documents", [[]])[0]
        return "\n".join(docs) if docs else ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SUPERVISOR] Battle outcomes query failed: %s", exc)
        return ""


def _build_game_summary(state_data: dict, state: dict) -> dict:
    """Build a dict of game-state fields for the supervisor prompt template."""
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

    last_node = state.get("last_node_fired") or "unknown"
    context = state.get("context") or "unknown"

    return {
        "current_location": location,
        "pos_x": x,
        "pos_y": y,
        "party_hp_summary": hp_summary,
        "badge_count": badges,
        "in_battle": in_battle,
        "last_node_fired": last_node,
        "previous_node": last_node,
        "current_node": context,
        "step_count": state.get("step_count", 0),
    }


def _call_supervisor_llm(
    vlm,
    current_goal: Optional[GoalNode],
    dialogue_ctx: str,
    battle_ctx: str,
    game_summary: dict,
    stack_repr: str,
) -> dict:
    """Call the VLM supervisor and parse its stack-operation decision.

    Builds SUPERVISOR_USER_TEMPLATE, calls ``vlm.get_json_query()``, and
    validates the returned JSON.  On any failure (network error, bad JSON,
    unrecognised operation) returns a safe CONTINUE payload so the node
    never crashes.

    Returns a dict with at minimum:
        ``{"operation": str, "reasoning": str, "new_goals": list}``
    """
    if vlm is None:
        return {"operation": "CONTINUE", "reasoning": "No VLM available.", "new_goals": []}

    if current_goal is None:
        return {"operation": "CONTINUE", "reasoning": "No current goal.", "new_goals": []}

    prompt = SUPERVISOR_USER_TEMPLATE.format(
        stack_repr=stack_repr,
        goal_id=current_goal.goal_id,
        goal_type=current_goal.goal_type,
        goal_description=current_goal.description,
        completion_condition=current_goal.completion_condition or "(none)",
        dialogue_context=dialogue_ctx or "(none)",
        battle_context=battle_ctx or "(none)",
        **game_summary,
    )
    try:
        raw = vlm.get_json_query(SUPERVISOR_SYSTEM_PROMPT, prompt, module_name="Supervisor")
        # Belt-and-suspenders: strip markdown fences for backends that don't
        # support native JSON mode and may wrap the output.
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        payload = json.loads(raw)
        assert payload.get("operation") in ("POP", "CONTINUE", "PUSH", "REPLACE")
        return payload
    except Exception as e:
        logger.warning("[SUPERVISOR] LLM parse error: %s — defaulting to CONTINUE", e)
        return {"operation": "CONTINUE", "reasoning": f"parse_error: {e}", "new_goals": []}
