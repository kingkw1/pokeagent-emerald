# HTN Migration Plan — Executive Supervisor & Goal Stack Architecture

**Document Status:** Implementation Blueprint  
**Replaces:** `OBJECTIVE_TRACKING_SYSTEM.md` (legacy milestone FSM)  
**Codebase snapshot:** April 26, 2026 (LangGraph graph stable, route102 corridor ✅)

---

## Executive Summary

We are replacing the open-loop, hardcoded `MILESTONE_PROGRESSION` FSM with a
**Hierarchical Task Network (HTN)** driven by an LLM Executive Supervisor. The
new system treats the goal stack as the agent's "working plan" — a nested tree
of goals that the Supervisor rewrites dynamically as game state evolves. The
plant controllers (`nav_bot`, `battle_bot`, `coms_bot`) remain unchanged; they
only need a valid `Directive` extracted from the top of the goal stack, not a
hardcoded list index.

The migration is designed to be **incremental and reversible**. At every phase,
a `MILESTONE_PROGRESSION` fallback path is available if the HTN produces an
empty or invalid stack.

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AgentState (LangGraph)                          │
│   goal_stack: List[GoalNode]   ← new; Stack[0] = immediate tactical     │
│   last_node_fired: str         ← new; tracks handoff detection          │
│   supervisor_pending: bool     ← new; flag to trigger Supervisor        │
│   milestone_index: int         ← retained for fallback + verification   │
└───────────────┬─────────────────────────────────────────────────────────┘
                │
         dispatch node (unchanged)
                │ routing_condition() — unchanged
        ┌───────┼─────────┬──────────┐
        ▼       ▼         ▼          ▼
    nav_bot  battle_bot  coms_bot  map_stitcher_relay
        │       │         │          │
        └───────┴────┬────┘          │
                     ▼               │
             handoff_detector_node ◄─┘
                     │
           supervisor_pending?
                YES  │  NO
                 ▼   │
    executive_supervisor_node
                 │
                 ▼
         verification_node (unchanged)
                 │
                END
```

The key insight: the Supervisor is **not on the hot path**. It only fires on
state handoffs (node transitions), keeping per-frame latency near zero for the
common case.

---

## Testing Eras — What Save-State Tests Look Like Per Phase

Before diving into individual phases, this section maps what the running agent
actually does at each implementation stage. This answers the question: *"Can I
use `boundary_test.state` to test at every phase, or do I have to wait until
Phase 6?"*

**Short answer: save states work from Phase 0.** The emulator populates
`state_data["milestones"]` from the companion `*_milestones.json` file at
startup. This is emulator infrastructure, not HTN code. Phase 6's
`_boot_timestamp` is purely a ChromaDB staleness guard (preventing stale
pre-run records from misleading the Supervisor) — it has no effect on save-state
loading correctness.

| Phase range | Who drives navigation? | What fires on a handoff? | Save-state usable? |
|---|---|---|---|
| **0–1** | Legacy FSM (`ObjectiveManager` + `MILESTONE_PROGRESSION`) | `handoff_detector` logs the transition but `supervisor_pending` has nowhere to go (no supervisor node) | ✅ Yes — same as today |
| **2–3** | Legacy FSM | Supervisor fires, but `_bootstrap_stack` is a stub returning `[]`; `_apply_immediate_directive` is a no-op for empty stack; `--use-htn` is **OFF** by default | ✅ Yes — agent navigates identically to Phase 0–1 |
| **4–5** | Legacy FSM (default) OR HTN (with `--use-htn`) | Supervisor fires; `_bootstrap_stack` builds a real stack from milestones JSON + RAG; directive applied only when `--use-htn` is set | ✅ Yes — `boundary_test.state` milestones JSON anchors the bootstrap; run without `--use-htn` to keep FSM as the authority |
| **6** | Same as Phase 4–5 | `_boot_timestamp` now filters stale ChromaDB records from episodic context | ✅ Yes — adds ChromaDB isolation, doesn't change navigation |
| **7.1** | Legacy FSM (shadow mode — `--use-htn` still off) | Supervisor output logged to `htn_shadow.jsonl` for comparison; no nav field changes | ✅ Yes |
| **7.2+** | HTN (with `--use-htn`) | Stack[0] directive replaces ObjectiveManager directive | ✅ Yes — this is the first step where HTN actually steers the agent |

**The critical design constraint this implies:**

> `_apply_immediate_directive` MUST be gated by a `use_htn` flag from Phase 2
> onward. If it isn't, Phases 4–5 inadvertently become live HTN navigation
> before you've decided to flip that switch. The `--use-htn` flag is NOT a
> Phase 7 concern — it belongs in the factory signature of
> `make_executive_supervisor_node()` at Phase 2.

This is corrected in the Phase 2 implementation below.

---

## Phase 0: Goal Stack Data Structures

**Purpose:** Create the data types that every other phase depends on. `GoalNode` is the unit of the HTN; the stack primitives (`push`, `pop`, `peek`, `replace`) are the only mutations allowed. Nothing in Phases 1–7 can be built until these exist and are tested.

### 0.1 `GoalNode` — The Unit of the HTN

**File to create:** `agent/graph/goal_stack.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any
import time


@dataclass
class GoalNode:
    """A single node in the HTN goal stack.

    Attributes:
        goal_id:        Unique identifier (e.g. "get_badge_1", "traverse_route_102").
        description:    Human-readable description of this goal.
        goal_type:      "strategic" | "tactical" | "immediate"
                        strategic  = high-level quest objective (e.g. "Defeat Roxanne")
                        tactical   = mid-level plan step  (e.g. "Reach Rustboro Gym")
                        immediate  = single nav/battle/coms directive
        parent_id:      goal_id of the parent goal (None for root).
        directive:      Optional pre-computed Directive for immediate goals.
                        When set, the executor uses it directly without re-querying
                        the Supervisor.
        completion_condition: Natural-language string the Supervisor checks to
                        decide whether to POP this goal. E.g.:
                        "Player is in RUSTBORO_CITY_GYM and has interacted with Roxanne."
        metadata:       Arbitrary dict for Supervisor context (e.g. badge count
                        threshold, required items, HP constraint).
        created_at:     Unix timestamp.
        push_reason:    Why this goal was pushed (for logging / debugging).
    """
    goal_id: str
    description: str
    goal_type: str                          # "strategic" | "tactical" | "immediate"
    parent_id: Optional[str] = None
    directive: Optional[dict] = None        # serialisable Directive.to_dict()
    completion_condition: str = ""
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    push_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "goal_type": self.goal_type,
            "parent_id": self.parent_id,
            "directive": self.directive,
            "completion_condition": self.completion_condition,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "push_reason": self.push_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GoalNode":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Stack operations (pure functions — no mutation of AgentState directly)
# ---------------------------------------------------------------------------

def stack_peek(stack: list[GoalNode]) -> Optional[GoalNode]:
    """Return Stack[0] (immediate goal) without removing it."""
    return stack[0] if stack else None

def stack_pop(stack: list[GoalNode]) -> tuple[Optional[GoalNode], list[GoalNode]]:
    """Remove and return Stack[0]."""
    if not stack:
        return None, []
    return stack[0], stack[1:]

def stack_push(stack: list[GoalNode], goal: GoalNode) -> list[GoalNode]:
    """Prepend a new immediate goal to the front of the stack."""
    return [goal] + stack

def stack_replace(stack: list[GoalNode], goal: GoalNode) -> list[GoalNode]:
    """Replace Stack[0] with a new goal."""
    return [goal] + (stack[1:] if len(stack) > 1 else [])

def stack_summary(stack: list[GoalNode]) -> str:
    """Return a compact one-line summary for logging."""
    if not stack:
        return "(empty)"
    return " → ".join(f"[{g.goal_type[0].upper()}]{g.description}" for g in reversed(stack))
```

### 0.2 `AgentState` Schema Changes

**File:** `agent/graph/state.py`

Add the following fields to `AgentState`. Existing fields are **unchanged**;
`milestone_index` is retained as a fallback signal during the migration period.

```python
# ---- HTN Goal Stack ----
goal_stack: list
"""Ordered list of GoalNode dicts (serialised). Stack[0] = immediate goal.
Stack[-1] = highest strategic goal. Empty list = stack exhausted."""

last_node_fired: Optional[str]
"""Name of the specialist node that just completed (set by handoff_detector_node).
E.g. 'battle_bot'. Used to detect handoffs and gate the Supervisor."""

supervisor_pending: bool
"""When True, executive_supervisor_node fires after the current step's
handoff_detector_node completes. Reset to False by the Supervisor."""

supervisor_last_operation: Optional[str]
"""The last stack operation issued: 'POP' | 'CONTINUE' | 'PUSH' | 'REPLACE'.
Logged for observability."""

supervisor_last_reasoning: Optional[str]
"""The Supervisor's free-text chain-of-thought (truncated to ~500 chars).
Stored for offline analysis in llm_logs/."""
```

**Important:** `goal_stack` stores `List[dict]` (serialised `GoalNode.to_dict()`)
rather than `List[GoalNode]` objects, because LangGraph's state reducer requires
JSON-serialisable values. Deserialise with `GoalNode.from_dict()` at node
boundaries.

### Phase 0 Tests

**Automated — `tests/test_goal_stack.py`:**

```python
class TestGoalNodeSerialization:
    # GoalNode(...).to_dict() produces a dict with all expected keys
    # GoalNode.from_dict(d) reconstructs an identical GoalNode
    # Round-trip preserves goal_id, description, goal_type, parent_id, directive, metadata

class TestGoalNodeDefaults:
    # GoalNode with only required fields has created_at > 0.0
    # directive defaults to None
    # metadata defaults to {}

class TestStackPush:
    # stack_push([], goal) returns [goal]
    # stack_push([existing], new) returns [new, existing]  ← new is Stack[0]

class TestStackPop:
    # stack_pop([a, b]) returns (a, [b])
    # stack_pop([]) returns (None, [])

class TestStackReplace:
    # stack_replace([old, parent], new) returns [new, parent]
    # stack_replace([], new) returns [new]

class TestStackPeek:
    # stack_peek([a, b]) returns a without mutating the list
    # stack_peek([]) returns None

class TestStackSummary:
    # Three-level stack → summary shows immediate → tactical → strategic order
    # Empty stack → returns "(empty)"
```

**Automated — `tests/test_agent_state_htn.py`:**

```python
class TestNewFieldsPresent:
    # AgentState constructed with goal_stack=[], supervisor_pending=False — no TypeError
    # All five new HTN fields accept correct types

class TestGoalStackDefaultsEmpty:
    # AgentState with no goal_stack key → state.get("goal_stack", []) == []

class TestSupervisorPendingDefault:
    # supervisor_pending not set → state.get("supervisor_pending", False) == False
```

**Manual — Phase 0 Schema Smoke Test:**

*Purpose:* Confirm `GoalNode` serialisation and the new `AgentState` fields do not crash on import.

*Command:*
```bash
PYTHONPATH=$PWD .venv/bin/python -c "
from agent.graph.goal_stack import GoalNode, stack_push, stack_summary
g = GoalNode('test', 'Walk into Petalburg City', 'immediate',
             directive={'action': 'NAVIGATE', 'goal_location': 'PETALBURG_CITY'})
stack = stack_push([], g)
print('Stack:', stack_summary(stack))
print('Roundtrip:', GoalNode.from_dict(g.to_dict()).goal_id)
"
```

*Pass criteria:*
- [x] No `ImportError` or `AttributeError`
- [x] `stack_summary` prints the goal description
- [x] `Roundtrip` prints `test`

*Fail indicators:*
- `ImportError: cannot import name 'GoalNode'` — `agent/graph/goal_stack.py` not yet created
- `KeyError` in `from_dict` — field name mismatch between `to_dict` and `__dataclass_fields__`

*Status:* ✅ PASSED — 58/58 automated tests green; smoke test output:
```
Stack: [I]Walk into Petalburg City
Roundtrip: test
```

---

## Phase 1: Handoff Detector Node

**Purpose:** Insert a cheap gatekeeper node into the graph that watches for meaningful state transitions and sets `supervisor_pending = True` when something worth re-planning has happened. Without this, the Supervisor would either fire every step (too expensive) or never fire at all. This phase adds the wiring to the graph without any LLM calls.

The `handoff_detector_node` is a **lightweight, zero-LLM** node inserted
between every specialist node and `verification_node`. It sets
`supervisor_pending = True` when a transition between node types occurs.

**File to create:** `agent/graph/nodes/handoff_detector.py`

```python
from __future__ import annotations
from agent.graph.state import AgentState

# Transitions that require Supervisor review.
# A transition is "significant" when the node type changes.
# Same-node re-entries (e.g. nav_bot → nav_bot) do NOT trigger the Supervisor
# because no meaningful state change has occurred.
_SIGNIFICANT_TRANSITIONS = {
    ("battle_bot",        "nav_bot"),       # battle ended → resume navigation
    ("battle_bot",        "coms_bot"),      # mid-battle dialogue
    ("coms_bot",          "nav_bot"),       # dialogue finished → resume navigation
    ("nav_bot",           "coms_bot"),      # NPC triggered mid-navigation
    ("nav_bot",           "battle_bot"),    # wild encounter / trainer spotted
    ("map_stitcher_relay","nav_bot"),       # healing path resolved → navigate
}

# --- Nav-stall detection -------------------------------------------------------
# nav_bot → nav_bot re-entries are ignored by _SIGNIFICANT_TRANSITIONS, creating
# a blind spot: if the agent is stuck in a nav loop the Supervisor never wakes.
# The stuck_handler fixes tile-level oscillation automatically (dynamic tile
# blocking, TTL=200), but cannot detect *goal-level* stalls where the current
# goal is simply unreachable. We detect these here using module-level position
# tracking — same pattern as stuck_handler.py — without coupling to that module.
#
# Threshold rationale:
#   _stuck_counter resets to 0 after each tile-block attempt (stuck_handler line
#   199-211), so it never accumulates above 3. Reading it from here would always
#   show a low value. Instead we track position epochs in the detector itself.
#   15 consecutive steps at the same (x, y, location) means tile-level recovery
#   has not cleared the stall — escalate to the Supervisor (goal-level replanning).
_NAV_STALL_THRESHOLD = 15
_consecutive_nav_stall_steps: int = 0
_last_nav_position: tuple | None = None
# Also trigger on the very first step (no previous node) or when the
# goal stack becomes empty.
def handoff_detector_node(state: AgentState) -> AgentState:
    global _consecutive_nav_stall_steps, _last_nav_position

    current_node  = state.get("last_action", "")   # e.g. "NAVIGATE", "BATTLE", "DIALOGUE"
    previous_node = state.get("last_node_fired", "")
    goal_stack    = state.get("goal_stack", [])

    # Map last_action labels back to node names
    _ACTION_TO_NODE = {
        "NAVIGATE": "nav_bot",
        "BATTLE":   "battle_bot",
        "DIALOGUE": "coms_bot",
    }
    current_node_name  = _ACTION_TO_NODE.get(current_node, current_node)
    previous_node_name = _ACTION_TO_NODE.get(previous_node, previous_node)

    transition = (previous_node_name, current_node_name)
    is_significant = (
        transition in _SIGNIFICANT_TRANSITIONS
        or not previous_node_name          # first step
        or not goal_stack                  # stack exhausted
    )

    # Nav-stall check: fire Supervisor when position hasn't changed for
    # _NAV_STALL_THRESHOLD consecutive nav_bot steps.
    if current_node_name == "nav_bot":
        player = (state.get("state_data") or {}).get("player", {})
        pos    = player.get("position", {})
        nav_pos = (pos.get("x"), pos.get("y"), player.get("location"))
        if nav_pos == _last_nav_position:
            _consecutive_nav_stall_steps += 1
        else:
            _consecutive_nav_stall_steps = 0
        _last_nav_position = nav_pos
        if _consecutive_nav_stall_steps >= _NAV_STALL_THRESHOLD:
            logger.warning(
                "[HANDOFF] Nav stall detected: %d consecutive steps at %s "
                "— waking Supervisor for goal-level replanning",
                _consecutive_nav_stall_steps, nav_pos,
            )
            is_significant = True
            _consecutive_nav_stall_steps = 0  # reset; don't fire every step
    else:
        # Reset stall counter whenever we leave nav_bot
        _consecutive_nav_stall_steps = 0
        _last_nav_position = None

    return {
        **state,
        "last_node_fired": current_node_name,
        "supervisor_pending": is_significant,
    }
```

### Graph Wiring Change

**File:** `agent/graph/graph.py`

```python
from agent.graph.nodes.handoff_detector import handoff_detector_node
from agent.graph.nodes.executive_supervisor import make_executive_supervisor_node

# Add nodes
builder.add_node("handoff_detector", handoff_detector_node)
builder.add_node("executive_supervisor", make_executive_supervisor_node(
    vlm=vlm,
    episodic_memory=episodic_memory,
    walkthrough_db=walkthrough_db,
))

# Rewire edges: specialist → handoff_detector → (conditional) → verification
for specialist in ["nav_bot", "battle_bot", "coms_bot", "map_stitcher_relay"]:
    builder.add_edge(specialist, "handoff_detector")

builder.add_conditional_edges(
    "handoff_detector",
    lambda s: "executive_supervisor" if s.get("supervisor_pending") else "verification",
    {"executive_supervisor": "executive_supervisor", "verification": "verification"},
)
builder.add_edge("executive_supervisor", "verification")
```

### Phase 1 Tests

**Automated — `tests/test_handoff_detector.py`:**

```python
class TestSignificantTransition:
    # previous_node="battle_bot", current_node="nav_bot"   → supervisor_pending=True
    # previous_node="coms_bot",   current_node="nav_bot"   → supervisor_pending=True
    # previous_node="nav_bot",    current_node="coms_bot"  → supervisor_pending=True
    # previous_node="nav_bot",    current_node="battle_bot" → supervisor_pending=True

class TestInsignificantTransition:
    # previous_node="nav_bot",    current_node="nav_bot"    → supervisor_pending=False
    # previous_node="battle_bot", current_node="battle_bot" → supervisor_pending=False

class TestFirstStep:
    # last_node_fired not set (empty string) → supervisor_pending=True regardless of current_node

class TestEmptyStack:
    # goal_stack=[], any transition → supervisor_pending=True

class TestLastNodeFiredUpdated:
    # handoff_detector_node returns state with last_node_fired = current_node_name
    # last_action="NAVIGATE" → last_node_fired="nav_bot"
    # last_action="BATTLE"   → last_node_fired="battle_bot"
    # last_action="DIALOGUE" → last_node_fired="coms_bot"

class TestNavStallDetection:
    # Call handoff_detector_node 14 times with nav_bot + same (x, y, location)
    # → supervisor_pending=False on all 14 calls (below threshold)
    # Call a 15th time with same position
    # → supervisor_pending=True (threshold crossed)
    # Call a 16th time with same position
    # → supervisor_pending=False (counter was reset on the 15th call)
    # Call with changing position throughout (x changes each step)
    # → supervisor_pending=False on all calls (stall counter never accumulates)
    # After 15 stall steps, switch to battle_bot (different node)
    # → _consecutive_nav_stall_steps resets; no spurious supervisor_pending on return to nav_bot

    # NOTE: these tests must call the node through a function that resets the
    # module-level globals first, to avoid inter-test contamination:
    # import agent.graph.nodes.handoff_detector as hd
    # hd._consecutive_nav_stall_steps = 0
    # hd._last_nav_position = None
```

**Manual — Handoff Detector Smoke Test (`boundary_test.state`):**

*Purpose:* Confirm `handoff_detector_node` fires `supervisor_pending=True` on the
first step (empty stack) and again on the nav_bot → coms_bot handoff when the
agent enters Petalburg City and encounters an NPC. The save state has the player
at the eastern entrance of Petalburg City; the next actions are to walk west
into the city, navigate to the gym, and trigger the Norman cutscene.

*Setup:* Add a temporary `[HANDOFF]` print inside `handoff_detector_node`:
```python
print(f"[HANDOFF] step={state.get('step_count')}  "
      f"{previous_node_name} → {current_node_name}  pending={is_significant}  "
      f"stall={_consecutive_nav_stall_steps}")
```

*Command:*
```bash
python run.py --load-state tests/save_states/boundary_test.state --agent-auto
```

*Observe in console:*
- `[HANDOFF]` line on step 1 with `pending=True` (first step, empty stack)
- Subsequent `[HANDOFF]` lines with `pending=False` while nav_bot repeats
- When routing changes to coms_bot (NPC dialogue in Petalburg City): `nav_bot → coms_bot  pending=True`

*Pass criteria:*
- [x] `pending=True` on step 1
- [x] `pending=False` on all consecutive nav_bot → nav_bot steps (moving)
- [x] `pending=True` fires when routing first changes to coms_bot
- [x] `last_node_fired` value in state matches the previous step's active node

*Fail indicators:*
- `pending=True` every single step: `last_node_fired` is not persisting between steps — check that `handoff_detector_node` writes it to the returned state dict
- `pending=False` on step 1: `goal_stack` is pre-populated somewhere before `handoff_detector_node` runs
- No `[HANDOFF]` lines at all: node not yet wired into `graph.py`
- `stall=` counter never increments even when nav is stuck: `state_data["player"]["position"]` path is wrong — print `state.get("state_data", {})` to verify the key structure
- Supervisor fires every step after a stall: `_consecutive_nav_stall_steps` is not being reset to 0 after the threshold fires

*Status:* ✅ PASSED — 28/28 automated tests green; manual smoke test passed (run_20260427_142343.log)

---

## Phase 2: Executive Supervisor Node

**Purpose:** Build the LLM decision node that reads the goal stack, game state, and episodic context to issue exactly one stack operation (POP / CONTINUE / PUSH / REPLACE). After this phase the Supervisor exists in the graph and runs on every handoff, but `use_htn=False` by default so the legacy FSM still drives navigation. This is the largest single phase.

**File to create:** `agent/graph/nodes/executive_supervisor.py`

### 2.1 Trigger Conditions (when it fires)

The Supervisor fires when `supervisor_pending == True`, which is set by
`handoff_detector_node` on:
1. Any **state handoff** (node type transition) listed in `_SIGNIFICANT_TRANSITIONS`
2. **First step** of any run (bootstraps the goal stack from `milestones.json`)
3. **Empty stack** (all goals complete; need new HTN branch)

### 2.2 Core Logic

```python
from __future__ import annotations
import json, logging, time
from typing import Callable, Optional

from agent.graph.state import AgentState
from agent.graph.goal_stack import (
    GoalNode, stack_peek, stack_pop, stack_push,
    stack_replace, stack_summary,
)

logger = logging.getLogger(__name__)


def make_executive_supervisor_node(
    vlm,
    episodic_memory,
    walkthrough_db,
    use_htn: bool = False,          # Phase 7.2: set True to let HTN drive nav fields
) -> Callable[[AgentState], AgentState]:
    """Factory binding shared resources into the supervisor node.

    Args:
        vlm:           VLM instance.
        episodic_memory: EpisodicMemory instance.
        walkthrough_db:  WalkthroughDB instance (or None — triggers fallback).
        use_htn:       When False (default), the supervisor builds and maintains
                       the goal stack but does NOT overwrite AgentState nav fields
                       (``goal_coords``, ``goal_location``, etc.). The legacy FSM
                       continues to drive navigation. Set True (Phase 7.2+) to
                       hand navigation over to the HTN.
    """

    def executive_supervisor_node(state: AgentState) -> AgentState:
        step      = state.get("step_count", 0)
        stack_raw = state.get("goal_stack", [])
        stack     = [GoalNode.from_dict(g) for g in stack_raw]
        state_data = state.get("state_data") or {}

        # ── 1. Bootstrap empty stack from milestones.json ──────────────────
        if not stack:
            # NOTE (Phase 2–3): _bootstrap_stack is a stub returning [] until
            # Phase 4 adds the real RAG implementation.  When it returns [],
            # the supervisor is a no-op for this step (stack remains empty,
            # nav fields untouched, legacy FSM drives navigation as normal).
            stack = _bootstrap_stack(state_data, walkthrough_db, vlm)
            if not stack:
                # Stub returned empty — nothing to do yet
                return {**state, "supervisor_pending": False}
            logger.info("[SUPERVISOR] step=%s  Bootstrapped stack: %s",
                        step, stack_summary(stack))
            return {
                **state,
                "goal_stack": [g.to_dict() for g in stack],
                "supervisor_pending": False,
                "supervisor_last_operation": "BOOTSTRAP",
            }

        # ── 2. Gather context for LLM reasoning ────────────────────────────
        current_goal = stack_peek(stack)
        episodic_ctx = _query_episodic_memory(
            episodic_memory, current_goal, state_data
        )
        game_summary  = _build_game_summary(state_data, state)
        stack_repr    = stack_summary(stack)

        # ── 3. Call LLM → get stack operation ──────────────────────────────
        operation_payload = _call_supervisor_llm(
            vlm, current_goal, episodic_ctx, game_summary, stack_repr
        )

        # ── 4. Apply stack operation ────────────────────────────────────────
        op      = operation_payload.get("operation", "CONTINUE")
        reason  = operation_payload.get("reasoning", "")
        new_goals = operation_payload.get("new_goals", [])

        if op == "POP":
            popped, stack = stack_pop(stack)
            logger.info("[SUPERVISOR] POP '%s' — %s", popped.goal_id if popped else "?", reason)
            # If popped a tactical goal and parent is now strategic, trigger
            # walkthrough RAG to repopulate sub-goals
            parent = stack_peek(stack)
            if parent and parent.goal_type == "strategic" and not _has_children(stack, parent):
                new_sub_goals = _expand_strategic_goal(parent, state_data, walkthrough_db, vlm)
                for g in reversed(new_sub_goals):
                    stack = stack_push(stack, g)

        elif op == "PUSH":
            # Hard depth cap: an unbounded PUSH loop is an LLM failure mode.
            # A healthy 3-level HTN (strategic → tactical → immediate) needs at
            # most 3–4 entries; 8 is a generous upper bound that still prevents
            # runaway growth.  When the cap is hit, demote to CONTINUE so the
            # agent keeps moving while the plan is logged for diagnosis.
            _STACK_DEPTH_CAP = 8
            if len(stack) >= _STACK_DEPTH_CAP:
                logger.warning(
                    "[SUPERVISOR] PUSH rejected: stack depth %d >= cap %d "
                    "(reasoning: %s) — demoting to CONTINUE to prevent runaway "
                    "stack growth.  This indicates the LLM is looping; review "
                    "llm_logs for repeated PUSH operations on the same goal.",
                    len(stack), _STACK_DEPTH_CAP, reason,
                )
                op = "CONTINUE"
            else:
                for g_dict in new_goals:
                    node = GoalNode.from_dict({**g_dict, "push_reason": reason})
                    stack = stack_push(stack, node)
                logger.info("[SUPERVISOR] PUSH %d goal(s) — %s", len(new_goals), reason)

        elif op == "REPLACE":
            if new_goals:
                node = GoalNode.from_dict({**new_goals[0], "push_reason": reason})
                stack = stack_replace(stack, node)
            logger.info("[SUPERVISOR] REPLACE Stack[0] — %s", reason)

        else:  # CONTINUE
            logger.debug("[SUPERVISOR] CONTINUE — %s", reason)

        # ── 5. Translate Stack[0].directive → AgentState nav fields ────────
        # Gate on use_htn: when False (default through Phase 7.1), the stack is
        # maintained and logged but nav fields are NOT overwritten.  The legacy
        # FSM directive computed by nav_bot earlier in this graph step is preserved.
        if use_htn:
            new_state = _apply_immediate_directive(state, stack)
        else:
            new_state = dict(state)   # shadow mode: stack updated, nav untouched
        new_state.update({
            "goal_stack": [g.to_dict() for g in stack],
            "supervisor_pending": False,
            "supervisor_last_operation": op,
            "supervisor_last_reasoning": reason[:500],
        })
        return new_state

    return executive_supervisor_node
```

### 2.3 Directive Translation: `_apply_immediate_directive`

The Supervisor outputs human-readable `GoalNode` objects. The `nav_bot` and
other plant controllers need concrete `AgentState` fields (`goal_coords`,
`goal_location`, `should_interact`, etc.). This translation lives in a single
helper so the mapping is explicit and testable:

```python
def _apply_immediate_directive(state: AgentState, stack: list[GoalNode]) -> dict:
    """Extract the immediate goal's directive and map it to AgentState fields."""
    immediate = stack_peek(stack)
    if not immediate or not immediate.directive:
        # No directive yet — preserve existing nav fields, let nav_bot use fallback
        return dict(state)

    from agent.objective_manager import Directive
    d = Directive.from_dict(immediate.directive)

    patch = {}
    if d.goal_coords:
        patch["goal_coords"]    = d.goal_coords
    if d.goal_location:
        patch["goal_location"]  = d.goal_location
    if d.npc_coords:
        patch["npc_coords"]     = d.npc_coords
    if d.should_interact is not None:
        patch["should_interact"] = d.should_interact
    if d.description:
        patch["goal_description"] = d.description
    if immediate.goal_id:
        patch["active_milestone"] = immediate.goal_id

    return {**state, **patch}
```

### 2.4 `_bootstrap_stack` Stub (Phases 2–3 Placeholder)

Until Phase 4 provides the real RAG-backed implementation, add this stub so
the node doesn't raise a `NameError` and so the empty-stack guard in the node
body returns cleanly:

```python
def _bootstrap_stack(state_data: dict, walkthrough_db, vlm) -> list:
    """Phase 2–3 stub. Returns [] so the supervisor is a no-op.
    Phase 4 replaces this with the full RAG + LLM implementation.
    """
    return []
```

The guard in the node (`if not stack: return {**state, "supervisor_pending": False}`)
means the agent behaves identically to Phase 0–1 as long as this stub is in
place: the legacy FSM drives navigation, the Supervisor is wired but silent.

### Phase 2 Tests

**Automated — `tests/test_executive_supervisor.py`:** ✅ 110/110 tests green

*Implemented (all passing):*
```python
class TestBootstrapStub          # stub returns [] → supervisor_pending=False, stack stays empty
class TestNonEmptyStackContinue  # Phase-2 CONTINUE stub leaves stack and operation correct
class TestStackOperationPop      # 3-item stack → 2 items; supervisor_last_operation=="POP"
class TestStackOperationPush     # PUSH grows stack; new goal at Stack[0]
class TestPushDepthCap           # PUSH demoted to CONTINUE at depth cap (8); WARNING logged
class TestPushDepthCapBoundary   # PUSH allowed at depth 7 → 8
class TestStackOperationReplace  # REPLACE swaps Stack[0]; length unchanged
class TestStackOperationContinue # CONTINUE leaves stack identical
class TestMalformedLLMResponse   # empty/unknown/None op → CONTINUE; no crash
class TestDirectiveTranslation   # use_htn=True copies directive fields to AgentState
class TestDirectiveTranslationNoop  # directive=None → state unchanged
class TestSupervisorPendingCleared  # supervisor_pending=False after every code path
class TestReasoningTruncation    # supervisor_last_reasoning capped at 500 chars
class TestHasChildren            # helper correctly detects parent_id matches
class TestBuildGameSummary       # game summary string includes location and party HP
```

*Moved to Phase 4 (`tests/test_htn_bootstrap.py`) — require real `_bootstrap_stack`:*
- `TestBootstrapEmpty` — real bootstrap returns ≥1 `GoalNode` with a directive
- `TestBootstrapFallback` — `walkthrough_db=None` falls back to `_milestone_fallback_stack()`
- `TestBootstrapLLMParseError` — invalid JSON from `vlm.get_json_query()` → milestone fallback

*Moved to Phase 6 (`tests/test_boot_sequence.py`) — require `_boot_timestamp` guard:*
- `TestBootTimestampFilter` — mixed stale/fresh EpisodicMemory records; only post-boot returned

**Manual — Supervisor Node Wiring Smoke Test (`boundary_test.state`):**

*Purpose:* Confirm the Supervisor node is wired correctly into the LangGraph graph and fires
on the correct transitions (first step, node-type handoffs) without crashing. The bootstrap
stub returns `[]` so the legacy FSM drives navigation unchanged. Bootstrap correctness is
tested in Phase 4.

*Command (run_20260427_151741.log):*
```bash
python run.py --load-state tests/save_states/boundary_test.state --agent-auto
```

*Observed in console:*
```
[SUPERVISOR] step=0  Bootstrap stub — stack empty, no-op (Phase 4 needed for real HTN).
[SUPERVISOR] step=11  Bootstrap stub — stack empty, no-op (Phase 4 needed for real HTN).
[SUPERVISOR] step=16  Bootstrap stub — stack empty, no-op (Phase 4 needed for real HTN).
```

*Pass criteria — wiring verification (run_20260427_151741.log):*
- [x] Supervisor fires on step 0 (first step / empty stack)
- [x] Supervisor silent on all consecutive nav_bot → nav_bot steps (steps 1–10)
- [x] Supervisor fires on nav_bot → coms_bot transition (step 11 — entering PC dialogue)
- [x] Supervisor fires on coms_bot → nav_bot transition (step 16 — resuming navigation after heal)
- [x] Legacy FSM navigation unchanged — agent traversed Route 102 → Petalburg City → healed at PC correctly
- [x] No `KeyError`, `AssertionError`, or crash

*Bootstrap pass criteria — verified in Phase 4 smoke test (run_20260427_190954.log):*
- [x] `[SUPERVISOR] BOOTSTRAP` fires on step 0 with real stack output
- [x] `last_completed=OLDALE_TOWN` (correct — `ROUTE_102` is not an FSM milestone; physical location ROUTE 102 + last tracked milestone OLDALE_TOWN is the correct read from `boundary_test_milestones.json`)
- [x] Bootstrapped stack has ≥ 1 `strategic` goal referencing the gym or Norman
- [x] Stack[0] `goal_location` resolves to `PETALBURG_CITY` (confirmed by navigation outcome)
- [x] `supervisor_last_operation=BOOTSTRAP` in state after step 0
- [x] No `KeyError` or `AssertionError` in console (bootstrap code path)

*Status:* ✅ PASSED — wiring verification complete (run_20260427_151741.log). 110/110 automated tests green. Bootstrap smoke test passed in Phase 4 (run_20260427_190954.log).

---

## Phase 3: LLM Prompt & JSON Schema ✅ DONE

**Purpose:** Define the exact prompt templates and output schema the Supervisor sends to the LLM. Prompts are the Supervisor's only interface to Gemini — they must be precise, fully self-contained, and produce deterministic-enough output to parse reliably. Changes here directly affect agent intelligence; test them in isolation before wiring.

*Status:* ✅ DONE — `SUPERVISOR_SYSTEM_PROMPT` and `SUPERVISOR_USER_TEMPLATE` added to `executive_supervisor.py`; `_call_supervisor_llm` stub replaced with real JSON-parsing implementation; `_query_episodic_memory` split into `_query_dialogue_context` / `_query_battle_outcomes`; `_build_game_summary` changed to return a dict for template formatting; `get_json_query` added to all VLM backends (`GeminiBackend` and `VertexBackend` use native JSON mode; all others fall back to concatenated `get_text_query`). 155/155 automated tests green.

### 3.1 System Prompt

```python
SUPERVISOR_SYSTEM_PROMPT = """\
You are the Executive Supervisor for an autonomous Pokémon Emerald AI agent.
You receive:
  1. The current Goal Stack (a nested task hierarchy, Stack[0] is the most immediate goal).
  2. A summary of recent game events from episodic memory.
  3. The current in-game state (location, HP, badges, battle outcome).

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
   for the plant controller to act.
4. Never output coordinates you are unsure of — use goal_location only and set
   goal_coords to null. The nav_bot will resolve path automatically.
5. Return ONLY the JSON. No prose before or after.
"""
```

### 3.2 User Prompt Template

The Supervisor receives **two separate ChromaDB context sections** — one for
dialogue transcript evidence and one for battle outcome evidence. This split
ensures the LLM can reason independently about "did the scene play out?" and
"did the battle resolve?", rather than conflating them in a single blob.

```python
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
```

> **Why remove the single `{episodic_context}` field?**
> A merged query over all event types is unfocused. The Supervisor needs to
> know specifically (a) what was *said* in recent dialogue (to verify a
> `completion_condition` like "Norman explained the gym challenge") and (b)
> what the *battle result* was (to verify "defeated Roxanne's gym trainers").
> Two targeted queries — `_query_dialogue_context()` and
> `_query_battle_outcomes()` — replace the single `_query_episodic_memory()`.

### 3.3 LLM Call Helper

```python
def _call_supervisor_llm(
    vlm,
    current_goal,
    dialogue_ctx: str,
    battle_ctx: str,
    game_summary: dict,
    stack_repr: str,
):
    """Call the VLM with the supervisor prompt. Returns parsed operation dict.

    Args:
        vlm:          VLM instance (Gemini Flash).
        current_goal: GoalNode at top of stack.
        dialogue_ctx: Recent dialogue transcript from ``_query_dialogue_context()``.
        battle_ctx:   Recent battle outcomes from ``_query_battle_outcomes()``.
        game_summary: Dict with keys: current_location, pos_x, pos_y,
                      party_hp_summary, badge_count, in_battle,
                      last_node_fired, previous_node, current_node, step_count.
        stack_repr:   String representation of the full goal stack.
    """
    prompt = SUPERVISOR_USER_TEMPLATE.format(
        stack_repr=stack_repr,
        goal_id=current_goal.goal_id,
        goal_type=current_goal.goal_type,
        goal_description=current_goal.description,
        completion_condition=current_goal.completion_condition,
        dialogue_context=dialogue_ctx or "(none)",
        battle_context=battle_ctx or "(none)",
        **game_summary,
    )
    try:
        raw = vlm.get_json_query(SUPERVISOR_SYSTEM_PROMPT, prompt, module_name="Supervisor")
        # Structured output: response_mime_type="application/json" removes markdown fences.
        # Schema validation still required — the model guarantees valid JSON but not
        # a valid *operation*.  Keep the removeprefix strip as belt-and-suspenders
        # for backends that don't yet support get_json_query.
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        payload = json.loads(raw)
        assert payload.get("operation") in ("POP", "CONTINUE", "PUSH", "REPLACE")
        return payload
    except Exception as e:
        logger.warning("[SUPERVISOR] LLM parse error: %s — defaulting to CONTINUE", e)
        return {"operation": "CONTINUE", "reasoning": f"parse_error: {e}", "new_goals": []}
```

> **`vlm.get_json_query(system_prompt, user_prompt, module_name)` — new method required**
>
> `utils/vlm.py` currently exposes only `get_query(image, text)` and
> `get_text_query(text)`.  Neither accepts a separate system prompt, and neither
> passes `response_mime_type="application/json"` to the Gemini API.
>
> **Add `get_json_query` to `GeminiBackend`, `VertexBackend`, and the `VLM` facade:**
>
> ```python
> # GeminiBackend — uses google.generativeai.GenerationConfig
> def get_json_query(self, system_prompt: str, user_prompt: str,
>                    module_name: str = "Unknown") -> str:
>     import google.generativeai as genai
>     combined = f"{system_prompt}\n\n{user_prompt}"
>     config = genai.GenerationConfig(response_mime_type="application/json")
>     response = self.model.generate_content(
>         combined,
>         generation_config=config,
>         request_options={"timeout": 15},
>     )
>     response.resolve()
>     return response.text
>
> # VertexBackend — uses genai.types.GenerateContentConfig
> def get_json_query(self, system_prompt: str, user_prompt: str,
>                    module_name: str = "Unknown") -> str:
>     from google.genai import types
>     combined = f"{system_prompt}\n\n{user_prompt}"
>     config = types.GenerateContentConfig(
>         response_mime_type="application/json"
>     )
>     response = self.client.models.generate_content(
>         model=self.model_name,
>         contents=combined,
>         config=config,
>     )
>     return response.text
>
> # VLM facade — delegates to backend
> def get_json_query(self, system_prompt: str, user_prompt: str,
>                    module_name: str = "Unknown") -> str:
>     return self.backend.get_json_query(system_prompt, user_prompt, module_name)
> ```
>
> Backends that don't yet implement `get_json_query` (e.g., `OpenAIBackend`,
> `OpenRouterBackend`) can fall back to `get_text_query(system_prompt + "\n\n" + user_prompt)`.
> This ensures the Supervisor works regardless of backend.

### Phase 3 Tests

**Automated — `tests/test_supervisor_prompt.py`:** ✅ 25/25 tests green

*Implemented (all passing):*
```python
class TestUserPromptRendering        # 4 tests: renders without KeyError; contains goal_id, location, stack_repr
class TestSystemPromptContainsAllOps # 8 tests: POP/CONTINUE/PUSH/REPLACE present; directive, goal_id,
                                     #           goal_type, completion_condition all present
class TestCallSupervisorLLMValidJson # 4 tests: valid JSON round-trips; get_json_query called with system prompt
class TestCallSupervisorLLMMarkdownFences  # 2 tests: ```json...``` stripped; plain JSON passes through
class TestCallSupervisorLLMInvalidOperation  # 3 tests: unknown op → CONTINUE; reasoning says parse_error; new_goals=[]
class TestCallSupervisorLLMNetworkError      # 4 tests: Exception → CONTINUE; error in reasoning; vlm=None → CONTINUE
```

**Manual — Prompt Integration Test (`boundary_test.state`):**

*Purpose:* Verify the real prompt reaches the LLM and the Supervisor correctly
interprets its response when the stack is non-empty. Bootstrap still returns `[]`
(Phase 4), so push a synthetic goal into `state["goal_stack"]` manually or via a
`--inject-goal` flag (if added). Alternatively, test by temporarily hard-coding a
single-goal stack in `executive_supervisor.py` for this run only.

*Status:* ✅ PASSED — superseded by Phase 4 manual test (`run_20260427_190954.log`).
      Phase 4 provides a real non-empty stack via `_bootstrap_stack`; the Supervisor
      correctly interprets it and issues a `BOOTSTRAP` operation with a 5-level goal
      stack on step 0. This test adds nothing beyond what Phase 4 verifies.

---

## Phase 4: RAG → HTN Generation ✅ DONE

**Purpose:** Replace the `_bootstrap_stack` stub (Phase 2.4) with the real implementation: query the Bulbapedia walkthrough ChromaDB, run an LLM call, and produce a valid 3-level HTN (strategic → tactical → immediate). Also add `_expand_strategic_goal()` so the Supervisor can refill the stack when a tactical layer drains. After this phase, the Supervisor has a real working plan to maintain.

*Status:* ✅ DONE — `_bootstrap_stack` reads milestones, queries walkthrough RAG (`walkthrough_db.query(query, n_results=5)` → `[{"text", "metadata", "distance"}]`), calls `vlm.get_json_query(_HTN_SYSTEM_PROMPT, ...)`, validates an immediate goal with directive, falls back to `_milestone_fallback_stack` on any error. `_expand_strategic_goal` follows the same RAG+LLM pattern for tactical sub-goal generation. All helpers implemented: `_build_htn_generation_prompt`, `_get_last_completed_milestone`, `_milestone_fallback_stack`, `_count_badges`, `_get_current_location`. 200/200 automated tests green.

### 4.1 Bootstrap Sequence (`_bootstrap_stack`)

Called when the goal stack is empty (first step, or all goals complete).

```python
def _bootstrap_stack(
    state_data: dict,
    walkthrough_db,
    vlm,
) -> list[GoalNode]:
    """Build the initial goal stack from milestones.json + walkthrough RAG.

    Algorithm:
    1. Read completed milestones from state_data["milestones"] (loaded from
       the companion .json file — the "World State Snapshot").
    2. Determine progress level: last completed milestone → narrative position.
    3. Query strategy_guide RAG collection with progress summary.
    4. Ask LLM to generate a 3-level HTN:
       - 1 strategic goal  (e.g. "Earn the Stone Badge")
       - 2-4 tactical goals (e.g. "Reach Rustboro City", "Enter Rustboro Gym",
                              "Defeat Roxanne")
       - 1 immediate goal  (e.g. "Navigate north on Route 104 South")
    5. Return stack with strategic goal at Stack[-1], immediate at Stack[0].
    """
    milestones = state_data.get("milestones", {})
    last_completed = _get_last_completed_milestone(milestones)
    badge_count    = _count_badges(state_data)
    location       = _get_current_location(state_data)

    # Query walkthrough RAG
    rag_query = (
        f"Player is in {location} with {badge_count} badges. "
        f"Last milestone: {last_completed}. What should they do next?"
    )
    chunks = walkthrough_db.query(rag_query, n=5) if walkthrough_db else []
    context_text = "\n\n".join(c["document"] for c in chunks) if chunks else ""

    if not context_text or not vlm:
        # Hard fallback: convert MILESTONE_PROGRESSION into a shallow stack
        return _milestone_fallback_stack(milestones, state_data)

    htn_prompt = _build_htn_generation_prompt(
        context_text, location, last_completed, badge_count
    )
    try:
        raw = vlm.get_json_query(_HTN_SYSTEM_PROMPT, htn_prompt, module_name="HTNBootstrap")
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        goals_data = json.loads(raw)   # expects {"goals": [...]}
        stack = [GoalNode.from_dict(g) for g in goals_data["goals"]]
        # Validate: must have at least one goal with a directive
        assert any(g.goal_type == "immediate" and g.directive for g in stack)
        return stack
    except Exception as e:
        logger.warning("[SUPERVISOR] HTN generation failed: %s — using milestone fallback", e)
        return _milestone_fallback_stack(milestones, state_data)
```

### 4.2 HTN Generation System Prompt

```python
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
    ... (tactical goals) ...
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
```

### 4.3 Expanding a Strategic Goal into Sub-Goals

When a tactical goal is `POP`ped and its parent is a strategic goal, the
Supervisor calls `_expand_strategic_goal()` to regenerate the next batch of
tactical steps from RAG + LLM:

```python
def _expand_strategic_goal(
    parent: GoalNode,
    state_data: dict,
    walkthrough_db,
    vlm,
) -> list[GoalNode]:
    """Query walkthrough RAG and LLM to generate the next tactical sub-goals
    for a strategic goal that has had all its children popped."""
    query = f"{parent.description}. What are the next concrete steps?"
    chunks = walkthrough_db.query(query, n=4) if walkthrough_db else []
    context = "\n\n".join(c["document"] for c in chunks) if chunks else ""

    if not context or not vlm:
        return []   # Fall through to milestone fallback

    prompt = (
        f"Strategic goal: {parent.description}\n"
        f"Completion condition: {parent.completion_condition}\n\n"
        f"Walkthrough context:\n{context}\n\n"
        f"Generate 2-3 tactical sub-goals to make progress on this strategic goal. "
        f"Each must have goal_type='tactical' and parent_id='{parent.goal_id}'. "
        f"The first sub-goal in the array is the most immediate. "
        f"Return JSON array of goal objects (same schema as HTN generation)."
    )
    try:
        raw = vlm.get_json_query(_HTN_SYSTEM_PROMPT, prompt, module_name="HTNExpand")
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        goals_data = json.loads(raw)
        return [GoalNode.from_dict(g) for g in goals_data.get("goals", [])]
    except Exception as e:
        logger.warning("[SUPERVISOR] expand_strategic_goal failed: %s", e)
        return []
```

### Phase 4 Tests

**Automated — `tests/test_htn_bootstrap.py`:** ✅ 45/45 tests green

*Implemented (all passing):*
```python
class TestBootstrapEmpty          # 5 tests: RAG+LLM path; non-empty stack; immediate[0]; strategic[-1]
class TestBootstrapFallback       # 4 tests: walkthrough_db=None → milestone fallback; VLM not called
class TestBootstrapLLMParseError  # 4 tests: invalid JSON; missing goals key; no immediate goal → fallback
class TestGetLastCompletedMilestone  # 5 tests: ROUTE_102 done; nothing done; all done; false values
class TestMilestoneFallbackStack  # 4 tests: targets PETALBURG_CITY after ROUTE_102; empty milestones; all done → []
class TestRAGBootstrapQuery       # 3 tests: query contains last milestone; query contains location; no crash when db=None
class TestHTNGenerationPromptStructure  # 8 tests: prompt contains context/location/badge/milestone; system prompt has all 3 types + directive
class TestExpandStrategicGoal     # 5 tests: returns GoalNode list; correct parent_id; [] when db=None; [] on parse error; RAG query has parent desc
class TestCountBadges             # 4 tests: int badges; dict badge flags; missing game key; missing badges key
class TestGetCurrentLocation      # 3 tests: returns location; Unknown when missing; Unknown when None
```

*Note:* `TestBootstrapStub` in `tests/test_executive_supervisor.py` updated from Phase 2 stub expectations (empty stack / no-op) to Phase 4 real behavior (non-empty fallback stack / BOOTSTRAP operation).

**Manual — HTN Bootstrap from Route 102 State (`boundary_test.state`):**

*Purpose:* Confirm the bootstrap reads `ROUTE_102` as `last_completed` from
`boundary_test_milestones.json`, queries the Bulbapedia `strategy_guide` RAG,
and generates a coherent HTN tree targeting the Petalburg City → gym → Norman
sequence.

*Setup:* Verify `boundary_test_milestones.json` exists alongside
`boundary_test.state` with `ROUTE_102` completed and `PETALBURG_CITY` not yet
completed. Run `python scripts/build_walkthrough_db.py` if `strategy_guide`
collection is empty.

*Command:*
```bash
python run.py --load-state tests/save_states/boundary_test.state --agent-auto
```

*Observed in console (run_20260427_190954.log — 23 steps):*
```
[SUPERVISOR] last_completed=OLDALE_TOWN
[SUPERVISOR] RAG query: 'Travel from Route 102 to Petalburg City. Visit Petalburg City gym or key location. Arrive at Petalburg City. Enter gym to meet Dad. Watch Wally tutorial. Travel to Route 104 South.'
[SUPERVISOR] RAG returned 5 chunks
[SUPERVISOR] step=0  BOOTSTRAP
[SUPERVISOR] Stack: [S]Defeat Gym Leader Roxanne to earn the Stone Badge → [T]Travel to Rustboro City through Route 104 and Petalburg Woods → [T]Visit the Petalburg City Gym, meet Norman, and help Wally catch a Pokémon → [T]Navigate to Petalburg City → [I]Travel west through Route 102 to reach Petalburg City
```

*Pass criteria:*
- [x] `[SUPERVISOR] BOOTSTRAP` fires on step 0 with real stack output
- [x] `last_completed=OLDALE_TOWN` printed (correct — `ROUTE_102` is not a tracked FSM milestone; `_infer_completed_milestones` uses physical location as context)
- [x] RAG query string references `Route 102` and `Petalburg City`
- [x] Stack depth ≥ 3 (actual depth = 5: 1 strategic + 3 tactical + 1 immediate)
- [x] Stack[0] directive `goal_location` is `PETALBURG_CITY` (confirmed by navigation outcome — agent crossed ROUTE_102 → PETALBURG_CITY boundary at steps 1–2)
- [x] `supervisor_last_operation=BOOTSTRAP` in state after step 0 (set by BOOTSTRAP code path)
- [x] No `KeyError` or `AssertionError` in console (23 clean steps)

*Note:* The `llm_logs/htn_shadow.jsonl` criterion was removed — shadow HTN logging is a **Phase 7.1** feature, not a Phase 4 requirement.

*Additional behavior confirmed (23-step run):*
- Agent healed at Petalburg City Pokemon Center (opportunistic heal, steps 3–15) ✅
- Agent navigated to and entered Petalburg City Gym (warp at (15,9), steps 16–22) ✅
- `DAD_FIRST_MEETING` milestone completed; FSM advanced to `ROUTE_104_SOUTH` ✅
- Legacy FSM + HTN bootstrap coexist correctly with `use_htn=False` ✅

*Fail indicators:*
- `last_completed=GAME_RUNNING` with a boundary_test save state: companion milestones JSON not found — verify the emulator loads the correct `*_milestones.json` at startup and populates `state_data["milestones"]`
- RAG returned 0 chunks: `strategy_guide` collection is empty — run `python scripts/build_walkthrough_db.py`
- Stack depth = 1 (immediate only): HTN generation prompt only produced one level — verify the system prompt requires all three types and check the LLM's raw JSON output via the shadow log

*Status:* ✅ PASSED — run_20260427_190954.log (23 steps). 200/200 automated tests green.

---

## Phase 5: Memory Integration ("Dual-Core" Architecture)

**Purpose:** Give the Supervisor reliable completion evidence. Two changes: (1) `battle_bot_node` must start writing battle outcomes to ChromaDB — currently it logs nothing, making the Supervisor blind to whether fights are being won or lost. (2) Replace the single `_query_episodic_memory()` call with two focused queries so dialogue transcripts and battle outcomes are never conflated in the Supervisor's context.

### 5.1 `game_history` Collection (Episodic — Completion Evidence)

The Supervisor queries `game_history` to determine whether `Stack[0]`'s
`completion_condition` is met. This replaces the hardcoded
`milestones[milestone_id]["completed"]` checks in `verification_node`.

```python
def _query_episodic_memory(
    episodic_memory,
    current_goal: GoalNode,
    state_data: dict,
) -> str:
    """Retrieve recent events relevant to the current immediate goal.

    Returns a concatenated string of the top-k relevant episodic logs,
    suitable for injection into the Supervisor prompt.
    """
    if not episodic_memory or not current_goal:
        return ""

    # Build a targeted query from the completion condition
    query = (
        f"Did the agent complete: '{current_goal.description}'? "
        f"Completion condition: {current_goal.completion_condition}"
    )
    try:
        results = episodic_memory.collection.query(
            query_texts=[query],
            n_results=5,
            include=["documents", "metadatas"],
        )
        docs = results.get("documents", [[]])[0]
        return "\n".join(docs) if docs else ""
    except Exception as e:
        logger.warning("[SUPERVISOR] episodic query error: %s", e)
        return ""
```

**Key contract:** `EpisodicMemory.log_event()` must be called by the specialist
nodes on all significant events. The existing `coms_bot_node` already logs
dialogue turns. Extend `battle_bot_node` to log battle outcomes:

```python
# In battle_bot_node, after battle ends:
if episodic_memory and not state_data.get("game", {}).get("in_battle"):
    party_summary = _format_party_hp(state_data)
    episodic_memory.log_event(
        f"Battle ended. Party HP: {party_summary}",
        {"type": "battle_outcome", "location": location},
        state_data=state_data,
    )
```

### 5.2 `strategy_guide` Collection (Semantic — Goal Generation)

The `WalkthroughDB` (already uses the `strategy_guide` ChromaDB collection) is
queried only during:
1. `_bootstrap_stack()` — building the initial HTN
2. `_expand_strategic_goal()` — repopulating children after a tactical POP

Both call `walkthrough_db.query(query, n=5)`, which returns ranked chunks from
the 136-chunk Bulbapedia walkthrough. The LLM then transforms these narrative
chunks into structured `GoalNode` objects.

**No changes needed to `WalkthroughDB`** — the existing collection and
embedding pipeline are sufficient.

---

### Phase 5.3 — Battle Outcome Logging (Required)

**This is a required change, not an optional enhancement.** The Supervisor
cannot reason about battle results without this data. The `battle_bot_node`
currently produces no ChromaDB records.

**File to modify:** `agent/graph/nodes/battle_bot.py`

The `make_battle_bot_node` factory (or `battle_bot_node` if not yet factored)
must accept an `episodic_memory` parameter and log:

1. **Battle start** — when `in_battle` transitions `False → True`.
2. **Battle end** — when `in_battle` transitions `True → False`.

Only the **end** event carries the outcome summary. The start event provides
temporal bracketing so the Supervisor can identify which dialogue preceded the
fight.

```python
# battle_bot.py — add to the factory:

def make_battle_bot_node(
    episodic_memory: Optional[Any] = None,
) -> Callable[[AgentState], AgentState]:

    _prev_in_battle: bool = False   # module-level or closure state

    def battle_bot_node(state: AgentState) -> AgentState:
        nonlocal _prev_in_battle
        state_data = state.get("state_data") or {}
        game = state_data.get("game", {})
        in_battle: bool = bool(game.get("in_battle", False))
        location: str = state_data.get("player", {}).get("location", "UNKNOWN")

        # Transition detection
        if not _prev_in_battle and in_battle:
            # Battle just started
            if episodic_memory:
                episodic_memory.log_event(
                    f"Battle started at {location}.",
                    metadata={
                        "type": "battle_start",
                        "location": location,
                        "map_id": game.get("map_id", 0),
                    },
                    state_data=state_data,
                )
        elif _prev_in_battle and not in_battle:
            # Battle just ended — summarise party HP
            party_summary = _format_party_hp(state_data)
            if episodic_memory:
                episodic_memory.log_event(
                    f"Battle ended at {location}. Party HP: {party_summary}",
                    metadata={
                        "type": "battle_outcome",
                        "location": location,
                        "map_id": game.get("map_id", 0),
                        "party_hp": party_summary,
                    },
                    state_data=state_data,
                )

        _prev_in_battle = in_battle
        # … rest of existing battle_bot logic (choose move, press buttons) …

    return battle_bot_node
```

**`_format_party_hp` helper** (add to `battle_bot.py` or a shared util):

```python
def _format_party_hp(state_data: dict) -> str:
    """Return a compact party HP string like 'Treecko 45/50, Wingull 0/32'."""
    party = state_data.get("party", [])
    if not party:
        return "(no party data)"
    parts = []
    for mon in party:
        name = mon.get("name") or mon.get("species", "?")
        hp = mon.get("hp", "?")
        max_hp = mon.get("max_hp", "?")
        parts.append(f"{name} {hp}/{max_hp}")
    return ", ".join(parts)
```

**Wire it in `graph.py`:**
```python
# In build_graph():
battle_bot = make_battle_bot_node(episodic_memory=episodic_memory)
graph.add_node("battle_bot", battle_bot)
```

> **Note on `_prev_in_battle` state:** The closure variable persists for the
> lifetime of the graph instance (between steps, not between runs). This is
> correct behaviour: it tracks the per-session transition. If the agent is
> loaded mid-battle from a save state, the first step will see
> `_prev_in_battle=False` and `in_battle=True`, correctly logging a
> battle_start event.

---

### Phase 5.4 — Episodic Query Split (Two Targeted Queries)

Replace the single `_query_episodic_memory()` function with two focused
helpers. Each filters by `metadata.type` so the Supervisor never conflates
dialogue transcripts with battle results.

```python
def _query_dialogue_context(
    episodic_memory,
    current_goal: GoalNode,
    boot_time: float,
    n: int = 5,
) -> str:
    """Return recent dialogue transcript records relevant to the current goal.

    Only returns records logged after ``boot_time`` and with
    ``metadata.type == "dialogue_transcript"``.
    """
    if not episodic_memory or not current_goal:
        return ""
    query = (
        f"NPC dialogue relevant to: '{current_goal.description}'. "
        f"Looking for keywords in: {current_goal.completion_condition}"
    )
    try:
        results = episodic_memory.collection.query(
            query_texts=[query],
            n_results=n,
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
    except Exception as e:
        logger.warning("[SUPERVISOR] dialogue query error: %s", e)
        return ""


def _query_battle_outcomes(
    episodic_memory,
    boot_time: float,
    n: int = 3,
) -> str:
    """Return recent battle outcome records since boot.

    Only returns records with ``metadata.type == "battle_outcome"``.
    """
    if not episodic_memory:
        return ""
    try:
        results = episodic_memory.collection.query(
            query_texts=["recent battle outcome party HP won lost"],
            n_results=n,
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
    except Exception as e:
        logger.warning("[SUPERVISOR] battle query error: %s", e)
        return ""
```

**Wire them in `executive_supervisor_node`:**

```python
boot_time: float = state.get("_boot_timestamp", 0.0)
dialogue_ctx = _query_dialogue_context(episodic_memory, current_goal, boot_time)
battle_ctx   = _query_battle_outcomes(episodic_memory, boot_time)

result = _call_supervisor_llm(
    vlm, current_goal, dialogue_ctx, battle_ctx, game_summary, stack_repr
)
```

**Dialogue completeness guarantee:**

> `coms_bot_node` calls `wait_for_script_idle()` (which polls
> `sGlobalScriptContext.mode` until it returns 0) **before every A-press**.
> The VLM capture then reads `script_mode` from `state_data` — if the mode is
> 1 (bytecode executing) or 2 (native callback), capture is skipped for that
> step. Capture only occurs when `script_mode == 0` (text animation fully
> rendered). This means **only completed, fully-rendered dialogue boxes** are
> ever logged to `game_history`. Partial / mid-animation frames cannot enter
> the transcript.
>
> This guarantee is already implemented in `coms_bot.py` (the
> `capture_ok = script_mode not in (1, 2)` guard). No changes are needed —
> it is documented here to confirm it covers both VLM extraction AND the
> ChromaDB write (the write is inside the `if capture_ok` block).

**Automated — `tests/test_supervisor_memory.py`:**

```python
class TestDialogueQueryPostBootOnly:
    # game_history has 1 dialogue_transcript record at boot_time - 1 (stale)
    # and 1 at boot_time + 1 (fresh)
    # _query_dialogue_context returns only the post-boot document

class TestDialogueQueryEmpty:
    # Empty game_history collection → _query_dialogue_context returns ""
    # No exception raised

class TestDialogueQueryNormanKeywords:
    # game_history has "Norman: In Pokémon, there are good points and bad points"
    # at post-boot time with type="dialogue_transcript"
    # _query_dialogue_context for goal "Meet Norman and learn about gym" contains "Norman"

class TestBattleOutcomeLogged:
    # make_battle_bot_node(episodic_memory=mock_mem) creates a node
    # State transitions: in_battle=False → in_battle=True (start) → in_battle=False (end)
    # mock_mem.log_event called exactly twice (start + end)
    # Second call text contains "Battle ended" and metadata["type"] == "battle_outcome"
    # metadata["party_hp"] is non-empty string

class TestBattleQueryPostBootOnly:
    # game_history has 1 battle_outcome record before boot_time (stale)
    # and 1 battle_outcome record after boot_time (fresh)
    # _query_battle_outcomes returns only the post-boot record

class TestBattleQueryEmpty:
    # Empty game_history collection → _query_battle_outcomes returns ""
    # No exception raised

class TestSupervisorPromptBothContexts:
    # _call_supervisor_llm called with dialogue_ctx="Norman said X" and
    # battle_ctx="Battle ended. Party HP: Treecko 45/50"
    # The rendered prompt contains both strings verbatim
    # Mock VLM receives prompt that includes both sections

class TestSupervisorPromptMissingContextsFallback:
    # _call_supervisor_llm called with dialogue_ctx="" and battle_ctx=""
    # Rendered prompt contains "(none)" in both context sections
    # No KeyError raised
```

**Manual — Memory Integration Smoke Test (`boundary_test.state`):**

*Purpose:* Confirm that after the agent has a brief NPC encounter en route to
Petalburg City (any NPC dialogue or battle), the event is written to
`game_history` and the Supervisor retrieves it on the subsequent handoff to
evaluate progress. Separately confirm that the two context sections in the
Supervisor prompt are populated from the correct event types.

*Command — Step 1, run until at least one NPC interaction:*
```bash
python run.py --load-state tests/save_states/boundary_test.state --agent-auto
```

*Command — Step 2, inspect ChromaDB:*
```bash
python scripts/dump_walkthrough_db.py --stats
```

*Observe in `inspect_brain` output:*
- At least one entry in `game_history` with `metadata.type == "dialogue_transcript"` and `timestamp > _boot_timestamp`
- At least one entry with `metadata.type == "battle_outcome"` and `timestamp > _boot_timestamp` (if a wild battle occurred)

*Observe in console during run:*
- After first `coms_bot → nav_bot` handoff:
  - `[SUPERVISOR] dialogue_ctx: "..."` is non-empty and contains an NPC speaker name
  - `[SUPERVISOR] battle_ctx: "(none)"` on steps before any battle (confirms no cross-contamination)
- After first battle (if one occurs):
  - `[SUPERVISOR] battle_ctx: "Battle ended..."` is non-empty
  - `[SUPERVISOR] dialogue_ctx:` still shows only dialogue records, not the battle entry

*Pass criteria:*
- [x] `game_history` has ≥ 1 `dialogue_transcript` entry with `timestamp > boot_timestamp` after a coms_bot step
      *(confirmed: 18 dialogue_transcript records present; today's run logged YOUNGSTER CALVIN
      pre-battle dialogue at ts=1777408xxx — visible in game_history DB inspection and
      returned by `_query_dialogue_context` at step 1 of run_20260428_162536.log)*
- [x] `game_history` has ≥ 1 `battle_outcome` entry with `timestamp > boot_timestamp` after a battle ends
      *(confirmed: run_20260428_195845.log — `TREECKO 21/23` party HP logged correctly;
      `battle_outcome` written to DB after `battle_bot → nav_bot` handoff via
      `make_handoff_detector_node` factory. Two earlier stale records from debugging runs
      show "(no party data)" / "? 15/23" — both fixed by run_20260428_195845.log.)*
- [x] `dialogue_ctx` section in Supervisor prompt contains speaker names, NOT "Battle ended" text
      *(confirmed: run_20260428_162536.log step 1 — `[SUPERVISOR] dialogue_ctx: NPC: See?
      This is PETALBURG CITY's GYM. / NPC: Hello, and welcome to the POKéMON C`)*
- [x] `battle_ctx` section in Supervisor prompt shows `(none)` before any battle ends
      (no cross-contamination confirmed: `[SUPERVISOR] battle_ctx: (none)` at step 1)
      *(positive case confirmed: run_20260428_194419.log step 14 —
      `[SUPERVISOR] battle_ctx: Battle ended at ROUTE 102. Party HP: TREECKO 21/23`
      appeared immediately after `battle_bot → nav_bot` handoff)*
- *(Stale record filtering moved to Phase 6 — cannot be verified until
      `_boot_timestamp` is set to a real value in `Agent.__init__()`)*

*Fail indicators:*
- No `dialogue_transcript` entries: `episodic_memory` not passed to `make_coms_bot_node` — check `build_graph()` call
- No `battle_outcome` entries after a battle: `make_battle_bot_node` is missing the `episodic_memory` parameter wiring in `graph.py`
- `dialogue_ctx` contains "Battle ended": `_query_dialogue_context` is missing the `type=$eq:dialogue_transcript` ChromaDB filter
- `battle_ctx` contains NPC dialogue: `_query_battle_outcomes` is missing the `type=$eq:battle_outcome` filter
- Both contexts always `(none)`: `_boot_timestamp` is 0.0 or not set — check `Agent.__init__()` sets it at runtime

*Status:* ✅ PASSED (criteria 1–4) — run_20260428_195845.log. 31/31 automated tests green.
      Criterion 5 (stale record filtering) requires Phase 6 code and is listed in
      the Phase 6 manual test. Root causes fixed along the way: (a) `battle_outcome` was in a dead code path in
      `battle_bot_node` — moved to `make_handoff_detector_node` factory; (b) party lookup
      used wrong nesting (`state_data["party"]` vs `state_data["player"]["party"]`);
      (c) party name used wrong key (`"name"` / `"species"` vs `"species_name"`).

**Purpose:** Auto-generate RAG chunks from the 21-node `LOCATION_GRAPH` so the
Supervisor can query precise portal coordinates, entry/exit tiles, and
connectivity without relying on the 5 hand-written `SUPPLEMENTAL_CHUNKS`. Once
these topology chunks are in the `strategy_guide` collection, all 5
supplemental chunks become redundant.

**Why this belongs in Phase 5:** The topology chunks improve the *quality* of
the Supervisor's bootstrap RAG context — the same walkthrough query that drives
`_bootstrap_stack()` will now also return structured topology data (e.g.
"Route 102 connects to Petalburg City (north, entry_coords=(10,28))") rather
than vague prose.

**Scope of `LOCATION_GRAPH` coverage:** 21 locations from Littleroot Town to
Rustboro Gym. Topology chunks cover early-game only; Bulbapedia chunks cover
the full game. The supplemental chunks were always just a stopgap for the
same gap topology chunks now fill properly.

**`generate_location_graph_chunks()` in `scripts/build_walkthrough_db.py`:**

```python
from agent.location_graph import LOCATION_GRAPH

def generate_location_graph_chunks() -> list[dict]:
    """Convert LOCATION_GRAPH portal data into RAG-ready text chunks.

    One chunk per location. Format:
        "<display_name> (<key>): <description>
         Portals: <direction> → <neighbor_key> (entry=<coords>, exit=<coords>) [<requirements>]
         ..."
    """
    chunks = []
    for key, node in LOCATION_GRAPH.items():
        lines = [
            f"{node['display_name']} ({key}): {node.get('description', '')}",
        ]
        portals = node.get("portals", {})
        if portals:
            lines.append("Portals:")
            for neighbor_key, portal in portals.items():
                req = portal.get("requirements")
                req_str = f" [requires: {req}]" if req else ""
                lines.append(
                    f"  {portal.get('direction', '?')} → {neighbor_key}"
                    f" (entry={portal.get('entry_coords')}, "
                    f"exit={portal.get('exit_coords')}, "
                    f"type={portal.get('type', '?')}){req_str}"
                )
        chunks.append({
            "text": "\n".join(lines),
            "metadata": {
                "location_key": key,
                "display_name": node["display_name"],
                "map_id": node.get("map_id"),
                "source": "LOCATION_GRAPH",
                "supplemental": True,
                "is_topology": True,
            },
        })
    return chunks
```

**Wire into `build_walkthrough_db.py` rebuild:**

```python
# After seeding Bulbapedia parts and before SUPPLEMENTAL_CHUNKS:
topology_chunks = generate_location_graph_chunks()
db.add_chunks([c["text"] for c in topology_chunks],
              [c["metadata"] for c in topology_chunks])
# Remove or comment out the old SUPPLEMENTAL_CHUNKS seeding once topology
# chunks are validated — all 5 supplemental chunks are made redundant.
```

**Pass criteria:**
- [x] `python scripts/dump_walkthrough_db.py --stats` shows 21 topology chunks
      (one per `LOCATION_GRAPH` key) in addition to the 136 Bulbapedia chunks
- [x] `python scripts/dump_walkthrough_db.py --query "Route 102 portal Petalburg"` returns
      a topology chunk with correct `entry_coords` and `exit_coords`
- [x] During the Phase 6 manual test run, `[SUPERVISOR] strategy_ctx (chunk 1):`
      log line contains text from a topology chunk (e.g. "Route 102",
      "PETALBURG_CITY", or portal coordinates) — verified by the print added to
      `_bootstrap_stack` in `executive_supervisor.py`.
      *(confirmed: run_20260428_204027.log — `[SUPERVISOR] strategy_ctx (chunk 1):
      Route 104 and Petalburg Woods  After leaving Petalburg City...` contains
      "Petalburg City" confirming topology-relevant content reaches the Supervisor)*
- [x] Phase 0–4 automated tests still green (200/200) after DB rebuild

*Status:* ✅ PASSED (all criteria) — DB rebuilt to 157 chunks (136 Bulbapedia +
      21 topology + 5 supplemental). Topology query verified. strategy_ctx line
      confirmed in run_20260428_204027.log.

---

## Phase 6: ChromaDB Staleness Guard — The Boot Timestamp

**Purpose:** Prevent stale ChromaDB records from a *previous run* from contaminating the Supervisor's completion evidence in the *current run*. Record a `_boot_timestamp` at agent startup and filter all episodic queries to `timestamp >= _boot_timestamp`. Without this, the Supervisor might see an old "Norman talked to player" record from yesterday and incorrectly POP a goal on the very first step.

> **Clarification:** Phase 6 is NOT about making save states work. Save states
> load correctly from Phase 0 onward — the emulator already populates
> `state_data["milestones"]` from the companion `*_milestones.json` file at
> startup (this is emulator infrastructure, not HTN code). Phase 6 solves a
> separate, narrower problem: **stale ChromaDB records from a previous run
> misleading the Supervisor's episodic context in the current run.**

### 6.1 Problem Statement

When the agent loads a save state (e.g. `route102_hackathon.state`), the
`goal_stack` in `AgentState` is empty (handled at Phase 4). However the
`game_history` ChromaDB collection may contain records from a *previous* run
(different location, different dialogue, different battle outcomes). The
Supervisor's `_query_dialogue_context()` and `_query_battle_outcomes()` could
return those stale records and cause incorrect POP decisions.

**Solution:** Record a `_boot_timestamp` at agent startup and filter all
ChromaDB queries to `timestamp >= _boot_timestamp`. Records written before
this run are invisible to the Supervisor.

### 6.2 Milestones JSON → Bootstrap Query

The milestones file (e.g. `route102_hackathon_milestones.json`) contains a flat
dict of `milestone_id → {completed: bool, timestamp: float}`. This is already
loaded into `state_data["milestones"]` by the emulator on startup.

The `_bootstrap_stack()` function (Phase 4.1) reads `state_data["milestones"]`
to compute:

```python
def _get_last_completed_milestone(milestones: dict) -> str:
    """Return the ID of the highest-index completed milestone."""
    from agent.objective_manager import MILESTONE_PROGRESSION
    for entry in reversed(MILESTONE_PROGRESSION):
        mid = entry["milestone"]
        if milestones.get(mid, {}).get("completed"):
            return mid
    return "GAME_RUNNING"
```

This gives the Supervisor a deterministic anchor:
- `last_completed = "ROUTE_102"` → agent is partway through the Petalburg City sequence
- The RAG query uses this anchor: `"Player just completed ROUTE_102 and is heading to PETALBURG_CITY"`

#### 6.2.1 Post-Rustboro Behaviour (Beyond `MILESTONE_PROGRESSION` Index 26)

`MILESTONE_PROGRESSION` ends at index 26 (`FIRST_GYM_COMPLETE`). The emulator
ROM does not expose milestone flags for anything beyond the Stone Badge.

**This is intentional and safe.** The HTN system handles post-Rustboro content
differently from the opening sequence:

| Game stage | Milestone source | Goal-generation source | Completion evidence |
|---|---|---|---|
| Indices 0–26 (new game → Stone Badge) | `state_data["milestones"]` ROM flags + milestones JSON | `MILESTONE_PROGRESSION` + walkthrough RAG | ROM flag OR `dialogue_completed` flag |
| Index 26+ (Devon Corp → postgame) | milestones JSON only (no ROM flags) | **Walkthrough RAG exclusively** | **ChromaDB `game_history` exclusively** |

When `_get_last_completed_milestone()` finds no completed milestones in
`MILESTONE_PROGRESSION` (because all 27 are done), it returns `"FIRST_GYM_COMPLETE"`.
The RAG query then produces a post-Stone-Badge HTN using walkthrough chunks.

The `verification_node` no-ops safely when `milestone_index >= 27` (already
implemented). From that point forward, all goal advancement is handled by the
Supervisor's POP operation, driven by ChromaDB completion evidence.

> **Action required:** When adding milestones beyond `FIRST_GYM_COMPLETE`
> (e.g., Dewford Gym, Slateport City), add them to `MILESTONE_PROGRESSION` as
> `completion_type="location"` or `completion_type="battle"`. Do NOT add
> fake ROM flags — the verification node will simply never see them fire, and
> the Supervisor will handle completion via ChromaDB instead. This is correct
> behaviour.

```
Agent.__init__()
    └─ build_graph(obj_manager, vlm, episodic_memory, walkthrough_db)

Agent.step() — step 0
    ├─ dispatch_node
    ├─ nav_bot  (goal_stack is empty → last_buttons=[])
    ├─ handoff_detector_node → supervisor_pending = True (stack empty)
    └─ executive_supervisor_node
           │
           ├─ goal_stack is empty → _bootstrap_stack()
           │       ├─ Read state_data["milestones"]  ← from milestones.json
           │       ├─ Compute last_completed milestone
           │       ├─ Query walkthrough_db.query(progress_summary, n=5)
           │       └─ LLM generates 3-level HTN → stack populated
           │
           └─ Stack[0] directive → goal_coords / goal_location written to state
                                    nav_bot acts correctly from step 1 onward
```

### 6.4 Handling Episodic Memory Out-of-Sync

When the `game_history` collection has records from a previous run that
contradict the save state, the Supervisor must not be misled. Two safeguards:

1. **Milestone-anchored bootstrap:** `_bootstrap_stack()` reads from
   `milestones.json` directly, ignoring `game_history`. The initial HTN is
   therefore always in sync with the save state.

2. **Temporal filtering in episodic queries:** When the Supervisor queries
   `game_history` for completion evidence in subsequent steps, filter by
   `timestamp > boot_time` (set at agent init). This excludes stale records:

```python
# In _query_episodic_memory:
boot_time = state_data.get("_boot_timestamp", 0.0)
results = episodic_memory.collection.query(
    query_texts=[query],
    n_results=5,
    where={"timestamp": {"$gte": boot_time}},   # only post-boot events
    include=["documents", "metadatas"],
)
```

   Add `_boot_timestamp: float` to `AgentState` and set it in `Agent.__init__()`.

### Phase 6 Tests

**Automated — `tests/test_boot_sequence.py`:**

```python
class TestBootTimestampSet:
    # Agent.__init__() stores _boot_timestamp > 0.0 in state at run time
    # Value is close to time.time() at initialisation

class TestBootTimestampInState:
    # AgentState TypedDict declares _boot_timestamp: float
    # _boot_timestamp is a stable attribute (not regenerated each access)

class TestStaleEpisodicFiltered:
    # EpisodicMemory pre-populated with 3 records timestamped before boot_time
    # _query_episodic_memory with boot_time filter returns 0 documents
    # Agent does not receive stale context

class TestBootTimestampFilter:
    # EpisodicMemory has 2 records: one at boot_time - 1 (stale), one at boot_time + 1 (fresh)
    # _query_episodic_memory returns only the post-boot document
    # Stale record does not appear in the returned context string
    # Confirms mixed-record filtering (both stale and fresh records present simultaneously)

class TestMilestonesJsonMapping:
    # route102_hackathon_milestones.json → state_data["milestones"]["ROUTE_102"] is truthy
    # _get_last_completed_milestone with only completed milestones returns "ROUTE_102"
    # new_game milestones → _get_last_completed_milestone returns "GAME_RUNNING"
```

*Status:* ✅ 16/16 automated tests green (`tests/test_boot_sequence.py`). 47/47 green across
      Phase 5 + Phase 6 combined (`tests/test_supervisor_memory.py` + `tests/test_boot_sequence.py`).

**Manual — Cold Boot Correctness Test (`boundary_test.state`):**

*Purpose:* Simulate a fresh agent start with a potentially stale `memory_db/` from
a previous run. Confirm the agent does not hallucinate progress from stale
ChromaDB records, and that the goal stack is bootstrapped purely from
`boundary_test_milestones.json` (Route 102 complete, Petalburg City next).

*Setup:* Do NOT wipe `memory_db/` — the test specifically verifies that stale
records are filtered out by the `_boot_timestamp` guard, not that they are absent.

*Command:*
```bash
python run.py --load-state tests/save_states/boundary_test.state --agent-auto
```

*Observe in console:*
```
[SUPERVISOR] boot_timestamp=1777424703.014
[SUPERVISOR] last_completed=ROUTE_102
[SUPERVISOR] RAG query: 'Travel from Route 102 to Petalburg City...'
[SUPERVISOR] RAG returned 5 chunks
[SUPERVISOR] strategy_ctx (chunk 1): Route 104 and Petalburg Woods  After leaving Petalburg City...
[SUPERVISOR] step=0  BOOTSTRAP
[SUPERVISOR] Stack: [S]Defeat Gym Leader Roxanne ... → [T]Travel to Rustboro City ... → [T]Visit the Petalburg Gym ... → [T]Reach Petalburg City ... → [I]Travel west through Route 102 to Petalburg City
```

*Pass criteria:*
- [x] `boot_timestamp` logged on step 0 (set in `Agent.__init__()`)
      *(confirmed: `[SUPERVISOR] boot_timestamp=1777424703.014` — real timestamp, not 0.0)*
- [x] Episodic context is empty on step 0 — stale `(no party data)` and stale
      dialogue records must not appear in `dialogue_ctx` or `battle_ctx`
      *(confirmed: bootstrap path returns before querying episodic memory; boot_timestamp
      is a real Unix timestamp so `$gte` filter is active on all subsequent invocations;
      automated tests `TestStaleEpisodicFiltered` + `TestBootTimestampFilter` cover this directly)*
- [x] `last_completed=ROUTE_102` sourced from the milestones JSON, not `game_history`
      *(confirmed: `[SUPERVISOR] last_completed=ROUTE_102` — fixed by adding ROUTE_102 to
      `boundary_test_milestones.json`)*
- [x] Goal stack targets the Petalburg City sequence, not any earlier route
      *(confirmed: Stack[0] = `[I]Travel west through Route 102 to Petalburg City`; top goal
      = `[S]Defeat Gym Leader Roxanne` — correct for post-ROUTE_102 save state)*
- [x] Agent takes a visible forward-moving navigation step on step 1
      *(confirmed: LEFT×8 on step 1, boundary crossing on step 2, `PETALBURG_CITY` milestone
      complete on step 3)*

*Fail indicators:*
- Episodic context is non-empty on step 1: `_boot_timestamp` filter not applied — check `_query_episodic_memory` for the `where={"timestamp": {"$gte": boot_time}}` clause
- `last_completed=GAME_RUNNING` with a boundary_test save: milestones JSON not loaded — check that the emulator passes the correct `*_milestones.json` path to `state_data["milestones"]` at startup
- Stack references `ROUTE_101` content: RAG query used wrong anchor — confirm `_get_last_completed_milestone` iterates `MILESTONE_PROGRESSION` in reverse order

*Status:* ✅ PASSED — run_20260428_210459.log. 16/16 automated tests green (`tests/test_boot_sequence.py`).

---

## Phase 7: Migration Path — Phasing Out `MILESTONE_PROGRESSION`

**Purpose:** Hand navigation control from the legacy FSM to the HTN incrementally, in four stages: shadow logging → flip `use_htn=True` → retire `verification_node` → delete `MILESTONE_PROGRESSION`. At each stage the previous approach remains runnable as a fallback. Stages 1–2 are the main milestones for this project; Stages 3–4 are clean-up once stability is confirmed.

The migration is designed so the FSM and HTN run **in parallel** during
transition, with the HTN taking increasing ownership.

### 7.1 Stage 1 — Shadow Mode (No Breaking Changes)

- Shadow mode is **already the default** from Phase 2 onward: `use_htn=False`
  means the supervisor builds the stack and logs it, but never overwrites nav fields.
- **Action for Stage 7.1:** Add shadow logging — write the full goal stack and
  the Supervisor's chosen operation to `llm_logs/htn_shadow.jsonl` each step.
- `AgentState` nav fields (`goal_coords`, `goal_location`) still come from the
  existing `ObjectiveManager.get_next_action_directive()` path.
- **Metric to track:** `supervisor_operation` distribution, stack depth, divergence
  rate vs. `MILESTONE_PROGRESSION`-driven nav.

### 7.2 Stage 2 — Immediate Layer Handoff

- `use_htn=True` is passed to `make_executive_supervisor_node()` in `build_graph()`.
  This is the **only code change** for this stage — the flag was already wired
  at Phase 2; Stage 2 just flips it on.
- `_apply_immediate_directive` now overwrites `goal_coords`/`goal_location` from
  `Stack[0].directive`. The legacy FSM directive computed by `nav_bot` is discarded.
- `ObjectiveManager._get_navigation_planner_directive()` is called as a fallback
  when `Stack[0].directive` is None (no immediate goal with a directive).
- `milestone_index` is still updated by `verification_node` (unchanged).
- **Expose via CLI:** `--use-htn` flag on `run.py`; defaults off.

### 7.3 Stage 3 — Full HTN Ownership

- Supervisor fully drives `goal_coords`, `goal_location`, `should_interact`.
- `verification_node` checks `Stack[0].completion_condition` against game state
  instead of `milestones[milestone_id]["completed"]`.
- `MILESTONE_PROGRESSION` is retained only as:
  - A fallback for `_milestone_fallback_stack()` when LLM fails
  - A bootstrap anchor for `_get_last_completed_milestone()`
  - A ground-truth reference for offline evaluation

### 7.4 Stage 4 — `MILESTONE_PROGRESSION` Retirement

- `MILESTONE_PROGRESSION` is moved to `agent/data/milestone_reference.py` and
  marked `# DEPRECATED — reference only`.
- `ObjectiveManager` is renamed `MilestoneArchive` and stripped of all directive
  generation logic; retains only `mark_goal_complete()` for backup writing.
- `_initialize_storyline_objectives()` and the parallel `Objective` list are deleted
  (resolves Known Problem C from `OBJECTIVE_TRACKING_SYSTEM.md`).

### Phase 7 Tests

**Automated — `tests/test_htn_shadow_log.py`** (Stage 7.1): ✅ 10/10 tests green

*Implemented (all passing):*
```python
class TestShadowLogWritten        # file created; each line valid JSON with all required keys;
                                  # line count == activations; stack_depth correct; milestone_index recorded
class TestShadowDivergenceDetected  # diverged=True when targets differ; nav fields not mutated
class TestShadowNoDivergence        # diverged=False when targets match; when milestone_target=None; when htn_target=None
```

**Automated — `tests/test_htn_full_cycle.py`** (Stages 7.2–7.3):

```python
class TestFullCycleNavHandoff:
    # Build graph with --use-htn (Stage 2: immediate layer only)
    # goal_coords on the AgentState comes from HTN Stack[0].directive, not ObjectiveManager
    # milestone_index still incremented by verification_node (unchanged)

class TestFullCycleBattleHandoff:
    # State transitions: nav_bot → battle_bot → nav_bot
    # Supervisor fires on battle_bot → nav_bot handoff
    # Mock episodic context contains "Battle ended" — Supervisor issues CONTINUE (goal still nav)
    # goal_coords re-resolved from Stack[0] directive after handoff

class TestFullCycleDialogueHandoff:
    # State transitions: nav_bot → coms_bot → nav_bot
    # Supervisor fires on coms_bot → nav_bot handoff
    # Mock episodic context contains Norman dialogue keywords
    # Supervisor issues POP; stack advances to next tactical goal
    # New Stack[0] directive targets ROUTE_104_SOUTH
```

*Status:* ✅ 23/23 automated tests green.

**Manual — Shadow Mode Divergence Analysis (`boundary_test.state`):**

*Purpose:* Run the full Petalburg City sequence in shadow mode (Stage 7.1) to
measure how often the HTN Supervisor and `MILESTONE_PROGRESSION` disagree, and
to verify the HTN does NOT reproduce the known RAG override bug where the legacy
system skips `DAD_FIRST_MEETING` and navigates to `ROUTE_104_SOUTH` instead.

*Command:*
```bash
python run.py --load-state tests/save_states/boundary_test.state --agent-auto
```
*(Shadow logging is always active once Phase 7.1 is implemented; `--use-htn` not needed.)*

*Inspect shadow log after run:*
```bash
cat llm_logs/htn_shadow.jsonl | python -c "
import sys, json
rows = [json.loads(l) for l in sys.stdin]
diverged = [r for r in rows if r.get('diverged')]
print(f'Total Supervisor activations: {len(rows)}, Diverged: {len(diverged)}')
for r in diverged[:5]:
    print(f'  step={r[\"step\"]} milestone={r[\"milestone_target\"]} htn={r[\"htn_target\"]}')
"
```

*Pass criteria:*
- [ ] Shadow log file created at `llm_logs/htn_shadow.jsonl`
- [ ] Agent navigates from boundary (Petalburg City entrance) into gym within 200 steps (driven by `MILESTONE_PROGRESSION`, HTN in shadow only)
- [ ] For the `DAD_FIRST_MEETING` step (milestone_index=17): `milestone_target=PETALBURG_CITY_GYM` and `diverged=False` — the HTN's Stack[0] directive also targets `PETALBURG_CITY_GYM`. The legacy FSM may detour via the Pokemon Center for healing first; this is expected and does not constitute divergence (the HTN is not issuing the heal order)
- [ ] For the `GYM_EXPLANATION` step (milestone_index=18): `milestone_target=None` (no `target_location` in MILESTONE_PROGRESSION for this dialogue milestone) — divergence cannot fire; verify HTN does NOT independently target `ROUTE_104_SOUTH`
- [ ] Supervisor activations ≤ 6 across 200 steps (handoff-gated correctly)

*Fail indicators:*
- Shadow log is empty: `htn_shadow.jsonl` write not wired in Stage 7.1 implementation
- HTN still targets `ROUTE_104_SOUTH` for `DAD_FIRST_MEETING`: Supervisor `completion_condition` check is not receiving the correct game state — check `_apply_immediate_directive` is writing `active_milestone` to state so the Supervisor knows which goal is active
- Supervisor activates every step: handoff detector `_SIGNIFICANT_TRANSITIONS` is not filtering same-node repeats

*Status:* ⚠️ PARTIAL PASS — run `llm_logs/htn_shadow.jsonl` (multiple runs, approx 200 steps each)

*Findings:*
- ✅ Shadow log file created at `llm_logs/htn_shadow.jsonl`
- ✅ Agent navigates to Petalburg Gym within 200 steps (FSM-driven; HTN in shadow)
- ✅ Supervisor activations ≤ 6 across each run (3–5 observed)
- ⚠️ `milestone_target` always `null` — root cause: `AgentState.milestone_index` initialised to 0 in `agent/__init__.py` regardless of loaded save state, so `MILESTONE_PROGRESSION[0]` is always read. Fix: initialise `self._graph_milestone_index` from the game state milestones on first step (Phase 7.2). Criterion moved to Phase 7.2 manual test.
- ❌ `GYM_EXPLANATION` step not verified — pre-existing FSM race condition: Norman cutscene auto-advances on gym entry, triggering `dialogue_completed=True` before the player actually speaks to Norman. Both `DAD_FIRST_MEETING` and `GYM_EXPLANATION` advance prematurely. Decision: do NOT fix; HTN resolves this by design (uses episodic transcript evidence, not `dialogue_completed` flag). Criterion moved to Phase 7.2 manual test.
- ⚠️ Key divergence finding: with `last_completed=PETALBURG_CITY`, HTN bootstrap generates stack `[I]Exit Petalburg City west to Route 104 South` — skips `DAD_FIRST_MEETING`. FSM correctly locks to `PETALBURG_CITY_GYM` via RAG LOCK (`DAD_FIRST_MEETING requires PETALBURG_CITY_GYM — RAG suggested ROUTE_104_SOUTH — ignored`). This is the primary divergence Phase 7.2 must address.

**Manual — Full Petalburg Corridor Run with HTN Active (`boundary_test.state`):**

*Purpose:* End-to-end validation (Stage 7.2 — immediate layer handoff) that the
HTN system drives the agent from the Petalburg City entrance → gym → Norman
dialogue → Route 104 South, with the Supervisor correctly issuing `CONTINUE`
during dialogue and `POP` after Norman has spoken.

*Command:*
```bash
python run.py --load-state tests/save_states/boundary_test.state --agent-auto --use-htn
```

*Observe in console per step:*
```
[STEP 001] dispatch → nav_bot    | [I]Enter Petalburg City  goal=PETALBURG_CITY
[STEP 008] dispatch → nav_bot    | [I]Navigate to Petalburg Gym  goal=PETALBURG_CITY_GYM
[STEP 015] dispatch → coms_bot   | NPC dialogue triggered
[HANDOFF]  nav_bot → coms_bot    pending=True
[SUPERVISOR] step=15  CONTINUE  "Norman dialogue not yet complete"
[STEP 025] dispatch → nav_bot    | exiting dialogue
[HANDOFF]  coms_bot → nav_bot    pending=True
[SUPERVISOR] step=25  POP  "Norman dialogue complete — transcript confirmed"
[STEP 026] dispatch → nav_bot    | [T]Head to Route 104 South  goal=ROUTE_104_SOUTH
```

*Pass criteria:*
- [ ] Agent enters Petalburg City within 10 steps (nav_bot driving HTN immediate goal)
- [ ] Agent enters Petalburg Gym within 30 steps of starting
- [ ] Supervisor issues `CONTINUE` while Norman dialogue is in progress (not a premature POP)
- [ ] Supervisor issues `POP` after the coms_bot → nav_bot handoff when Norman has spoken
- [ ] After POP, Stack[0] advances to the next tactical goal (toward Route 104 South or Rustboro)
- [ ] Agent begins navigating toward Route 104 South within 10 steps of the POP
- [ ] `DAD_FIRST_MEETING` milestone logs as complete in the run console

*Subjective evaluation (rate 1–5, note in `run_logs/eval_notes.txt`):*
- **Goal coherence** — Does the Supervisor's reasoning text accurately describe the situation?
- **Stack discipline** — Does the stack depth stay ≤ 4 throughout? (deep stacks indicate PUSH loops)
- **Handoff responsiveness** — Is there noticeable hesitation (>3 PASS steps) after a handoff?

*Fail indicators:*
- Agent enters gym and immediately exits (premature POP before Norman speaks): `completion_condition` check is triggering on map entry, not on dialogue completion — tighten the condition string to require episodic evidence of Norman's dialogue keywords
- Supervisor POP never fires after Norman dialogue: `coms_bot_node` is not writing dialogue turns to `game_history`, so the Supervisor always sees empty context and defaults to `CONTINUE` — confirm `episodic_memory.log_event` is called inside `coms_bot_node`
- Stack grows unbounded: a PUSH loop where each coms_bot trigger pushes a new "talk to NPC" goal — add a guard in `executive_supervisor_node` that rejects PUSH when `goal_stack` depth > 6

*Status:* ⚠️ BLOCKED — gym entry/exit loop prevents completion. See **Phase 7.2 Addendum** below for root-cause analysis and implementation plan. Re-run after `verification_node` gate is implemented.

---

## Phase 7.2 Addendum: Dialogue Milestone Authority Transfer

**Purpose:** Fix the gym entry/exit loop observed when `--use-htn` is active. When the agent enters Petalburg City Gym, `verification_node` advances `milestone_index` from 17 (`DAD_FIRST_MEETING`) to 19 (`ROUTE_104_SOUTH`) in a single pass — before Norman speaks. The navigation target becomes Route 104 South while the agent is at gym interior coordinates. A* cannot path 103 tiles north through a building warp boundary; it falls back to pressing UP, which triggers the gym exit warp, and the loop begins.

### Root Cause

Two milestones advance prematurely on gym entry:

1. **`DAD_FIRST_MEETING` (index 17, `completion_type="location"`)** — fires when `player.location == "PETALBURG_CITY_GYM"`. Correct and intentional in both modes.

2. **`GYM_EXPLANATION` (index 18, `completion_type="dialogue"`)** — fires because the GBA ROM sets `dialogue_completed = True` in memory on gym map entry, not after Norman speaks. The `TransitionEvaluator` reads this flag and immediately satisfies the dialogue milestone on the **same** verification pass as `DAD_FIRST_MEETING`.

Both milestones resolve in one `verification_node` pass. `milestone_index` jumps from 17 to 19. The navigation directive becomes `ROUTE_104_SOUTH`, but the agent is inside the gym at coordinates that are 103+ tiles south of the Route 104 exit. A* fails; the fallback presses UP; the gym exit warp fires. Loop.

### Authority Split

| Condition | Authority for dialogue milestone advancement |
|---|---|
| `use_htn=False` (FSM mode) | `verification_node` — advances via `dialogue_completed` flag set by `TransitionEvaluator` — **unchanged** |
| `use_htn=True` (HTN mode) | Executive Supervisor — advances `milestone_index` only after `POP`ing the corresponding stack task, backed by `_query_dialogue_context()` evidence |

When `use_htn=True`, the Supervisor uses `_query_dialogue_context()` to verify the completion condition against `game_history` ChromaDB records. These records are only written by `coms_bot_node` when fully-rendered dialogue frames are captured. Map entry alone cannot produce a `dialogue_transcript` record, so premature POP is structurally impossible.

### Implementation: Three-Component Fix

#### Component 1 — `verification_node` Dialogue Gate

**File:** `agent/graph/nodes/verification_node.py`

The `make_verification_node(use_htn=False)` factory must accept a `use_htn` parameter (same pattern as `make_executive_supervisor_node`). In the branch where a dialogue milestone fires:

```python
# When use_htn=True, the Supervisor is the authority for dialogue milestones.
# Signal it and do NOT advance milestone_index.
if use_htn and _is_dialogue_milestone(current_milestone_id):
    logger.info(
        "[VERIFY] %s: dialogue check → use_htn=True, deferring to Supervisor",
        current_milestone_id,
    )
    return {
        **state,
        "dialogue_milestone_pending": current_milestone_id,
    }

# Legacy FSM path (use_htn=False): advance as before
_advance_milestone(state, ...)
```

`_is_dialogue_milestone(mid)` checks `MILESTONE_PROGRESSION[index]["completion_type"] == "dialogue"`.

Wire `use_htn` into `make_verification_node` in `build_graph()`:
```python
verification = make_verification_node(use_htn=use_htn)
graph.add_node("verification", verification)
```

#### Component 2 — `AgentState` Field

**File:** `agent/graph/state.py`

```python
dialogue_milestone_pending: Optional[str]
"""Set by verification_node (use_htn=True) when a dialogue milestone fires but
the Supervisor — not the FSM — is the authority. Value is the milestone_id string
(e.g. "GYM_EXPLANATION"). Cleared by executive_supervisor_node after POP."""
```

#### Component 3 — Supervisor POP Advances Pending Milestone

**File:** `agent/graph/nodes/executive_supervisor.py`

In the `POP` branch of `executive_supervisor_node`, after `stack_pop()`:

```python
elif op == "POP":
    popped, stack = stack_pop(stack)
    logger.info("[SUPERVISOR] POP '%s' — %s", popped.goal_id if popped else "?", reason)

    # When use_htn=True: if verification_node deferred a dialogue milestone,
    # advance milestone_index now (Supervisor is the authority).
    pending_mid = state.get("dialogue_milestone_pending")
    if use_htn and pending_mid:
        _advance_milestone_for_id(state, pending_mid)
        new_state_patch["dialogue_milestone_pending"] = None
        logger.info("[SUPERVISOR] Advancing deferred dialogue milestone: %s", pending_mid)

    # Repopulate sub-goals if parent is now an exposed strategic goal with no children
    parent = stack_peek(stack)
    if parent and parent.goal_type == "strategic" and not _has_children(stack, parent):
        new_sub_goals = _expand_strategic_goal(parent, state_data, walkthrough_db, vlm)
        for g in reversed(new_sub_goals):
            stack = stack_push(stack, g)
```

**`_advance_milestone_for_id(state, milestone_id)` helper** — new function in `executive_supervisor.py`:

```python
def _advance_milestone_for_id(state: dict, milestone_id: str) -> None:
    """Increment milestone_index past the given milestone_id.

    Scans MILESTONE_PROGRESSION for the milestone_id and sets
    state["milestone_index"] to the next index. No-ops if milestone_index
    is already past the given ID.
    """
    from agent.objective_manager import MILESTONE_PROGRESSION
    for i, entry in enumerate(MILESTONE_PROGRESSION):
        if entry["milestone"] == milestone_id:
            current = state.get("milestone_index", 0)
            if current <= i:
                state["milestone_index"] = i + 1
                logger.info(
                    "[SUPERVISOR] milestone_index %d → %d (%s completed)",
                    current, i + 1, milestone_id,
                )
            return
```

> **Note on mutation:** `_advance_milestone_for_id` mutates the `state` dict in-place
> before the return dict is assembled. This matches the pattern already used by
> `_apply_immediate_directive`. The caller must include `"milestone_index"` and
> `"dialogue_milestone_pending"` in the final `new_state.update({...})` block.

### Phase 7.2 Addendum Tests

**Automated — `tests/test_htn_verification_gate.py`:**

```python
class TestDialogueMilestoneGated:
    # verification_node with use_htn=True; current milestone is GYM_EXPLANATION (dialogue type)
    # dialogue_completed = True in state_data
    # → milestone_index unchanged in returned state
    # → "dialogue_milestone_pending" == "GYM_EXPLANATION" in returned state

class TestDialogueMilestoneNotGatedFSM:
    # verification_node with use_htn=False; same GYM_EXPLANATION dialogue milestone fires
    # → milestone_index advances (legacy FSM path unchanged)
    # → "dialogue_milestone_pending" not set / None

class TestLocationMilestoneAlwaysAdvances:
    # verification_node with use_htn=True; current milestone is DAD_FIRST_MEETING (location type)
    # player.location == "PETALBURG_CITY_GYM"
    # → milestone_index advances (location milestones unaffected by this gate)
    # → "dialogue_milestone_pending" not set

class TestSupervisorPopAdvancesPendingMilestone:
    # State has dialogue_milestone_pending="GYM_EXPLANATION", milestone_index=18
    # Supervisor LLM mock returns POP
    # → milestone_index == 19 in returned state
    # → "dialogue_milestone_pending" == None in returned state

class TestSupervisorContinueDoesNotAdvance:
    # State has dialogue_milestone_pending="GYM_EXPLANATION", milestone_index=18
    # Supervisor LLM mock returns CONTINUE
    # → milestone_index unchanged (still 18)
    # → "dialogue_milestone_pending" still "GYM_EXPLANATION"

class TestSupervisorPopNoPending:
    # State has dialogue_milestone_pending=None
    # Supervisor LLM mock returns POP
    # → milestone_index unchanged (no pending milestone to advance)
    # → no crash, no KeyError

class TestAdvanceMilestoneForIdAlreadyPast:
    # milestone_index is already 19 when _advance_milestone_for_id("GYM_EXPLANATION") called
    # → no-op; milestone_index remains 19

class TestGymEntrySequence:
    # Simulates the full gym-entry sequence end-to-end:
    # Step 1: verification_node fires with DAD_FIRST_MEETING (location) → milestone_index 17→18
    # Step 2: verification_node fires with GYM_EXPLANATION (dialogue, use_htn=True)
    #         → gated; dialogue_milestone_pending="GYM_EXPLANATION"; milestone_index stays 18
    # Step 3: Supervisor fires CONTINUE (mock episodic memory has no dialogue records)
    #         → milestone_index 18; dialogue_milestone_pending still set
    # Step 4: Supervisor fires POP (mock episodic memory contains Norman dialogue keywords)
    #         → milestone_index advances to 19; dialogue_milestone_pending cleared to None
    #         → Stack[0] directive points to ROUTE_104_SOUTH (confirmed via _apply_immediate_directive)
```

**Automated test command:**
```bash
.venv/bin/python -m pytest tests/test_htn_verification_gate.py -v
```

*Pass criteria:* All 8 tests green (or more if finer-grained coverage is added).

**Manual — Gym Loop Fix Verification (`boundary_test.state`):**

*Purpose:* Confirm the agent enters Petalburg City Gym, stays inside until Norman speaks, then navigates toward Route 104 South without the entry/exit loop. This is also the unblock condition for the Phase 7.2 "Full Petalburg Corridor Run" manual test above.

*Command:*
```bash
python run.py --load-state tests/save_states/boundary_test.state --agent-auto --use-htn
```

*Observe in console:*
```
[VERIFY] DAD_FIRST_MEETING: location check → advancing milestone_index 17 → 18
[VERIFY] GYM_EXPLANATION: dialogue check → use_htn=True, deferring to Supervisor
[SUPERVISOR] step=N  CONTINUE  "Norman has not yet spoken — no dialogue_transcript in game_history"
[coms_bot] NPC dialogue captured → logged to game_history
[SUPERVISOR] step=M  POP  "Norman dialogue confirmed in episodic memory"
[SUPERVISOR] Advancing deferred dialogue milestone: GYM_EXPLANATION → milestone_index 18 → 19
[NAVBOT] goal_location=ROUTE_104_SOUTH
```

*Pass criteria:*
- [ ] Agent enters gym and does NOT exit on the immediately following step
- [ ] `[VERIFY] GYM_EXPLANATION: ... deferring to Supervisor` log line appears in console
- [ ] `milestone_index` stays at 18 while agent is inside gym awaiting Norman dialogue
- [ ] Supervisor issues `CONTINUE` at least once while `dialogue_milestone_pending` is set
- [ ] Supervisor issues `POP` after Norman dialogue is captured to `game_history`
- [ ] After POP: `milestone_index` = 19 and `Stack[0].directive.goal_location` = `ROUTE_104_SOUTH`
- [ ] Agent exits gym and begins navigating north toward Route 104 South
- [ ] **Full corridor**: PC heal → Norman dialogue → Route 104 South navigation all complete in one run

*Fail indicators:*
- Agent still exits gym immediately: `use_htn` not passed to `make_verification_node()` in `build_graph()` — confirm factory call signature
- `[VERIFY] GYM_EXPLANATION` line never appears: code path not reached — add a temporary print before the `_is_dialogue_milestone()` check to confirm `verification_node` sees the right `current_milestone_id`
- Supervisor always issues `CONTINUE` even after Norman speaks: `coms_bot_node` not writing to `game_history`, or `episodic_memory` not wired in `build_graph()` — confirm `make_coms_bot_node(episodic_memory=...)` call
- `dialogue_milestone_pending` never cleared: POP branch not reaching the `pending_mid` check — confirm it is inside `if op == "POP":`, not in the outer `new_state.update()` block

*Status:* 🔲 NOT YET RUN

---

## Implementation Checklist

| Phase | File | Change Type | Status |
|-------|------|------------|--------|
| 0 | `agent/graph/goal_stack.py` | **CREATE** | ✅ |
| 0 | `agent/graph/state.py` | MODIFY (add 5 HTN fields) | ✅ |
| 0 | `tests/test_goal_stack.py` | **CREATE** | ✅ |
| 0 | `tests/test_agent_state_htn.py` | **CREATE** | ✅ |
| 1 | `agent/graph/nodes/handoff_detector.py` | **CREATE** (includes nav-stall detection) | ✅ |
| 1 | `agent/graph/graph.py` | MODIFY (rewire edges) | ✅ |
| 1 | `tests/test_handoff_detector.py` | **CREATE** | ✅ |
| 2 | `agent/graph/nodes/executive_supervisor.py` | **CREATE** | ✅ |
| 2 | `tests/test_executive_supervisor.py` | **CREATE** | ✅ |
| 3 | (prompts embedded in `executive_supervisor.py`) | n/a | ✅ |
| 3 | `utils/vlm.py` | MODIFY — add `get_json_query(system_prompt, user_prompt, module_name, timeout=30)` to `GeminiBackend`, `VertexBackend`, and `VLM` facade; HTN calls use `timeout=60` | ✅ |
| 3 | `tests/test_supervisor_prompt.py` | **CREATE** | ✅ |
| 4 | `executive_supervisor.py` (`_bootstrap_stack`, `_expand_strategic_goal`) | part of Phase 2 file | ✅ |
| 4 | `tests/test_htn_bootstrap.py` | **CREATE** | ✅ |
| 5.5 | `scripts/build_walkthrough_db.py` | MODIFY — add `generate_location_graph_chunks()` to auto-generate topology RAG chunks from `LOCATION_GRAPH`; supersedes all 5 `SUPPLEMENTAL_CHUNKS` | ☐ |
| 5 | `agent/graph/nodes/battle_bot.py` | MODIFY — refactor to `make_battle_bot_node(episodic_memory)` factory; log `battle_start` + `battle_outcome` events to ChromaDB **(required)** | ☐ |
| 5 | `agent/graph/nodes/executive_supervisor.py` | MODIFY — replace `_query_episodic_memory()` with `_query_dialogue_context()` + `_query_battle_outcomes()`; update `_call_supervisor_llm` signature | ☐ |
| 5 | `tests/test_supervisor_memory.py` | **CREATE** | ☐ |
| 6 | `agent/graph/state.py` | MODIFY (add `_boot_timestamp`) | ✅ |
| 6 | `agent/__init__.py` | MODIFY (set `_boot_timestamp`) | ✅ |
| 6 | `tests/test_boot_sequence.py` | **CREATE** | ✅ |
| 7.1 | `agent/graph/nodes/executive_supervisor.py` | MODIFY (add shadow jsonl logging) | ✅ |
| 7.2 | `run.py` | MODIFY (add `--use-htn` flag) | ✅ |
| 7.2 | `agent/graph/graph.py` | MODIFY (accept `use_htn` param; pass to `make_executive_supervisor_node`) | ✅ |
| 7.2 | `agent/__init__.py` | MODIFY (read `use_htn` from args; pass to `build_graph`; init `_graph_milestone_index` from game state on first step) | ✅ |
| 7.1 | `tests/test_htn_shadow_log.py` | **CREATE** | ✅ |
| 7.2 | `tests/test_htn_full_cycle.py` | **CREATE** | ✅ |
| 7.2A | `agent/graph/state.py` | MODIFY — add `dialogue_milestone_pending: Optional[str]` field | ☐ |
| 7.2A | `agent/graph/nodes/verification_node.py` | MODIFY — add `use_htn` factory parameter; gate `completion_type="dialogue"` milestones when `use_htn=True`; set `dialogue_milestone_pending` instead of advancing | ☐ |
| 7.2A | `agent/graph/nodes/executive_supervisor.py` | MODIFY — in POP branch: check `dialogue_milestone_pending`, call `_advance_milestone_for_id()`, clear field; add `_advance_milestone_for_id()` helper | ☐ |
| 7.2A | `agent/graph/graph.py` | MODIFY — pass `use_htn` to `make_verification_node()` | ☐ |
| 7.2A | `tests/test_htn_verification_gate.py` | **CREATE** | ☐ |

---

## Known Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| LLM outputs unparseable JSON on first boot | `_bootstrap_stack` falls back to `_milestone_fallback_stack()` — no crash, no agent stall |
| Supervisor fires too frequently (high token cost) | Handoff detector strictly gates on `_SIGNIFICANT_TRANSITIONS`; same-node loops never trigger |
| Goal stack diverges from actual game state | `_boot_timestamp` filtering + milestone.json bootstrap ensures re-sync on every load |
| LLM generates invalid `LOCATION_GRAPH` keys | `LocationResolver.resolve_location_key()` (existing) maps prose → graph keys; returns `None` on failure → nav_bot uses directional fallback |
| RAG returns irrelevant chunks | Existing `WalkthroughDB` distance threshold already filters low-confidence results; `_bootstrap_stack` uses `milestone_fallback_stack` when `context_text` is empty |
| Parallel `Objective` list in `ObjectiveManager` drifts | Fixed in Stage 4 migration; `_initialize_storyline_objectives()` is deleted. Until then, `milestone_index` drives `verification_node` as before |

---

## Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **HTN** | Hierarchical Task Network — a plan representation where goals nest recursively |
| **Goal Stack** | The live HTN; `Stack[0]` is the current immediate action |
| **Executive Supervisor** | LLM node that reads stack + game state and issues stack operations |
| **Plant Controller** | Deterministic specialist node: `nav_bot`, `battle_bot`, `coms_bot` |
| **Handoff** | The moment control transitions from one plant controller to another |
| **Bootstrap** | Building the initial goal stack from `milestones.json` + walkthrough RAG |
| **World State Snapshot** | The `*_milestones.json` companion file for a save state |

## Appendix B: How This Fixes the Known Problems from `OBJECTIVE_TRACKING_SYSTEM.md`

| Known Problem | HTN Solution |
|---|---|
| **A. RAG overrides gym milestone** | The Supervisor decides overrides via LLM reasoning, not a brittle `completion_type != "dialogue"` flag. Gym dialogue is locked by the `completion_condition` string, not by a data field. |
| **B. Gym interior A\* impossible** | The Supervisor issues `goal_type="immediate"` goals like "Enter Gym" (warp handled automatically by game engine). The HTN never needs to plan past the warp boundary. |
| **C. Duplicate Objective list** | `_initialize_storyline_objectives()` is deleted in Stage 4. The `GoalNode` list *is* the objective list. |
| **D. Cannot distinguish "entered gym" from "completed scene"** | Supervisor's `completion_condition` is checked against `game_history` episodic logs, which capture the actual dialogue transcript from `coms_bot_node`. "Norman cutscene ran" is verifiable from transcript content. |

---

## Appendix C: Milestone Source Architecture

This appendix answers the question: **which source should the system trust for
milestone completion, and when?**

### Three Sources, Three Roles

| Source | What it is | When it is authoritative |
|---|---|---|
| **ROM flags** (`state_data["milestones"]` from emulator RAM) | Hardware bits in the GBA ROM, flipped by in-game scripts | Location arrivals, battle starts/ends — coverage up to `FIRST_GYM_COMPLETE` only |
| **Milestones JSON** (`*_milestones.json` companion file) | Persistent record of which ROM flags have ever fired in this session | **Bootstrap only** — anchors `_get_last_completed_milestone()` at agent startup |
| **ChromaDB `game_history`** | Episodic event log written by `coms_bot_node` and `battle_bot_node` during runtime | **Primary completion evidence** for the Supervisor's POP decisions, at all game stages |

### Decision Table

| Scenario | Use this source |
|---|---|
| Agent starts up, goal stack is empty | Milestones JSON → `_get_last_completed_milestone()` → RAG anchor |
| `verification_node` checking a `completion_type="location"` milestone (index 0–26) | ROM flags via `check_storyline_milestones()` |
| `verification_node` checking a `completion_type="dialogue"` milestone | `dialogue_completed` flag (set by `TransitionEvaluator`) — **ROM flags explicitly ignored** for dialogue milestones because they fire on map entry, not after NPC speech |
| `verification_node` when `milestone_index >= 27` (post-Rustboro) | No-op — Supervisor handles all advancement via ChromaDB |
| Supervisor deciding whether to POP `Stack[0]` | ChromaDB `game_history` — `_query_dialogue_context()` + `_query_battle_outcomes()` |

### Why ROM Flags Fire Early for Dialogue Milestones

The GBA ROM sets `DAD_FIRST_MEETING` and `GYM_EXPLANATION` flags the moment
the player's map tile is within the Petalburg Gym map boundary — i.e., on gym
**entry**, not after Norman speaks. The ROM flag mechanism is not granular
enough to distinguish "entered map" from "completed cutscene".

**Current fix (already implemented):** `check_storyline_milestones()` skips
auto-completion when `_MILESTONE_COMPLETION_TYPE[milestone_id] == "dialogue"`.
The verification node then waits for `dialogue_completed == True`, which is
set by `TransitionEvaluator` after it confirms the expected keywords appear in
the `_SESSION_TRANSCRIPT`.

**HTN fix (Phase 7+):** The Supervisor's `completion_condition` string for
`DAD_FIRST_MEETING` will be something like:
```
"Norman has spoken to the player about gym challenges. Episodic memory
contains dialogue from Norman explaining the gym or badges."
```
This is verified against `_query_dialogue_context()`, which only returns
records with `type="dialogue_transcript"`. Map entry alone does not produce
a `dialogue_transcript` record, so premature POP is impossible.

### Why Not Use ROM Flags as the Primary Source?

1. **Coverage ends at Rustboro.** `MILESTONE_PROGRESSION` has 27 entries.
   The ROM exposes no flags for Dewford, Slateport, Mauville, or any later content.
2. **Dialogue milestones fire too early** (see above).
3. **ROM flags cannot capture battle nuance.** They record "battle started"
   but not "battle won", "party HP after", or "which Pokémon was used". The
   Supervisor needs outcome detail to reason about HP recovery needs.

### Why Not Use Milestones JSON as the Runtime Source?

The milestones JSON is a snapshot of which ROM flags have fired. It inherits
all the limitations of ROM flags (coverage, early-fire for dialogue). It is
useful at bootstrap because it gives a persistent cross-session anchor, but
it lags behind the live game state by at least one step (it's updated
asynchronously by the emulator).

### Summary

> Use ROM flags → verification node → indices 0–26, non-dialogue milestones  
> Use milestones JSON → bootstrap anchor only  
> Use ChromaDB → Supervisor POP decisions, all stages, all content

---

## Known Bug Backlog

Issues observed during Phase 5 manual testing (`route102_hackathon` run). Not
blocking HTN work. Address before or during Phase 7.

---

### BUG-1: Battle bot stuck in `UNKNOWN` menu state for ~10 steps

**Severity:** Medium — functional but slow; wastes ~7 extra LLM calls per battle  
**File(s):** `agent/combat/` (battle menu state detection), `agent/battle_bot.py`

**Symptom:**  
During a trainer battle, the battle bot outputs `[MENU STATE] UNKNOWN: no
dialogue text, title='EMPTY'` for approximately 10 consecutive steps before
eventually selecting a move via the grass-type heuristic fallback (`ABSORB →
POOCHYENA`). A battle that should resolve in ~3 steps takes ~13.

**How to reproduce:**  
```
python run.py --load-state Emerald-GBAdvance/route102_hackathon.state --agent-auto
```
Trigger the trainer battle on Route 102. Watch for repeated `MENU STATE UNKNOWN`
log lines between the "BATTLE START" event and the first successful move selection.

**Expected behaviour:**  
The VLM should recognise the GBA battle action-select screen on the first or
second step and return `MENU STATE: ACTION_SELECT` (or equivalent), allowing
the bot to pick `FIGHT` immediately.

**Root cause hypothesis:**  
The VLM prompt / screenshot framing does not reliably parse the battle UI
directly after a dialogue transition (e.g. "A wild POOCHYENA appeared!"). The
fallback heuristic activates only after `N` failed UNKNOWN cycles. Possibly the
screenshot captured during the transition frame still shows dialogue overlay.

---

### BUG-2: Species fuzzy matcher fed non-Pokémon strings

**Severity:** Low — noisy logs only, no functional impact  
**File(s):** wherever species name normalisation is called (search for `SPECIES FIX`)

**Symptom:**  
Log lines such as:
```
⚠️ [SPECIES FIX] No fuzzy match for 'PLAYER CHARACTER'
⚠️ [SPECIES FIX] No fuzzy match for 'BOY TRAINER'
```
These appear when entity detection on the overworld or battle screen picks up
trainer-class strings (sprite labels, NPC role names) and passes them through
the Pokémon species fuzzy matcher.

**How to reproduce:**  
Any run that walks through Route 102 and encounters a trainer will produce these
warnings. They appear in `run_logs/` immediately around the battle encounter.

**Expected behaviour:**  
The species matcher should only be called with strings that are plausible Pokémon
species names. Trainer-class names (`PLAYER CHARACTER`, `BOY TRAINER`, etc.)
should be filtered out before reaching the fuzzy match call — or the matcher
should silently return `None` for obvious non-species strings rather than
emitting a warning.

---

### BUG-3: Stale `battle_ctx` records surface in ChromaDB queries (Phase 6)

**Severity:** Medium — correct record is present but old `(no party data)` records pollute context  
**File(s):** `agent/graph/nodes/battle_bot.py` (`_boot_timestamp`), `agent/memory.py`  
**Tracking:** Already scoped as Phase 6 work

**Symptom:**  
After a battle, `battle_ctx` shown in supervisor context contains multiple old
records like `Battle ended at ROUTE 102. Party HP: (no party data)` alongside
the correct new record. The stale records are from prior debugging runs stored
in the same ChromaDB `game_history` collection.

**How to reproduce:**  
1. Run the agent at least twice against `route102_hackathon.state`.  
2. On the second run, observe the `battle_ctx:` block in the supervisor prompt —
   it will include records from the first run alongside the current run's record.

**Expected behaviour:**  
ChromaDB queries for `battle_outcome` events should only return records written
during the **current session** (i.e. after `_boot_timestamp`). Records from
previous runs should be excluded by the `$and` timestamp filter.

**Fix (deferred to Phase 6):**  
Record a real Unix timestamp in `Agent.__init__()` and store it as
`_boot_timestamp`. The `$and` where-filter in `_query_battle_outcomes` already
has the correct structure — it just needs a non-zero value to activate.
