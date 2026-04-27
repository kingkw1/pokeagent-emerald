"""
agent/graph/goal_stack — HTN Goal Stack data structures and pure stack operations.

Defines:
  GoalNode       — a single node in the HTN hierarchy (strategic / tactical / immediate)
  stack_peek     — return Stack[0] without removing it
  stack_pop      — remove and return Stack[0]
  stack_push     — prepend a new goal to the front of the stack
  stack_replace  — replace Stack[0] with a new goal
  stack_summary  — compact one-line string for logging

Design notes:
  • All stack operations are **pure functions** — they return a new list and
    never mutate the argument.  This keeps LangGraph state transitions clean.
  • GoalNode.to_dict() / GoalNode.from_dict() round-trip through plain dicts
    so the stack can be stored as List[dict] in AgentState (LangGraph requires
    JSON-serialisable state values).
  • goal_type must be one of {"strategic", "tactical", "immediate"}.
    strategic  = high-level quest objective (e.g. "Defeat Roxanne")
    tactical   = mid-level plan step        (e.g. "Reach Rustboro Gym")
    immediate  = single nav / battle / coms directive
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# GoalNode
# ---------------------------------------------------------------------------

_VALID_GOAL_TYPES = frozenset({"strategic", "tactical", "immediate"})


@dataclass
class GoalNode:
    """A single node in the HTN goal stack.

    Attributes:
        goal_id:              Unique identifier, e.g. "get_badge_1" or
                              "traverse_route_102".
        description:          Human-readable description of this goal.
        goal_type:            "strategic" | "tactical" | "immediate"
        parent_id:            goal_id of the parent goal (None for root goals).
        directive:            Optional pre-computed Directive dict for immediate
                              goals.  When set the executor uses it directly
                              without re-querying the Supervisor.
        completion_condition: Natural-language string the Supervisor checks to
                              decide whether to POP this goal.  Example:
                              "Player is in RUSTBORO_CITY_GYM and has interacted
                              with Roxanne."
        metadata:             Arbitrary dict for Supervisor context (e.g.
                              badge_count_threshold, required_items, hp_floor).
        created_at:           Unix timestamp set at construction.
        push_reason:          Why this goal was pushed — kept for logging and
                              offline analysis.
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

    def __post_init__(self) -> None:
        if self.goal_type not in _VALID_GOAL_TYPES:
            raise ValueError(
                f"GoalNode.goal_type must be one of {sorted(_VALID_GOAL_TYPES)!r}, "
                f"got {self.goal_type!r}"
            )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of all fields."""
        return {
            "goal_id":              self.goal_id,
            "description":          self.description,
            "goal_type":            self.goal_type,
            "parent_id":            self.parent_id,
            "directive":            self.directive,
            "completion_condition": self.completion_condition,
            "metadata":             self.metadata,
            "created_at":           self.created_at,
            "push_reason":          self.push_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GoalNode":
        """Reconstruct a GoalNode from a plain dict (e.g. from AgentState)."""
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Pure stack operations
# ---------------------------------------------------------------------------

def stack_peek(stack: list[GoalNode]) -> Optional[GoalNode]:
    """Return Stack[0] (the most immediate goal) without removing it.

    Returns None when the stack is empty.
    """
    return stack[0] if stack else None


def stack_pop(stack: list[GoalNode]) -> tuple[Optional[GoalNode], list[GoalNode]]:
    """Remove and return Stack[0], yielding (popped_goal, remaining_stack).

    Returns (None, []) when the stack is empty.
    """
    if not stack:
        return None, []
    return stack[0], stack[1:]


def stack_push(stack: list[GoalNode], goal: GoalNode) -> list[GoalNode]:
    """Prepend *goal* to the front of *stack* so it becomes the new Stack[0].

    The caller is responsible for ensuring *goal.goal_type == "immediate"* when
    pushing an execution-ready sub-goal.
    """
    return [goal] + stack


def stack_replace(stack: list[GoalNode], goal: GoalNode) -> list[GoalNode]:
    """Replace Stack[0] with *goal*, preserving the rest of the stack.

    If the stack is empty, returns [goal] (equivalent to a push onto empty).
    """
    tail = stack[1:] if len(stack) > 1 else []
    return [goal] + tail


def stack_summary(stack: list[GoalNode]) -> str:
    """Return a compact one-line summary suitable for logging.

    Example output (3-level stack, strategic at back, immediate at front):
        "[S]Defeat Roxanne → [T]Reach Rustboro Gym → [I]Walk north on Route 104"

    The summary is ordered from most-strategic to most-immediate (left-to-right)
    to mirror how humans read a plan.
    """
    if not stack:
        return "(empty)"
    return " → ".join(
        f"[{g.goal_type[0].upper()}]{g.description}"
        for g in reversed(stack)
    )
