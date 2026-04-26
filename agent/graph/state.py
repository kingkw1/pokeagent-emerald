"""
agent/graph/state — Core state types for the LangGraph dispatch graph.

Defines:
  RewardVector       — per-step scalar reward signal (for Karpathy meta-loop)
  TelemetrySnapshot  — per-step VLM call / token / latency snapshot
  AgentState         — the single TypedDict threaded through every graph node
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# RewardVector
# ---------------------------------------------------------------------------


@dataclass
class RewardVector:
    """Per-step reward signal used by the Karpathy meta-loop."""

    milestone_delta: int = 0
    """Number of milestones completed this step (+1 per milestone)."""

    manhattan_delta: float = 0.0
    """Change in Manhattan distance to current goal (positive = getting closer)."""

    party_level_sum_delta: int = 0
    """Total party level change this step."""

    pokédollar_delta: int = 0
    """Money gained (positive) or lost (negative) this step."""

    WEIGHTS: tuple = field(default=(10.0, 0.01, 1.0, 0.001), repr=False)
    """Scalar weights: (milestone, manhattan, level, money)."""

    @property
    def total(self) -> float:
        """Weighted scalar sum of all deltas."""
        deltas = (
            self.milestone_delta,
            self.manhattan_delta,
            self.party_level_sum_delta,
            self.pokédollar_delta,
        )
        return sum(w * d for w, d in zip(self.WEIGHTS, deltas))

    @classmethod
    def compute_delta(cls, prev: dict, curr: dict) -> "RewardVector":
        """Compute a RewardVector from two successive state_data dicts.

        Args:
            prev: state_data dict from the previous step.
            curr: state_data dict from the current step.

        Returns:
            RewardVector with all deltas populated.
        """
        def _money(state: dict) -> int:
            return (
                state.get("player", {}).get("money", 0)
                or state.get("money", 0)
            )

        def _level_sum(state: dict) -> int:
            party = (
                state.get("player", {}).get("party", [])
                or state.get("party", [])
            )
            return sum(p.get("level", 0) for p in (party or []))

        def _milestone_index(state: dict) -> int:
            return state.get("milestone_index", 0)

        def _position(state: dict) -> Optional[tuple[float, float]]:
            pos = (
                state.get("player", {}).get("position")
                or state.get("position")
            )
            if pos is None:
                return None
            return (pos.get("x", 0), pos.get("y", 0))

        def _goal_coords(state: dict) -> Optional[tuple[float, float]]:
            gc = state.get("goal_coords")
            if gc is None:
                return None
            if isinstance(gc, (list, tuple)) and len(gc) >= 2:
                return (float(gc[0]), float(gc[1]))
            return None

        milestone_delta = _milestone_index(curr) - _milestone_index(prev)
        pokédollar_delta = _money(curr) - _money(prev)
        party_level_sum_delta = _level_sum(curr) - _level_sum(prev)

        # Manhattan distance delta: positive means closer to goal
        manhattan_delta = 0.0
        prev_pos = _position(prev)
        curr_pos = _position(curr)
        goal = _goal_coords(curr) or _goal_coords(prev)
        if prev_pos and curr_pos and goal:
            prev_dist = abs(prev_pos[0] - goal[0]) + abs(prev_pos[1] - goal[1])
            curr_dist = abs(curr_pos[0] - goal[0]) + abs(curr_pos[1] - goal[1])
            manhattan_delta = prev_dist - curr_dist  # positive = getting closer

        return cls(
            milestone_delta=milestone_delta,
            manhattan_delta=manhattan_delta,
            party_level_sum_delta=party_level_sum_delta,
            pokédollar_delta=pokédollar_delta,
        )

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict (excludes WEIGHTS)."""
        d = asdict(self)
        d.pop("WEIGHTS", None)
        return d


# ---------------------------------------------------------------------------
# TelemetrySnapshot
# ---------------------------------------------------------------------------


@dataclass
class TelemetrySnapshot:
    """Per-step VLM API call / token / latency snapshot."""

    vlm_calls: int = 0
    """Number of Gemini API calls made this step."""

    input_tokens: int = 0
    """Prompt tokens consumed this step (from response.usage_metadata)."""

    output_tokens: int = 0
    """Completion tokens generated this step."""

    step_latency_ms: float = 0.0
    """Wall-clock time for graph.invoke() in milliseconds."""

    node_fired: str = ""
    """Name of the last node that made a VLM call this step."""

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# AgentState
# ---------------------------------------------------------------------------


class AgentState(TypedDict, total=False):
    """Single state object threaded through every LangGraph node.

    Required fields (must be present at graph entry):
        frame, state_data, perception, milestone_index,
        context, step_count, last_buttons

    Optional fields default to None / empty when absent.
    """

    # ---- Raw game state ----
    frame: Any
    """Current GBA frame object (PIL Image or numpy array)."""

    state_data: dict
    """Full memory_reader / get_comprehensive_state() output."""

    # ---- Perception ----
    perception: dict
    """VLM output dict from perception_step()."""

    # ---- Navigation targets ----
    goal_coords: Optional[tuple]
    """(x, y) tile coordinates of current navigation target."""

    goal_location: Optional[str]
    """Location name of the current navigation target (e.g. 'ROUTE_101')."""

    npc_coords: Optional[tuple]
    """(x, y) tile coordinates of the target NPC to interact with."""

    should_interact: bool
    """If True, nav_bot appends 'A' press when adjacent to npc_coords."""

    # ---- Objective tracking ----
    milestone_index: int
    """Pointer into MILESTONE_PROGRESSION list."""

    context: str
    """Active routing context: 'navigation' | 'battle' | 'dialogue' | 'healing_needed'."""

    # ---- Reward (Karpathy loop) ----
    reward: Optional[RewardVector]
    """Reward computed for this step; None until karpathy_meta_agent wraps the loop."""

    prev_state_snapshot: Optional[dict]
    """Shallow snapshot of state_data from the previous step for delta computation."""

    # ---- Node outputs ----
    last_action: Optional[str]
    """Human-readable label for the action taken this step (e.g. 'NAVIGATE', 'BATTLE')."""

    last_buttons: list
    """GBA button list returned by the active node (e.g. ['RIGHT', 'RIGHT', 'A'])."""

    step_count: int
    """Cumulative step counter."""

    # ---- Telemetry ----
    telemetry: Optional[TelemetrySnapshot]
    """Populated by TelemetryLogger.end_step() after each graph.invoke()."""

    # ---- Phase 5: Dialogue completion tracking ----
    dialogue_completed: Optional[bool]
    """True on the step immediately after a dialogue session ends and
    TransitionEvaluator confirmed the milestone keywords were spoken.
    Set by Agent.step() on the dialogue→navigation transition;
    consumed (and reset) by verification_node."""

    dialogue_transcript: list
    """Ordered list of dialogue turns captured this session.
    Each entry: {\"speaker\": str, \"text\": str, \"step\": int}.
    Accumulated by coms_bot_node when a VLM instance is available."""

    # ---- Navigation goal observability ----
    goal_description: Optional[str]
    """Short human-readable description of what the nav_bot is currently
    navigating toward (e.g. 'Enter Petalburg Gym to meet Dad').
    Populated by Agent.step() from the active directive's description field."""

    active_milestone: Optional[str]
    """ID of the milestone the agent is currently working toward
    (e.g. 'DAD_FIRST_MEETING').  Printed by nav_bot_node for observability."""
