"""
tests/test_agent_state_htn.py — Phase 0 unit tests for the five new HTN fields
added to AgentState.

Covers:
  TestNewFieldsPresent        — new keys can be set and read back without error
  TestGoalStackDefaultsEmpty  — absent goal_stack → .get() returns []
  TestSupervisorPendingDefault — absent supervisor_pending → .get() returns False
  TestHTNFieldTypes            — correct types accepted for all five fields
  TestGoalStackStoredAsDicts   — goal_stack stores serialised dicts, not GoalNode objects
"""

from __future__ import annotations

import pytest

from agent.graph.goal_stack import GoalNode
from agent.graph.state import AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> AgentState:
    """Minimal AgentState that satisfies required fields."""
    base: AgentState = {
        "frame": None,
        "state_data": {},
        "perception": {},
        "goal_coords": None,
        "goal_location": None,
        "npc_coords": None,
        "should_interact": False,
        "milestone_index": 0,
        "context": "navigation",
        "reward": None,
        "prev_state_snapshot": None,
        "last_action": None,
        "last_buttons": [],
        "step_count": 0,
        "telemetry": None,
    }
    base.update(overrides)
    return base


def _make_goal_dict(
    goal_id: str = "test_goal",
    goal_type: str = "immediate",
) -> dict:
    """Return a serialised GoalNode dict."""
    return GoalNode(
        goal_id=goal_id,
        description="Test goal",
        goal_type=goal_type,
    ).to_dict()


# ---------------------------------------------------------------------------
# TestNewFieldsPresent
# ---------------------------------------------------------------------------

class TestNewFieldsPresent:
    def test_goal_stack_can_be_set(self):
        state = _base_state(goal_stack=[])
        assert state["goal_stack"] == []

    def test_last_node_fired_can_be_set(self):
        state = _base_state(last_node_fired="nav_bot")
        assert state["last_node_fired"] == "nav_bot"

    def test_supervisor_pending_can_be_set(self):
        state = _base_state(supervisor_pending=True)
        assert state["supervisor_pending"] is True

    def test_supervisor_last_operation_can_be_set(self):
        state = _base_state(supervisor_last_operation="POP")
        assert state["supervisor_last_operation"] == "POP"

    def test_supervisor_last_reasoning_can_be_set(self):
        reasoning = "Goal is complete because player entered the gym."
        state = _base_state(supervisor_last_reasoning=reasoning)
        assert state["supervisor_last_reasoning"] == reasoning

    def test_all_five_fields_together(self):
        """All five HTN fields coexist without key collision."""
        state = _base_state(
            goal_stack=[_make_goal_dict()],
            last_node_fired="coms_bot",
            supervisor_pending=False,
            supervisor_last_operation="CONTINUE",
            supervisor_last_reasoning="Still working on this goal.",
        )
        assert len(state["goal_stack"]) == 1
        assert state["last_node_fired"] == "coms_bot"
        assert state["supervisor_pending"] is False
        assert state["supervisor_last_operation"] == "CONTINUE"


# ---------------------------------------------------------------------------
# TestGoalStackDefaultsEmpty
# ---------------------------------------------------------------------------

class TestGoalStackDefaultsEmpty:
    def test_absent_goal_stack_returns_empty_list(self):
        """Nodes that haven't set goal_stack yet should fall back to []."""
        state = _base_state()  # goal_stack not set
        assert state.get("goal_stack", []) == []

    def test_empty_goal_stack_is_falsy(self):
        state = _base_state(goal_stack=[])
        assert not state["goal_stack"]

    def test_non_empty_goal_stack_is_truthy(self):
        state = _base_state(goal_stack=[_make_goal_dict()])
        assert state["goal_stack"]


# ---------------------------------------------------------------------------
# TestSupervisorPendingDefault
# ---------------------------------------------------------------------------

class TestSupervisorPendingDefault:
    def test_absent_supervisor_pending_returns_false(self):
        state = _base_state()  # supervisor_pending not set
        assert state.get("supervisor_pending", False) is False

    def test_supervisor_pending_true_is_truthy(self):
        state = _base_state(supervisor_pending=True)
        assert state["supervisor_pending"]

    def test_supervisor_pending_false_is_falsy(self):
        state = _base_state(supervisor_pending=False)
        assert not state["supervisor_pending"]


# ---------------------------------------------------------------------------
# TestHTNFieldTypes
# ---------------------------------------------------------------------------

class TestHTNFieldTypes:
    def test_goal_stack_accepts_list_of_dicts(self):
        goals = [_make_goal_dict("g1", "immediate"), _make_goal_dict("g2", "strategic")]
        state = _base_state(goal_stack=goals)
        assert isinstance(state["goal_stack"], list)
        assert all(isinstance(g, dict) for g in state["goal_stack"])

    def test_last_node_fired_accepts_none(self):
        state = _base_state(last_node_fired=None)
        assert state["last_node_fired"] is None

    def test_last_node_fired_accepts_string(self):
        for node in ("nav_bot", "battle_bot", "coms_bot", "map_stitcher_relay"):
            state = _base_state(last_node_fired=node)
            assert state["last_node_fired"] == node

    def test_supervisor_last_operation_accepts_valid_ops(self):
        for op in ("POP", "CONTINUE", "PUSH", "REPLACE"):
            state = _base_state(supervisor_last_operation=op)
            assert state["supervisor_last_operation"] == op

    def test_supervisor_last_reasoning_accepts_long_string(self):
        long_text = "x" * 1000
        state = _base_state(supervisor_last_reasoning=long_text)
        assert len(state["supervisor_last_reasoning"]) == 1000


# ---------------------------------------------------------------------------
# TestGoalStackStoredAsDicts
# ---------------------------------------------------------------------------

class TestGoalStackStoredAsDicts:
    def test_goal_stack_entries_are_plain_dicts_not_goal_nodes(self):
        """AgentState must store serialised dicts, not GoalNode objects.
        This is required for LangGraph JSON-serialisable state."""
        g = GoalNode("nav_to_gym", "Walk into the gym", "immediate")
        state = _base_state(goal_stack=[g.to_dict()])
        entry = state["goal_stack"][0]
        assert isinstance(entry, dict), "goal_stack entries must be dicts, not GoalNode"
        assert not isinstance(entry, GoalNode)

    def test_goal_stack_dict_can_be_deserialised(self):
        """Entries in goal_stack must survive a GoalNode.from_dict() round-trip."""
        g = GoalNode(
            goal_id="reach_rustboro",
            description="Walk to Rustboro City",
            goal_type="tactical",
            parent_id="beat_roxanne",
            completion_condition="Player is in RUSTBORO_CITY.",
            metadata={"badge_index": 0},
        )
        state = _base_state(goal_stack=[g.to_dict()])
        reconstructed = GoalNode.from_dict(state["goal_stack"][0])
        assert reconstructed.goal_id == "reach_rustboro"
        assert reconstructed.goal_type == "tactical"
        assert reconstructed.metadata == {"badge_index": 0}

    def test_multiple_goals_in_stack(self):
        """goal_stack holds any number of serialised GoalNode dicts."""
        goals = [
            _make_goal_dict("imm", "immediate"),
            _make_goal_dict("tac", "tactical"),
            _make_goal_dict("strat", "strategic"),
        ]
        state = _base_state(goal_stack=goals)
        assert len(state["goal_stack"]) == 3
        assert state["goal_stack"][0]["goal_id"] == "imm"
        assert state["goal_stack"][2]["goal_id"] == "strat"
