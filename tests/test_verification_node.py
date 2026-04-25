"""
Tests for Phase 3 — VerificationNode.

Covers:
  TestMilestoneAdvance   — completed milestone → milestone_index incremented
  TestNoAdvance          — incomplete milestone → state unchanged
  TestOutOfBoundsIndex   — milestone_index beyond list end → no crash
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent.graph.nodes.verification import make_verification_node
from agent.graph.state import AgentState
from agent.objective_manager import MILESTONE_PROGRESSION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> AgentState:
    """Return a minimal valid AgentState for verification scenarios."""
    base: AgentState = {
        "frame": None,
        "state_data": {
            "player": {"position": {"x": 5, "y": 5}},
            "game": {},
            "milestones": {},
        },
        "perception": {},
        "goal_coords": None,
        "goal_location": None,
        "npc_coords": None,
        "should_interact": False,
        "milestone_index": 0,
        "context": "navigation",
        "reward": None,
        "prev_state_snapshot": None,
        "last_action": "NAVIGATE",
        "last_buttons": ["RIGHT"],
        "step_count": 1,
        "telemetry": None,
    }
    base.update(overrides)
    return base


def _mock_obj_manager(completed: dict | None = None) -> MagicMock:
    """Create a minimal mock ObjectiveManager."""
    obj_manager = MagicMock()
    obj_manager.completed_goals = completed if completed is not None else {}
    return obj_manager


# ---------------------------------------------------------------------------
# TestMilestoneAdvance
# ---------------------------------------------------------------------------


class TestMilestoneAdvance:
    def test_increments_milestone_index_when_complete(self):
        """When current milestone is in completed_goals, index is incremented."""
        milestone_name = MILESTONE_PROGRESSION[0]["milestone"]
        obj_manager = _mock_obj_manager({milestone_name: True})
        node = make_verification_node(obj_manager)
        state = _make_state(milestone_index=0)
        result = node(state)
        assert result["milestone_index"] == 1

    def test_check_storyline_milestones_called_each_step(self):
        """check_storyline_milestones is called once per node invocation."""
        obj_manager = _mock_obj_manager()
        node = make_verification_node(obj_manager)
        state = _make_state(milestone_index=0)
        node(state)
        obj_manager.check_storyline_milestones.assert_called_once_with(
            state["state_data"]
        )

    def test_non_index_fields_unchanged_on_advance(self):
        """Fields other than milestone_index are untouched when advancing."""
        milestone_name = MILESTONE_PROGRESSION[1]["milestone"]
        obj_manager = _mock_obj_manager({milestone_name: True})
        node = make_verification_node(obj_manager)
        state = _make_state(milestone_index=1, step_count=88, context="navigation")
        result = node(state)
        assert result["step_count"] == 88
        assert result["context"] == "navigation"
        assert result["last_action"] == "NAVIGATE"

    def test_advance_works_at_any_valid_index(self):
        """Advancing works correctly in the middle of the progression list."""
        mid_idx = len(MILESTONE_PROGRESSION) // 2
        milestone_name = MILESTONE_PROGRESSION[mid_idx]["milestone"]
        obj_manager = _mock_obj_manager({milestone_name: True})
        node = make_verification_node(obj_manager)
        state = _make_state(milestone_index=mid_idx)
        result = node(state)
        assert result["milestone_index"] == mid_idx + 1


# ---------------------------------------------------------------------------
# TestNoAdvance
# ---------------------------------------------------------------------------


class TestNoAdvance:
    def test_state_unchanged_when_milestone_incomplete(self):
        """When current milestone is NOT in completed_goals, index stays."""
        obj_manager = _mock_obj_manager({})
        node = make_verification_node(obj_manager)
        state = _make_state(milestone_index=0)
        result = node(state)
        assert result["milestone_index"] == 0

    def test_other_milestones_complete_does_not_advance(self):
        """Completing a different milestone must not advance the current index."""
        # Mark a later milestone as complete but not the current one (index 0)
        later_name = MILESTONE_PROGRESSION[3]["milestone"]
        obj_manager = _mock_obj_manager({later_name: True})
        node = make_verification_node(obj_manager)
        state = _make_state(milestone_index=0)
        result = node(state)
        assert result["milestone_index"] == 0

    def test_other_fields_unchanged_when_no_advance(self):
        """No mutation of non-index fields on a no-op step."""
        obj_manager = _mock_obj_manager()
        node = make_verification_node(obj_manager)
        state = _make_state(milestone_index=2, step_count=99, context="battle")
        result = node(state)
        assert result["step_count"] == 99
        assert result["context"] == "battle"


# ---------------------------------------------------------------------------
# TestOutOfBoundsIndex
# ---------------------------------------------------------------------------


class TestOutOfBoundsIndex:
    def test_out_of_bounds_returns_state_unchanged(self):
        """milestone_index >= len(MILESTONE_PROGRESSION) → no crash, no change."""
        obj_manager = _mock_obj_manager()
        node = make_verification_node(obj_manager)
        big_idx = len(MILESTONE_PROGRESSION) + 10
        state = _make_state(milestone_index=big_idx)
        result = node(state)
        assert result["milestone_index"] == big_idx

    def test_check_not_called_when_out_of_bounds(self):
        """check_storyline_milestones is not called when index is out of range."""
        obj_manager = _mock_obj_manager()
        node = make_verification_node(obj_manager)
        big_idx = len(MILESTONE_PROGRESSION) + 5
        state = _make_state(milestone_index=big_idx)
        node(state)
        obj_manager.check_storyline_milestones.assert_not_called()

    def test_exact_boundary_index_is_out_of_bounds(self):
        """Index exactly equal to len(MILESTONE_PROGRESSION) is treated as done."""
        obj_manager = _mock_obj_manager()
        node = make_verification_node(obj_manager)
        exact_end = len(MILESTONE_PROGRESSION)
        state = _make_state(milestone_index=exact_end)
        result = node(state)
        assert result["milestone_index"] == exact_end
        obj_manager.check_storyline_milestones.assert_not_called()
