"""
tests/test_executive_supervisor.py — Phase 2 unit tests for the
executive_supervisor node.

Covers:
  TestBootstrapStub              — empty stack with Phase-2 stub returns no-op
  TestNonEmptyStackContinue      — non-empty stack + CONTINUE stub leaves stack unchanged
  TestStackOperationPop          — mocked LLM PUSH triggers stack shrink
  TestStackOperationPush         — mocked LLM PUSH grows stack
  TestPushDepthCap               — PUSH rejected when stack is at _STACK_DEPTH_CAP
  TestPushDepthCapBoundary       — PUSH allowed when stack is exactly one below cap
  TestStackOperationReplace      — mocked LLM REPLACE swaps Stack[0]
  TestStackOperationContinue     — mocked CONTINUE leaves stack intact
  TestMalformedLLMResponse       — empty / missing keys default gracefully to CONTINUE
  TestDirectiveTranslation       — use_htn=True applies directive to state fields
  TestDirectiveTranslationNoop   — use_htn=True with no directive leaves state unchanged
  TestSupervisorPendingCleared   — supervisor_pending always False after node runs
  TestReasoningTruncation        — supervisor_last_reasoning capped at 500 chars

All tests call ``make_executive_supervisor_node`` with ``None`` VLM/memory
(safe: the Phase-2 stubs never actually call them) and mock
``_call_supervisor_llm`` at module level when they need specific operations.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.graph.goal_stack import GoalNode
from agent.graph.nodes.executive_supervisor import (
    _STACK_DEPTH_CAP,
    _apply_immediate_directive,
    _build_game_summary,
    _has_children,
    make_executive_supervisor_node,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODULE = "agent.graph.nodes.executive_supervisor"


def _make_node(use_htn: bool = False):
    """Return a supervisor node with no-op VLM / memory dependencies."""
    return make_executive_supervisor_node(
        vlm=None,
        episodic_memory=None,
        walkthrough_db=None,
        use_htn=use_htn,
    )


def _goal(goal_id: str, goal_type: str = "immediate", **kwargs) -> GoalNode:
    return GoalNode(goal_id=goal_id, description=f"desc:{goal_id}", goal_type=goal_type, **kwargs)


def _stack_dicts(*nodes: GoalNode) -> list[dict]:
    return [n.to_dict() for n in nodes]


def _base_state(**overrides) -> dict:
    base = {
        "step_count": 1,
        "goal_stack": [],
        "supervisor_pending": True,
        "supervisor_last_operation": None,
        "supervisor_last_reasoning": None,
        "state_data": {},
        "last_action": "NAVIGATE",
        "last_node_fired": "nav_bot",
    }
    base.update(overrides)
    return base


def _llm_response(operation: str, reasoning: str = "test", new_goals: list | None = None) -> dict:
    return {"operation": operation, "reasoning": reasoning, "new_goals": new_goals or []}


# ---------------------------------------------------------------------------
# TestBootstrapStub
# ---------------------------------------------------------------------------

class TestBootstrapStub:
    """Phase-2: _bootstrap_stack is a stub returning [] → node is a no-op."""

    def test_empty_stack_returns_supervisor_pending_false(self):
        node = _make_node()
        result = node(_base_state(goal_stack=[]))
        assert result["supervisor_pending"] is False

    def test_empty_stack_goal_stack_remains_empty(self):
        node = _make_node()
        result = node(_base_state(goal_stack=[]))
        assert result["goal_stack"] == []

    def test_empty_stack_no_operation_set(self):
        node = _make_node()
        result = node(_base_state(goal_stack=[]))
        # supervisor_last_operation is not touched by the no-op branch
        assert result.get("supervisor_last_operation") is None


# ---------------------------------------------------------------------------
# TestNonEmptyStackContinue
# ---------------------------------------------------------------------------

class TestNonEmptyStackContinue:
    """With a non-empty stack the Phase-2 LLM stub always returns CONTINUE."""

    def test_stack_unchanged(self):
        g = _goal("g1")
        node = _make_node()
        result = node(_base_state(goal_stack=_stack_dicts(g)))
        result_ids = [d["goal_id"] for d in result["goal_stack"]]
        assert result_ids == ["g1"]

    def test_operation_is_continue(self):
        g = _goal("g1")
        node = _make_node()
        result = node(_base_state(goal_stack=_stack_dicts(g)))
        assert result["supervisor_last_operation"] == "CONTINUE"

    def test_supervisor_pending_cleared(self):
        g = _goal("g1")
        node = _make_node()
        result = node(_base_state(goal_stack=_stack_dicts(g)))
        assert result["supervisor_pending"] is False


# ---------------------------------------------------------------------------
# TestStackOperationPop
# ---------------------------------------------------------------------------

class TestStackOperationPop:
    def test_pop_reduces_stack_by_one(self):
        g1 = _goal("g1")
        g2 = _goal("g2")
        g3 = _goal("g3")
        node = _make_node()
        with patch(f"{_MODULE}._call_supervisor_llm", return_value=_llm_response("POP")):
            result = node(_base_state(goal_stack=_stack_dicts(g1, g2, g3)))
        assert len(result["goal_stack"]) == 2

    def test_pop_removes_stack_zero(self):
        g1 = _goal("g1")
        g2 = _goal("g2")
        node = _make_node()
        with patch(f"{_MODULE}._call_supervisor_llm", return_value=_llm_response("POP")):
            result = node(_base_state(goal_stack=_stack_dicts(g1, g2)))
        assert result["goal_stack"][0]["goal_id"] == "g2"

    def test_pop_sets_operation(self):
        g1 = _goal("g1")
        g2 = _goal("g2")
        node = _make_node()
        with patch(f"{_MODULE}._call_supervisor_llm", return_value=_llm_response("POP")):
            result = node(_base_state(goal_stack=_stack_dicts(g1, g2)))
        assert result["supervisor_last_operation"] == "POP"

    def test_pop_to_empty_does_not_crash(self):
        g1 = _goal("g1")
        node = _make_node()
        with patch(f"{_MODULE}._call_supervisor_llm", return_value=_llm_response("POP")):
            result = node(_base_state(goal_stack=_stack_dicts(g1)))
        assert result["goal_stack"] == []


# ---------------------------------------------------------------------------
# TestStackOperationPush
# ---------------------------------------------------------------------------

class TestStackOperationPush:
    def _new_goal_dict(self, gid: str) -> dict:
        return {
            "goal_id": gid,
            "description": f"desc:{gid}",
            "goal_type": "immediate",
        }

    def test_push_grows_stack(self):
        g1 = _goal("g1")
        g2 = _goal("g2")
        g3 = _goal("g3")
        new = [self._new_goal_dict("g_new")]
        node = _make_node()
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value=_llm_response("PUSH", new_goals=new),
        ):
            result = node(_base_state(goal_stack=_stack_dicts(g1, g2, g3)))
        assert len(result["goal_stack"]) == 4

    def test_push_prepends_to_front(self):
        g1 = _goal("g1")
        new = [self._new_goal_dict("g_new")]
        node = _make_node()
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value=_llm_response("PUSH", new_goals=new),
        ):
            result = node(_base_state(goal_stack=_stack_dicts(g1)))
        assert result["goal_stack"][0]["goal_id"] == "g_new"
        assert result["goal_stack"][1]["goal_id"] == "g1"

    def test_push_sets_operation(self):
        g1 = _goal("g1")
        new = [self._new_goal_dict("g_new")]
        node = _make_node()
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value=_llm_response("PUSH", new_goals=new),
        ):
            result = node(_base_state(goal_stack=_stack_dicts(g1)))
        assert result["supervisor_last_operation"] == "PUSH"

    def test_push_multiple_goals(self):
        g1 = _goal("g1")
        new = [self._new_goal_dict("a"), self._new_goal_dict("b")]
        node = _make_node()
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value=_llm_response("PUSH", new_goals=new),
        ):
            result = node(_base_state(goal_stack=_stack_dicts(g1)))
        assert len(result["goal_stack"]) == 3


# ---------------------------------------------------------------------------
# TestPushDepthCap
# ---------------------------------------------------------------------------

class TestPushDepthCap:
    def _make_full_stack(self) -> list[dict]:
        return _stack_dicts(*[_goal(f"g{i}") for i in range(_STACK_DEPTH_CAP)])

    def test_push_at_cap_demotes_to_continue(self):
        new = [{"goal_id": "overflow", "description": "x", "goal_type": "immediate"}]
        node = _make_node()
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value=_llm_response("PUSH", new_goals=new),
        ):
            result = node(_base_state(goal_stack=self._make_full_stack()))
        assert result["supervisor_last_operation"] == "CONTINUE"

    def test_push_at_cap_stack_size_unchanged(self):
        new = [{"goal_id": "overflow", "description": "x", "goal_type": "immediate"}]
        node = _make_node()
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value=_llm_response("PUSH", new_goals=new),
        ):
            result = node(_base_state(goal_stack=self._make_full_stack()))
        assert len(result["goal_stack"]) == _STACK_DEPTH_CAP


class TestPushDepthCapBoundary:
    def _make_stack_one_below_cap(self) -> list[dict]:
        return _stack_dicts(*[_goal(f"g{i}") for i in range(_STACK_DEPTH_CAP - 1)])

    def test_push_allowed_one_below_cap(self):
        new = [{"goal_id": "g_new", "description": "x", "goal_type": "immediate"}]
        node = _make_node()
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value=_llm_response("PUSH", new_goals=new),
        ):
            result = node(_base_state(goal_stack=self._make_stack_one_below_cap()))
        assert len(result["goal_stack"]) == _STACK_DEPTH_CAP
        assert result["supervisor_last_operation"] == "PUSH"


# ---------------------------------------------------------------------------
# TestStackOperationReplace
# ---------------------------------------------------------------------------

class TestStackOperationReplace:
    def test_replace_swaps_stack_zero(self):
        g1 = _goal("old_top")
        g2 = _goal("g2")
        new = [{"goal_id": "new_top", "description": "x", "goal_type": "immediate"}]
        node = _make_node()
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value=_llm_response("REPLACE", new_goals=new),
        ):
            result = node(_base_state(goal_stack=_stack_dicts(g1, g2)))
        assert result["goal_stack"][0]["goal_id"] == "new_top"
        assert result["goal_stack"][1]["goal_id"] == "g2"

    def test_replace_keeps_stack_size(self):
        g1 = _goal("old_top")
        g2 = _goal("g2")
        new = [{"goal_id": "new_top", "description": "x", "goal_type": "immediate"}]
        node = _make_node()
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value=_llm_response("REPLACE", new_goals=new),
        ):
            result = node(_base_state(goal_stack=_stack_dicts(g1, g2)))
        assert len(result["goal_stack"]) == 2

    def test_replace_sets_operation(self):
        g1 = _goal("old_top")
        new = [{"goal_id": "new_top", "description": "x", "goal_type": "immediate"}]
        node = _make_node()
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value=_llm_response("REPLACE", new_goals=new),
        ):
            result = node(_base_state(goal_stack=_stack_dicts(g1)))
        assert result["supervisor_last_operation"] == "REPLACE"

    def test_replace_no_new_goals_is_no_op(self):
        """REPLACE with no new_goals list leaves the stack untouched."""
        g1 = _goal("g1")
        node = _make_node()
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value=_llm_response("REPLACE", new_goals=[]),
        ):
            result = node(_base_state(goal_stack=_stack_dicts(g1)))
        assert result["goal_stack"][0]["goal_id"] == "g1"


# ---------------------------------------------------------------------------
# TestStackOperationContinue
# ---------------------------------------------------------------------------

class TestStackOperationContinue:
    def test_continue_stack_unchanged(self):
        g1 = _goal("g1")
        g2 = _goal("g2")
        node = _make_node()
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value=_llm_response("CONTINUE"),
        ):
            result = node(_base_state(goal_stack=_stack_dicts(g1, g2)))
        ids = [d["goal_id"] for d in result["goal_stack"]]
        assert ids == ["g1", "g2"]

    def test_continue_sets_operation(self):
        g1 = _goal("g1")
        node = _make_node()
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value=_llm_response("CONTINUE"),
        ):
            result = node(_base_state(goal_stack=_stack_dicts(g1)))
        assert result["supervisor_last_operation"] == "CONTINUE"


# ---------------------------------------------------------------------------
# TestMalformedLLMResponse
# ---------------------------------------------------------------------------

class TestMalformedLLMResponse:
    def test_empty_dict_defaults_to_continue(self):
        g1 = _goal("g1")
        node = _make_node()
        with patch(f"{_MODULE}._call_supervisor_llm", return_value={}):
            result = node(_base_state(goal_stack=_stack_dicts(g1)))
        assert result["supervisor_last_operation"] == "CONTINUE"
        assert result["goal_stack"][0]["goal_id"] == "g1"

    def test_unknown_operation_defaults_to_continue(self):
        g1 = _goal("g1")
        node = _make_node()
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value={"operation": "TELEPORT", "reasoning": "?", "new_goals": []},
        ):
            result = node(_base_state(goal_stack=_stack_dicts(g1)))
        assert result["supervisor_last_operation"] == "CONTINUE"

    def test_none_operation_defaults_to_continue(self):
        g1 = _goal("g1")
        node = _make_node()
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value={"operation": None, "reasoning": ""},
        ):
            result = node(_base_state(goal_stack=_stack_dicts(g1)))
        assert result["supervisor_last_operation"] == "CONTINUE"


# ---------------------------------------------------------------------------
# TestDirectiveTranslation
# ---------------------------------------------------------------------------

class TestDirectiveTranslation:
    """When use_htn=True the top-of-stack directive is applied to AgentState."""

    def test_goal_location_applied(self):
        g = _goal(
            "nav_gym",
            goal_type="immediate",
            directive={"action": "NAVIGATE", "goal_location": "PETALBURG_CITY_GYM"},
        )
        node = _make_node(use_htn=True)
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value=_llm_response("CONTINUE"),
        ):
            result = node(_base_state(goal_stack=_stack_dicts(g)))
        assert result.get("goal_location") == "PETALBURG_CITY_GYM"

    def test_goal_coords_applied(self):
        g = _goal(
            "nav_coords",
            goal_type="immediate",
            directive={"action": "NAVIGATE", "goal_coords": (10, 20, "ROUTE_102")},
        )
        node = _make_node(use_htn=True)
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value=_llm_response("CONTINUE"),
        ):
            result = node(_base_state(goal_stack=_stack_dicts(g)))
        assert result.get("goal_coords") == (10, 20, "ROUTE_102")

    def test_active_milestone_set_to_goal_id(self):
        g = _goal(
            "beat_roxanne",
            goal_type="immediate",
            directive={"action": "BATTLE", "description": "Defeat Roxanne"},
        )
        node = _make_node(use_htn=True)
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value=_llm_response("CONTINUE"),
        ):
            result = node(_base_state(goal_stack=_stack_dicts(g)))
        assert result.get("active_milestone") == "beat_roxanne"

    def test_use_htn_false_does_not_apply_directive(self):
        """Default use_htn=False: directive fields are NOT copied to state."""
        g = _goal(
            "nav_gym",
            goal_type="immediate",
            directive={"action": "NAVIGATE", "goal_location": "PETALBURG_CITY_GYM"},
        )
        node = _make_node(use_htn=False)
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value=_llm_response("CONTINUE"),
        ):
            result = node(_base_state(goal_stack=_stack_dicts(g)))
        # goal_location was not in the original base state, so it should be absent
        assert result.get("goal_location") is None


# ---------------------------------------------------------------------------
# TestDirectiveTranslationNoop
# ---------------------------------------------------------------------------

class TestDirectiveTranslationNoop:
    def test_no_directive_state_unchanged(self):
        g = _goal("g1", directive=None)
        state = _base_state(goal_stack=_stack_dicts(g), goal_location=None)
        result = _apply_immediate_directive(state, [g])
        # No directive → state returned as-is
        assert result.get("goal_location") is None
        assert result.get("goal_coords") is None

    def test_empty_directive_dict_no_patch(self):
        g = _goal("g1", directive={})
        state = _base_state(goal_stack=_stack_dicts(g))
        result = _apply_immediate_directive(state, [g])
        assert result.get("goal_location") is None

    def test_apply_on_empty_stack_returns_state_unchanged(self):
        state = _base_state()
        result = _apply_immediate_directive(state, [])
        assert result.get("goal_location") is None


# ---------------------------------------------------------------------------
# TestSupervisorPendingCleared
# ---------------------------------------------------------------------------

class TestSupervisorPendingCleared:
    def test_pending_false_after_bootstrap_stub(self):
        node = _make_node()
        result = node(_base_state(goal_stack=[], supervisor_pending=True))
        assert result["supervisor_pending"] is False

    def test_pending_false_after_continue(self):
        g = _goal("g1")
        node = _make_node()
        result = node(_base_state(goal_stack=_stack_dicts(g), supervisor_pending=True))
        assert result["supervisor_pending"] is False

    def test_pending_false_after_pop(self):
        g1 = _goal("g1")
        g2 = _goal("g2")
        node = _make_node()
        with patch(f"{_MODULE}._call_supervisor_llm", return_value=_llm_response("POP")):
            result = node(_base_state(goal_stack=_stack_dicts(g1, g2), supervisor_pending=True))
        assert result["supervisor_pending"] is False


# ---------------------------------------------------------------------------
# TestReasoningTruncation
# ---------------------------------------------------------------------------

class TestReasoningTruncation:
    def test_long_reasoning_truncated_to_500(self):
        long_reason = "x" * 600
        g = _goal("g1")
        node = _make_node()
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value=_llm_response("CONTINUE", reasoning=long_reason),
        ):
            result = node(_base_state(goal_stack=_stack_dicts(g)))
        assert len(result["supervisor_last_reasoning"]) == 500

    def test_short_reasoning_not_padded(self):
        short = "short reason"
        g = _goal("g1")
        node = _make_node()
        with patch(
            f"{_MODULE}._call_supervisor_llm",
            return_value=_llm_response("CONTINUE", reasoning=short),
        ):
            result = node(_base_state(goal_stack=_stack_dicts(g)))
        assert result["supervisor_last_reasoning"] == short


# ---------------------------------------------------------------------------
# TestHasChildren (unit tests for the helper)
# ---------------------------------------------------------------------------

class TestHasChildren:
    def test_returns_true_when_child_present(self):
        parent = _goal("parent", goal_type="strategic")
        child = GoalNode(
            goal_id="child",
            description="child",
            goal_type="tactical",
            parent_id="parent",
        )
        assert _has_children([parent, child], parent) is True

    def test_returns_false_when_no_children(self):
        parent = _goal("parent", goal_type="strategic")
        other = _goal("other", goal_type="immediate")
        assert _has_children([parent, other], parent) is False

    def test_returns_false_on_empty_stack(self):
        parent = _goal("parent", goal_type="strategic")
        assert _has_children([], parent) is False


# ---------------------------------------------------------------------------
# TestBuildGameSummary (unit tests for the context helper)
# ---------------------------------------------------------------------------

class TestBuildGameSummary:
    def test_basic_output_has_location(self):
        state_data = {
            "player": {"location": "OLDALE_TOWN", "position": {"x": 5, "y": 10}},
            "game": {"badges": 0, "in_battle": False},
            "party": [{"name": "Treecko", "current_hp": 20, "max_hp": 25}],
        }
        summary = _build_game_summary(state_data, {})
        assert "OLDALE_TOWN" in summary
        assert "Treecko" in summary

    def test_dict_badges_counted(self):
        state_data = {
            "player": {"location": "X"},
            "game": {"badges": {"stone": True, "knuckle": False}, "in_battle": False},
        }
        summary = _build_game_summary(state_data, {})
        assert "Badges: 1" in summary

    def test_missing_state_data_does_not_crash(self):
        summary = _build_game_summary({}, {})
        assert "Unknown" in summary
