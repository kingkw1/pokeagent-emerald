"""
tests/test_goal_stack.py — Phase 0 unit tests for GoalNode and stack operations.

Covers:
  TestGoalNodeSerialization — to_dict / from_dict round-trip
  TestGoalNodeDefaults      — default field values
  TestGoalNodeValidation    — invalid goal_type raises ValueError
  TestStackPush             — stack_push prepends correctly
  TestStackPop              — stack_pop removes Stack[0]
  TestStackReplace          — stack_replace swaps Stack[0]
  TestStackPeek             — stack_peek is non-destructive
  TestStackSummary          — human-readable summary string
"""

from __future__ import annotations

import time

import pytest

from agent.graph.goal_stack import (
    GoalNode,
    stack_peek,
    stack_pop,
    stack_push,
    stack_replace,
    stack_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _immediate(goal_id: str = "nav_to_gym", description: str = "Walk into the gym") -> GoalNode:
    return GoalNode(
        goal_id=goal_id,
        description=description,
        goal_type="immediate",
        directive={"action": "NAVIGATE", "goal_location": "RUSTBORO_CITY_GYM"},
    )


def _tactical(goal_id: str = "reach_rustboro", description: str = "Reach Rustboro City") -> GoalNode:
    return GoalNode(
        goal_id=goal_id,
        description=description,
        goal_type="tactical",
        parent_id="beat_roxanne",
        completion_condition="Player is in RUSTBORO_CITY.",
    )


def _strategic(goal_id: str = "beat_roxanne", description: str = "Defeat Roxanne") -> GoalNode:
    return GoalNode(
        goal_id=goal_id,
        description=description,
        goal_type="strategic",
        completion_condition="Player has the Stone Badge.",
        metadata={"badge_index": 0},
    )


# ---------------------------------------------------------------------------
# TestGoalNodeSerialization
# ---------------------------------------------------------------------------

class TestGoalNodeSerialization:
    def test_to_dict_has_all_keys(self):
        g = _immediate()
        d = g.to_dict()
        expected_keys = {
            "goal_id", "description", "goal_type", "parent_id",
            "directive", "completion_condition", "metadata",
            "created_at", "push_reason",
        }
        assert set(d.keys()) == expected_keys

    def test_from_dict_round_trip_goal_id(self):
        g = _immediate("my_goal")
        assert GoalNode.from_dict(g.to_dict()).goal_id == "my_goal"

    def test_from_dict_round_trip_description(self):
        g = _tactical()
        assert GoalNode.from_dict(g.to_dict()).description == g.description

    def test_from_dict_round_trip_goal_type(self):
        g = _strategic()
        assert GoalNode.from_dict(g.to_dict()).goal_type == "strategic"

    def test_from_dict_round_trip_parent_id(self):
        g = _tactical()
        assert GoalNode.from_dict(g.to_dict()).parent_id == "beat_roxanne"

    def test_from_dict_round_trip_directive(self):
        g = _immediate()
        assert GoalNode.from_dict(g.to_dict()).directive == g.directive

    def test_from_dict_round_trip_metadata(self):
        g = _strategic()
        assert GoalNode.from_dict(g.to_dict()).metadata == {"badge_index": 0}

    def test_from_dict_ignores_unknown_keys(self):
        """Extra keys in the dict (e.g. future schema additions) don't crash."""
        d = _immediate().to_dict()
        d["future_field"] = "some_value"
        node = GoalNode.from_dict(d)
        assert node.goal_id == d["goal_id"]

    def test_to_dict_is_json_serialisable(self):
        """Serialised dict must be JSON-safe (no datetime, no custom objects)."""
        import json
        d = _strategic().to_dict()
        # Should not raise
        json.dumps(d)


# ---------------------------------------------------------------------------
# TestGoalNodeDefaults
# ---------------------------------------------------------------------------

class TestGoalNodeDefaults:
    def test_created_at_is_positive(self):
        before = time.time()
        g = GoalNode("x", "desc", "immediate")
        after = time.time()
        assert before <= g.created_at <= after

    def test_directive_defaults_none(self):
        g = GoalNode("x", "desc", "tactical")
        assert g.directive is None

    def test_metadata_defaults_empty_dict(self):
        g = GoalNode("x", "desc", "strategic")
        assert g.metadata == {}

    def test_parent_id_defaults_none(self):
        g = GoalNode("x", "desc", "immediate")
        assert g.parent_id is None

    def test_push_reason_defaults_empty_string(self):
        g = GoalNode("x", "desc", "immediate")
        assert g.push_reason == ""

    def test_completion_condition_defaults_empty_string(self):
        g = GoalNode("x", "desc", "tactical")
        assert g.completion_condition == ""

    def test_metadata_instances_are_independent(self):
        """Two GoalNodes must not share the same metadata dict (mutable default)."""
        a = GoalNode("a", "A", "immediate")
        b = GoalNode("b", "B", "immediate")
        a.metadata["key"] = "value"
        assert "key" not in b.metadata


# ---------------------------------------------------------------------------
# TestGoalNodeValidation
# ---------------------------------------------------------------------------

class TestGoalNodeValidation:
    def test_invalid_goal_type_raises(self):
        with pytest.raises(ValueError, match="goal_type"):
            GoalNode("x", "desc", "invalid_type")

    def test_valid_goal_types_accepted(self):
        for gt in ("strategic", "tactical", "immediate"):
            g = GoalNode("x", "desc", gt)
            assert g.goal_type == gt


# ---------------------------------------------------------------------------
# TestStackPush
# ---------------------------------------------------------------------------

class TestStackPush:
    def test_push_onto_empty(self):
        g = _immediate()
        result = stack_push([], g)
        assert result == [g]

    def test_push_prepends(self):
        existing = _tactical()
        new = _immediate()
        result = stack_push([existing], new)
        assert result[0] is new
        assert result[1] is existing

    def test_push_does_not_mutate_original(self):
        original = [_tactical()]
        stack_push(original, _immediate())
        assert len(original) == 1

    def test_push_increases_length_by_one(self):
        stack = [_strategic(), _tactical()]
        result = stack_push(stack, _immediate())
        assert len(result) == 3


# ---------------------------------------------------------------------------
# TestStackPop
# ---------------------------------------------------------------------------

class TestStackPop:
    def test_pop_returns_stack0(self):
        a, b = _immediate("a"), _tactical("b")
        popped, remaining = stack_pop([a, b])
        assert popped is a
        assert remaining == [b]

    def test_pop_empty_returns_none_and_empty(self):
        popped, remaining = stack_pop([])
        assert popped is None
        assert remaining == []

    def test_pop_single_element(self):
        g = _immediate()
        popped, remaining = stack_pop([g])
        assert popped is g
        assert remaining == []

    def test_pop_does_not_mutate_original(self):
        original = [_immediate(), _tactical()]
        stack_pop(original)
        assert len(original) == 2


# ---------------------------------------------------------------------------
# TestStackReplace
# ---------------------------------------------------------------------------

class TestStackReplace:
    def test_replace_swaps_stack0(self):
        old = _immediate("old")
        parent = _tactical()
        new = _immediate("new")
        result = stack_replace([old, parent], new)
        assert result[0] is new
        assert result[1] is parent

    def test_replace_preserves_rest(self):
        a, b, c = _immediate("a"), _tactical("b"), _strategic("c")
        new = _immediate("new")
        result = stack_replace([a, b, c], new)
        assert len(result) == 3
        assert result[1] is b
        assert result[2] is c

    def test_replace_on_empty_stack(self):
        new = _immediate()
        result = stack_replace([], new)
        assert result == [new]

    def test_replace_same_length(self):
        stack = [_immediate(), _tactical(), _strategic()]
        new = _immediate("replacement")
        result = stack_replace(stack, new)
        assert len(result) == len(stack)

    def test_replace_does_not_mutate_original(self):
        original = [_immediate("old"), _tactical()]
        stack_replace(original, _immediate("new"))
        assert original[0].goal_id == "old"


# ---------------------------------------------------------------------------
# TestStackPeek
# ---------------------------------------------------------------------------

class TestStackPeek:
    def test_peek_returns_stack0(self):
        a = _immediate()
        assert stack_peek([a, _tactical()]) is a

    def test_peek_empty_returns_none(self):
        assert stack_peek([]) is None

    def test_peek_does_not_mutate(self):
        stack = [_immediate(), _tactical()]
        stack_peek(stack)
        assert len(stack) == 2


# ---------------------------------------------------------------------------
# TestStackSummary
# ---------------------------------------------------------------------------

class TestStackSummary:
    def test_empty_stack(self):
        assert stack_summary([]) == "(empty)"

    def test_single_element(self):
        g = GoalNode("id", "Walk north", "immediate")
        summary = stack_summary([g])
        assert "[I]Walk north" in summary

    def test_three_level_stack_order(self):
        """Summary is ordered strategic → tactical → immediate (left-to-right)."""
        imm = GoalNode("i", "Enter gym", "immediate")
        tac = GoalNode("t", "Reach Rustboro", "tactical")
        strat = GoalNode("s", "Defeat Roxanne", "strategic")
        # Stack stored [immediate, tactical, strategic] (stack[0] = most immediate)
        summary = stack_summary([imm, tac, strat])
        assert summary.index("[S]") < summary.index("[T]") < summary.index("[I]")

    def test_type_prefixes(self):
        stack = [
            GoalNode("i", "immediate goal", "immediate"),
            GoalNode("t", "tactical goal", "tactical"),
            GoalNode("s", "strategic goal", "strategic"),
        ]
        summary = stack_summary(stack)
        assert "[I]" in summary
        assert "[T]" in summary
        assert "[S]" in summary
