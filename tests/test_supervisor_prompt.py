"""
tests/test_supervisor_prompt.py — Phase 3 tests for the Supervisor LLM prompt
templates and JSON schema parsing.

Covers:
  TestUserPromptRendering        — SUPERVISOR_USER_TEMPLATE renders without error
  TestSystemPromptContainsAllOps — SUPERVISOR_SYSTEM_PROMPT contains required fields
  TestCallSupervisorLLMValidJson — well-formed JSON round-trips through _call_supervisor_llm
  TestCallSupervisorLLMMarkdownFences — markdown-fenced JSON is stripped and parsed
  TestCallSupervisorLLMInvalidOperation — unknown op → fallback to CONTINUE
  TestCallSupervisorLLMNetworkError — VLM exception → safe CONTINUE fallback
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.graph.goal_stack import GoalNode
from agent.graph.nodes.executive_supervisor import (
    SUPERVISOR_SYSTEM_PROMPT,
    SUPERVISOR_USER_TEMPLATE,
    _call_supervisor_llm,
    _build_game_summary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODULE = "agent.graph.nodes.executive_supervisor"


def _make_goal(goal_id: str = "test_goal", **kwargs) -> GoalNode:
    return GoalNode(
        goal_id=goal_id,
        description="Walk north through Route 104",
        goal_type="immediate",
        completion_condition="Player enters PETALBURG_WOODS",
        **kwargs,
    )


def _game_summary() -> dict:
    return _build_game_summary(
        {
            "player": {"location": "ROUTE_104_SOUTH", "position": {"x": 10, "y": 20}},
            "game": {"badges": 0, "in_battle": False},
            "party": [{"name": "Treecko", "current_hp": 22, "max_hp": 25}],
        },
        {"step_count": 5, "last_node_fired": "nav_bot", "context": "navigation"},
    )


def _make_vlm(return_value: str) -> MagicMock:
    vlm = MagicMock()
    vlm.get_json_query.return_value = return_value
    return vlm


# ---------------------------------------------------------------------------
# TestUserPromptRendering
# ---------------------------------------------------------------------------


class TestUserPromptRendering:
    def test_renders_without_key_error(self):
        goal = _make_goal()
        rendered = SUPERVISOR_USER_TEMPLATE.format(
            stack_repr="[I] Walk north",
            goal_id=goal.goal_id,
            goal_type=goal.goal_type,
            goal_description=goal.description,
            completion_condition=goal.completion_condition or "(none)",
            dialogue_context="(none)",
            battle_context="(none)",
            **_game_summary(),
        )
        # No KeyError → we reach this assert
        assert isinstance(rendered, str)

    def test_rendered_prompt_contains_goal_id(self):
        goal = _make_goal(goal_id="traverse_route_104")
        rendered = SUPERVISOR_USER_TEMPLATE.format(
            stack_repr="[I] Walk north",
            goal_id=goal.goal_id,
            goal_type=goal.goal_type,
            goal_description=goal.description,
            completion_condition=goal.completion_condition or "(none)",
            dialogue_context="(none)",
            battle_context="(none)",
            **_game_summary(),
        )
        assert "traverse_route_104" in rendered

    def test_rendered_prompt_contains_location(self):
        goal = _make_goal()
        rendered = SUPERVISOR_USER_TEMPLATE.format(
            stack_repr="[I] Walk north",
            goal_id=goal.goal_id,
            goal_type=goal.goal_type,
            goal_description=goal.description,
            completion_condition=goal.completion_condition or "(none)",
            dialogue_context="(none)",
            battle_context="(none)",
            **_game_summary(),
        )
        assert "ROUTE_104_SOUTH" in rendered

    def test_rendered_prompt_contains_stack_repr(self):
        goal = _make_goal()
        rendered = SUPERVISOR_USER_TEMPLATE.format(
            stack_repr="[I] unique_stack_repr_marker",
            goal_id=goal.goal_id,
            goal_type=goal.goal_type,
            goal_description=goal.description,
            completion_condition=goal.completion_condition or "(none)",
            dialogue_context="(none)",
            battle_context="(none)",
            **_game_summary(),
        )
        assert "unique_stack_repr_marker" in rendered


# ---------------------------------------------------------------------------
# TestSystemPromptContainsAllOps
# ---------------------------------------------------------------------------


class TestSystemPromptContainsAllOps:
    def test_contains_pop(self):
        assert "POP" in SUPERVISOR_SYSTEM_PROMPT

    def test_contains_continue(self):
        assert "CONTINUE" in SUPERVISOR_SYSTEM_PROMPT

    def test_contains_push(self):
        assert "PUSH" in SUPERVISOR_SYSTEM_PROMPT

    def test_contains_replace(self):
        assert "REPLACE" in SUPERVISOR_SYSTEM_PROMPT

    def test_contains_directive(self):
        assert "directive" in SUPERVISOR_SYSTEM_PROMPT

    def test_contains_goal_id_key(self):
        assert "goal_id" in SUPERVISOR_SYSTEM_PROMPT

    def test_contains_goal_type_key(self):
        assert "goal_type" in SUPERVISOR_SYSTEM_PROMPT

    def test_contains_completion_condition_key(self):
        assert "completion_condition" in SUPERVISOR_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# TestCallSupervisorLLMValidJson
# ---------------------------------------------------------------------------


class TestCallSupervisorLLMValidJson:
    def test_valid_continue_response(self):
        vlm = _make_vlm('{"operation": "CONTINUE", "reasoning": "ok", "new_goals": []}')
        result = _call_supervisor_llm(
            vlm, _make_goal(), "(none)", "(none)", _game_summary(), "[I] Walk north"
        )
        assert result["operation"] == "CONTINUE"

    def test_valid_pop_response(self):
        vlm = _make_vlm('{"operation": "POP", "reasoning": "done", "new_goals": []}')
        result = _call_supervisor_llm(
            vlm, _make_goal(), "(none)", "(none)", _game_summary(), "[I] Walk north"
        )
        assert result["operation"] == "POP"

    def test_no_exception_raised(self):
        vlm = _make_vlm('{"operation": "CONTINUE", "reasoning": "ok", "new_goals": []}')
        # Should not raise
        _call_supervisor_llm(
            vlm, _make_goal(), "(none)", "(none)", _game_summary(), "[I] Walk north"
        )

    def test_get_json_query_called_with_system_prompt(self):
        vlm = _make_vlm('{"operation": "CONTINUE", "reasoning": "ok", "new_goals": []}')
        _call_supervisor_llm(
            vlm, _make_goal(), "(none)", "(none)", _game_summary(), "[I] Walk north"
        )
        vlm.get_json_query.assert_called_once()
        call_args = vlm.get_json_query.call_args
        # First positional arg is system_prompt
        assert call_args[0][0] == SUPERVISOR_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# TestCallSupervisorLLMMarkdownFences
# ---------------------------------------------------------------------------


class TestCallSupervisorLLMMarkdownFences:
    def test_json_fences_stripped(self):
        payload = json.dumps({"operation": "POP", "reasoning": "done", "new_goals": []})
        vlm = _make_vlm(f"```json\n{payload}\n```")
        result = _call_supervisor_llm(
            vlm, _make_goal(), "(none)", "(none)", _game_summary(), "[I] Walk north"
        )
        assert result["operation"] == "POP"

    def test_plain_json_without_fences(self):
        payload = json.dumps({"operation": "PUSH", "reasoning": "urgent",
                              "new_goals": [{"goal_id": "g", "description": "d",
                                             "goal_type": "immediate", "parent_id": None,
                                             "completion_condition": "x",
                                             "directive": {"action": "NAVIGATE",
                                                           "goal_location": "OLDALE_TOWN",
                                                           "goal_coords": None,
                                                           "should_interact": False,
                                                           "npc_coords": None,
                                                           "description": "go"},
                                             "metadata": {}}]})
        vlm = _make_vlm(payload)
        result = _call_supervisor_llm(
            vlm, _make_goal(), "(none)", "(none)", _game_summary(), "[I] Walk north"
        )
        assert result["operation"] == "PUSH"


# ---------------------------------------------------------------------------
# TestCallSupervisorLLMInvalidOperation
# ---------------------------------------------------------------------------


class TestCallSupervisorLLMInvalidOperation:
    def test_unknown_op_falls_back_to_continue(self):
        vlm = _make_vlm('{"operation": "DANCE", "reasoning": "?", "new_goals": []}')
        result = _call_supervisor_llm(
            vlm, _make_goal(), "(none)", "(none)", _game_summary(), "[I] Walk north"
        )
        assert result["operation"] == "CONTINUE"

    def test_reasoning_mentions_parse_error(self):
        vlm = _make_vlm('{"operation": "MOONWALK"}')
        result = _call_supervisor_llm(
            vlm, _make_goal(), "(none)", "(none)", _game_summary(), "[I] Walk north"
        )
        assert "parse_error" in result.get("reasoning", "")

    def test_new_goals_empty_on_fallback(self):
        vlm = _make_vlm('{"operation": "INVALID_OP"}')
        result = _call_supervisor_llm(
            vlm, _make_goal(), "(none)", "(none)", _game_summary(), "[I] Walk north"
        )
        assert result.get("new_goals") == []


# ---------------------------------------------------------------------------
# TestCallSupervisorLLMNetworkError
# ---------------------------------------------------------------------------


class TestCallSupervisorLLMNetworkError:
    def test_network_error_returns_continue(self):
        vlm = MagicMock()
        vlm.get_json_query.side_effect = Exception("network error")
        result = _call_supervisor_llm(
            vlm, _make_goal(), "(none)", "(none)", _game_summary(), "[I] Walk north"
        )
        assert result["operation"] == "CONTINUE"

    def test_network_error_reasoning_contains_error_message(self):
        vlm = MagicMock()
        vlm.get_json_query.side_effect = Exception("timeout after 15s")
        result = _call_supervisor_llm(
            vlm, _make_goal(), "(none)", "(none)", _game_summary(), "[I] Walk north"
        )
        assert "timeout after 15s" in result["reasoning"]

    def test_network_error_does_not_propagate(self):
        vlm = MagicMock()
        vlm.get_json_query.side_effect = RuntimeError("connection refused")
        # Must not raise
        _call_supervisor_llm(
            vlm, _make_goal(), "(none)", "(none)", _game_summary(), "[I] Walk north"
        )

    def test_vlm_none_returns_continue(self):
        result = _call_supervisor_llm(
            None, _make_goal(), "(none)", "(none)", _game_summary(), "[I] Walk north"
        )
        assert result["operation"] == "CONTINUE"
