"""
tests/test_htn_bootstrap.py — Phase 4 automated tests for _bootstrap_stack,
_expand_strategic_goal, and all associated helpers in executive_supervisor.py.

Run with:  pytest tests/test_htn_bootstrap.py -v
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.graph.goal_stack import GoalNode
from agent.graph.nodes.executive_supervisor import (
    _HTN_SYSTEM_PROMPT,
    _bootstrap_stack,
    _build_htn_generation_prompt,
    _build_rag_query,
    _count_badges,
    _expand_strategic_goal,
    _get_current_location,
    _get_effective_progress_index,
    _get_last_completed_milestone,
    _milestone_fallback_stack,
)
from agent.objective_manager import MILESTONE_PROGRESSION

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_IMMEDIATE_GOAL_DICT = {
    "goal_id": "navigate_route_104_south",
    "description": "Walk north through Route 104 South",
    "goal_type": "immediate",
    "parent_id": "reach_rustboro",
    "completion_condition": "Player enters PETALBURG_WOODS",
    "directive": {
        "action": "NAVIGATE",
        "goal_coords": None,
        "goal_location": "PETALBURG_WOODS",
        "should_interact": False,
        "npc_coords": None,
        "description": "Head north toward Petalburg Woods",
    },
    "metadata": {},
}

_TACTICAL_GOAL_DICT = {
    "goal_id": "reach_rustboro",
    "description": "Travel from Petalburg City to Rustboro City",
    "goal_type": "tactical",
    "parent_id": "earn_stone_badge",
    "completion_condition": "Player is in RUSTBORO_CITY",
    "directive": None,
    "metadata": {},
}

_STRATEGIC_GOAL_DICT = {
    "goal_id": "earn_stone_badge",
    "description": "Earn the Stone Badge from Roxanne",
    "goal_type": "strategic",
    "parent_id": None,
    "completion_condition": "Player has 1 badge",
    "directive": None,
    "metadata": {"required_badge_count": 1},
}

_VALID_HTN_RESPONSE = json.dumps({
    "goals": [_IMMEDIATE_GOAL_DICT, _TACTICAL_GOAL_DICT, _STRATEGIC_GOAL_DICT]
})

_ROUTE_102_MILESTONES = {
    "GAME_RUNNING": True,
    "PLAYER_NAME_SET": True,
    "INTRO_CUTSCENE_COMPLETE": True,
    "LITTLEROOT_TOWN": True,
    "PLAYER_HOUSE_ENTERED": True,
    "PLAYER_BEDROOM": True,
    "RIVAL_HOUSE": True,
    "RIVAL_BEDROOM": True,
    "ROUTE_101": True,
    "STARTER_CHOSEN": True,
    "BIRCH_LAB_VISITED": True,
    "OLDALE_TOWN": True,
    "ROUTE_103": True,
    "RIVAL_BATTLE_1": True,
    "RECEIVED_POKEDEX": True,
    "ROUTE_102": True,
    # PETALBURG_CITY not yet done
}

_PETALBURG_STATE_DATA = {
    "player": {"location": "PETALBURG_CITY", "position": {"x": 10, "y": 20}},
    "game": {"badges": 0, "in_battle": False},
    "party": [{"name": "Treecko", "current_hp": 25, "max_hp": 30}],
    "milestones": _ROUTE_102_MILESTONES,
}


def _make_vlm(response: str) -> MagicMock:
    vlm = MagicMock()
    vlm.get_json_query.return_value = response
    return vlm


def _make_walkthrough_db(chunks: list) -> MagicMock:
    db = MagicMock()
    db.query.return_value = chunks
    return db


# ---------------------------------------------------------------------------
# TestBootstrapEmpty — real RAG + LLM path
# ---------------------------------------------------------------------------

class TestBootstrapEmpty:
    """_bootstrap_stack with mock walkthrough_db + vlm producing a valid HTN."""

    def _run(self):
        chunks = [
            {"text": "Head to Petalburg City after completing Route 102.", "metadata": {}, "distance": 0.3},
            {"text": "Enter Petalburg City Gym to meet your father Norman.", "metadata": {}, "distance": 0.4},
            {"text": "Travel through Route 104 South after meeting Norman.", "metadata": {}, "distance": 0.5},
        ]
        db = _make_walkthrough_db(chunks)
        # Use a response that correctly targets PETALBURG_CITY_GYM — the
        # mandatory next milestone when last_completed=ROUTE_102 and the
        # player is already in PETALBURG_CITY.
        vlm = _make_vlm(_gym_response())
        return _bootstrap_stack(_PETALBURG_CITY_STATE_DATA, db, vlm)

    def test_returns_non_empty_list(self):
        stack = self._run()
        assert isinstance(stack, list)
        assert len(stack) > 0

    def test_stack_0_is_immediate(self):
        stack = self._run()
        assert stack[0].goal_type == "immediate"

    def test_stack_0_has_directive(self):
        stack = self._run()
        assert stack[0].directive is not None

    def test_stack_last_is_strategic(self):
        stack = self._run()
        assert stack[-1].goal_type == "strategic"

    def test_goal_nodes_correct_type(self):
        stack = self._run()
        for g in stack:
            assert isinstance(g, GoalNode)


# ---------------------------------------------------------------------------
# TestBootstrapFallback — walkthrough_db=None triggers milestone fallback
# ---------------------------------------------------------------------------

class TestBootstrapFallback:
    """When walkthrough_db is None, _bootstrap_stack falls back to milestones."""

    def _run(self):
        vlm = _make_vlm(_VALID_HTN_RESPONSE)
        return _bootstrap_stack(_PETALBURG_STATE_DATA, None, vlm)

    def test_returns_non_empty_stack(self):
        stack = self._run()
        assert len(stack) > 0

    def test_stack_0_is_immediate(self):
        stack = self._run()
        assert stack[0].goal_type == "immediate"

    def test_no_exception(self):
        # Should not raise regardless of state
        _bootstrap_stack({}, None, None)

    def test_vlm_not_called_when_no_db(self):
        vlm = _make_vlm(_VALID_HTN_RESPONSE)
        _bootstrap_stack(_PETALBURG_STATE_DATA, None, vlm)
        vlm.get_json_query.assert_not_called()


# ---------------------------------------------------------------------------
# TestBootstrapLLMParseError — invalid JSON triggers milestone fallback
# ---------------------------------------------------------------------------

class TestBootstrapLLMParseError:
    """When the LLM returns invalid JSON, _bootstrap_stack falls back gracefully."""

    def test_invalid_json_falls_back(self):
        chunks = [{"text": "Some walkthrough text.", "metadata": {}, "distance": 0.2}]
        db = _make_walkthrough_db(chunks)
        vlm = _make_vlm("this is not valid json {{{")
        stack = _bootstrap_stack(_PETALBURG_STATE_DATA, db, vlm)
        assert isinstance(stack, list)
        # No crash

    def test_missing_goals_key_falls_back(self):
        chunks = [{"text": "Some walkthrough text.", "metadata": {}, "distance": 0.2}]
        db = _make_walkthrough_db(chunks)
        vlm = _make_vlm(json.dumps({"something_else": []}))
        stack = _bootstrap_stack(_PETALBURG_STATE_DATA, db, vlm)
        assert isinstance(stack, list)

    def test_no_immediate_goal_falls_back(self):
        """HTN missing an immediate goal must trigger fallback (assertion fails)."""
        chunks = [{"text": "Some text.", "metadata": {}, "distance": 0.1}]
        db = _make_walkthrough_db(chunks)
        # Only strategic and tactical, no immediate
        bad_response = json.dumps({"goals": [_TACTICAL_GOAL_DICT, _STRATEGIC_GOAL_DICT]})
        vlm = _make_vlm(bad_response)
        stack = _bootstrap_stack(_PETALBURG_STATE_DATA, db, vlm)
        # Falls back to milestone stack; still non-empty
        assert isinstance(stack, list)

    def test_returns_non_empty_on_parse_error(self):
        chunks = [{"text": "Text.", "metadata": {}, "distance": 0.1}]
        db = _make_walkthrough_db(chunks)
        vlm = _make_vlm("")
        stack = _bootstrap_stack(_PETALBURG_STATE_DATA, db, vlm)
        # Milestone fallback for PETALBURG_STATE_DATA (ROUTE_102 done, PETALBURG_CITY not)
        assert len(stack) >= 1


# ---------------------------------------------------------------------------
# TestGetLastCompletedMilestone
# ---------------------------------------------------------------------------

class TestGetLastCompletedMilestone:
    def test_route_102_completed(self):
        result = _get_last_completed_milestone(_ROUTE_102_MILESTONES)
        assert result == "ROUTE_102"

    def test_nothing_completed(self):
        result = _get_last_completed_milestone({})
        assert result == "GAME_RUNNING"

    def test_only_game_running(self):
        result = _get_last_completed_milestone({"GAME_RUNNING": True})
        assert result == "GAME_RUNNING"

    def test_all_completed_returns_last(self):
        all_done = {entry["milestone"]: True for entry in MILESTONE_PROGRESSION}
        result = _get_last_completed_milestone(all_done)
        assert result == MILESTONE_PROGRESSION[-1]["milestone"]

    def test_false_values_not_counted(self):
        milestones = {entry["milestone"]: False for entry in MILESTONE_PROGRESSION}
        milestones["ROUTE_101"] = True
        result = _get_last_completed_milestone(milestones)
        assert result == "ROUTE_101"


# ---------------------------------------------------------------------------
# TestMilestoneFallbackStack
# ---------------------------------------------------------------------------

class TestMilestoneFallbackStack:
    def test_route_102_done_targets_petalburg(self):
        stack = _milestone_fallback_stack(_ROUTE_102_MILESTONES, _PETALBURG_STATE_DATA)
        assert len(stack) >= 1
        assert stack[0].goal_type == "immediate"
        # Next incomplete after ROUTE_102 is PETALBURG_CITY
        assert stack[0].directive is not None
        assert stack[0].directive["goal_location"] == "PETALBURG_CITY"

    def test_empty_milestones_targets_first(self):
        stack = _milestone_fallback_stack({}, {})
        assert len(stack) >= 1
        # GAME_RUNNING has no target_location, so directive may be None
        assert isinstance(stack[0], GoalNode)

    def test_all_done_returns_empty(self):
        all_done = {entry["milestone"]: True for entry in MILESTONE_PROGRESSION}
        stack = _milestone_fallback_stack(all_done, {})
        assert stack == []

    def test_returns_goal_nodes(self):
        stack = _milestone_fallback_stack(_ROUTE_102_MILESTONES, _PETALBURG_STATE_DATA)
        for g in stack:
            assert isinstance(g, GoalNode)


# ---------------------------------------------------------------------------
# TestRAGBootstrapQuery — verify RAG is called with sensible query
# ---------------------------------------------------------------------------

class TestRAGBootstrapQuery:
    def test_rag_query_contains_last_milestone(self):
        db = _make_walkthrough_db([])
        vlm = _make_vlm(_VALID_HTN_RESPONSE)
        _bootstrap_stack(_PETALBURG_STATE_DATA, db, vlm)
        call_args = db.query.call_args
        query_text = call_args[0][0] if call_args[0] else call_args[1].get("query_text", "")
        # Query uses natural language; check it contains the location natural name
        # and forward-looking milestone content (not raw enum names)
        assert "Petalburg" in query_text or "Route" in query_text

    def test_rag_query_contains_location(self):
        db = _make_walkthrough_db([])
        vlm = _make_vlm(_VALID_HTN_RESPONSE)
        _bootstrap_stack(_PETALBURG_STATE_DATA, db, vlm)
        call_args = db.query.call_args
        query_text = call_args[0][0] if call_args[0] else call_args[1].get("query_text", "")
        # Location converted to natural language: PETALBURG_CITY → "Petalburg City"
        assert "Petalburg City" in query_text

    def test_rag_not_called_when_db_is_none(self):
        # When db is None, we should NOT crash and should not call db.query
        stack = _bootstrap_stack(_PETALBURG_STATE_DATA, None, None)
        assert isinstance(stack, list)


# ---------------------------------------------------------------------------
# TestHTNGenerationPromptStructure
# ---------------------------------------------------------------------------

class TestHTNGenerationPromptStructure:
    def test_prompt_contains_context(self):
        prompt = _build_htn_generation_prompt("Some walkthrough text.", "PETALBURG_CITY", "ROUTE_102", 0)
        assert "Some walkthrough text." in prompt

    def test_prompt_contains_location(self):
        prompt = _build_htn_generation_prompt("ctx", "PETALBURG_CITY", "ROUTE_102", 0)
        assert "PETALBURG_CITY" in prompt

    def test_prompt_contains_badge_count(self):
        prompt = _build_htn_generation_prompt("ctx", "PETALBURG_CITY", "ROUTE_102", 2)
        assert "2" in prompt

    def test_prompt_contains_last_milestone(self):
        prompt = _build_htn_generation_prompt("ctx", "PETALBURG_CITY", "ROUTE_102", 0)
        assert "ROUTE_102" in prompt

    def test_system_prompt_mentions_immediate(self):
        assert "immediate" in _HTN_SYSTEM_PROMPT

    def test_system_prompt_mentions_tactical(self):
        assert "tactical" in _HTN_SYSTEM_PROMPT

    def test_system_prompt_mentions_strategic(self):
        assert "strategic" in _HTN_SYSTEM_PROMPT

    def test_system_prompt_requires_directive(self):
        assert "directive" in _HTN_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# TestExpandStrategicGoal
# ---------------------------------------------------------------------------

class TestExpandStrategicGoal:
    def _make_parent(self) -> GoalNode:
        return GoalNode.from_dict(_STRATEGIC_GOAL_DICT)

    def _make_tactical_response(self, parent_id: str = "earn_stone_badge") -> str:
        return json.dumps({"goals": [
            {
                "goal_id": "reach_rustboro_city",
                "description": "Travel to Rustboro City",
                "goal_type": "tactical",
                "parent_id": parent_id,
                "completion_condition": "Player is in RUSTBORO_CITY",
                "directive": None,
                "metadata": {},
            },
            {
                "goal_id": "enter_rustboro_gym",
                "description": "Enter Rustboro City Gym",
                "goal_type": "tactical",
                "parent_id": parent_id,
                "completion_condition": "Player is in RUSTBORO_CITY_GYM",
                "directive": None,
                "metadata": {},
            },
        ]})

    def test_returns_list_of_goal_nodes(self):
        parent = self._make_parent()
        db = _make_walkthrough_db([{"text": "Head north to Rustboro.", "metadata": {}, "distance": 0.2}])
        vlm = _make_vlm(self._make_tactical_response())
        result = _expand_strategic_goal(parent, {}, db, vlm)
        assert isinstance(result, list)
        assert len(result) > 0
        for g in result:
            assert isinstance(g, GoalNode)

    def test_returned_nodes_have_correct_parent_id(self):
        parent = self._make_parent()
        db = _make_walkthrough_db([{"text": "Some text.", "metadata": {}, "distance": 0.1}])
        vlm = _make_vlm(self._make_tactical_response(parent_id=parent.goal_id))
        result = _expand_strategic_goal(parent, {}, db, vlm)
        for g in result:
            assert g.parent_id == parent.goal_id

    def test_returns_empty_when_db_is_none(self):
        parent = self._make_parent()
        result = _expand_strategic_goal(parent, {}, None, None)
        assert result == []

    def test_returns_empty_on_parse_error(self):
        parent = self._make_parent()
        db = _make_walkthrough_db([{"text": "Some text.", "metadata": {}, "distance": 0.1}])
        vlm = _make_vlm("not valid json")
        result = _expand_strategic_goal(parent, {}, db, vlm)
        assert result == []

    def test_rag_query_contains_parent_description(self):
        parent = self._make_parent()
        db = _make_walkthrough_db([])
        vlm = _make_vlm(self._make_tactical_response())
        _expand_strategic_goal(parent, {}, db, vlm)
        call_args = db.query.call_args
        query_text = call_args[0][0] if call_args[0] else call_args[1].get("query_text", "")
        assert "Roxanne" in query_text or "Stone Badge" in query_text


# ---------------------------------------------------------------------------
# TestCountBadges / TestGetCurrentLocation (helper coverage)
# ---------------------------------------------------------------------------

class TestCountBadges:
    def test_int_badge_count(self):
        assert _count_badges({"game": {"badges": 3}}) == 3

    def test_dict_badge_flags(self):
        badges = {"stone": True, "knuckle": True, "dynamo": False, "heat": False}
        assert _count_badges({"game": {"badges": badges}}) == 2

    def test_missing_game_key(self):
        assert _count_badges({}) == 0

    def test_missing_badges_key(self):
        assert _count_badges({"game": {}}) == 0


class TestGetCurrentLocation:
    def test_returns_location_string(self):
        state = {"player": {"location": "RUSTBORO_CITY"}}
        assert _get_current_location(state) == "RUSTBORO_CITY"

    def test_returns_unknown_when_missing(self):
        assert _get_current_location({}) == "Unknown"

    def test_returns_unknown_when_none(self):
        assert _get_current_location({"player": {"location": None}}) == "Unknown"


# ---------------------------------------------------------------------------
# _PETALBURG_CITY milestones — used by bootstrap-fix tests below
# ---------------------------------------------------------------------------

_PETALBURG_CITY_MILESTONES = {
    **_ROUTE_102_MILESTONES,
    "PETALBURG_CITY": True,   # arrived; DAD_FIRST_MEETING is next
}

_PETALBURG_CITY_STATE_DATA = {
    "player": {"location": "PETALBURG_CITY", "position": {"x": 10, "y": 20}},
    "game": {"badges": 0, "in_battle": False},
    "party": [{"name": "Treecko", "current_hp": 25, "max_hp": 30}],
    "milestones": _PETALBURG_CITY_MILESTONES,
}


def _gym_response() -> str:
    """LLM response that correctly targets PETALBURG_CITY_GYM."""
    return json.dumps({"goals": [
        {
            "goal_id": "enter_petalburg_gym",
            "description": "Enter Petalburg City Gym to meet Dad",
            "goal_type": "immediate",
            "parent_id": "meet_dad_norman",
            "completion_condition": "milestone DAD_FIRST_MEETING completed",
            "directive": {
                "action": "NAVIGATE",
                "goal_coords": None,
                "goal_location": "PETALBURG_CITY_GYM",
                "should_interact": False,
                "npc_coords": None,
                "description": "Go to Petalburg City Gym",
            },
            "metadata": {},
        },
        {
            "goal_id": "meet_dad_norman",
            "description": "Meet Norman in the gym",
            "goal_type": "tactical",
            "parent_id": "earn_stone_badge",
            "completion_condition": "DAD_FIRST_MEETING completed",
            "directive": None,
            "metadata": {},
        },
        {
            "goal_id": "earn_stone_badge",
            "description": "Earn the Stone Badge",
            "goal_type": "strategic",
            "parent_id": None,
            "completion_condition": "Player has 1 badge",
            "directive": None,
            "metadata": {},
        },
    ]})


def _route104_response() -> str:
    """LLM response that wrongly skips the gym and targets ROUTE_104_SOUTH."""
    return json.dumps({"goals": [
        {
            "goal_id": "head_to_route_104",
            "description": "Walk west out of Petalburg City to Route 104 South",
            "goal_type": "immediate",
            "parent_id": "reach_rustboro",
            "completion_condition": "Player is on ROUTE_104_SOUTH",
            "directive": {
                "action": "NAVIGATE",
                "goal_coords": None,
                "goal_location": "ROUTE_104_SOUTH",
                "should_interact": False,
                "npc_coords": None,
                "description": "Head to Route 104 South",
            },
            "metadata": {},
        },
        {
            "goal_id": "reach_rustboro",
            "description": "Reach Rustboro City",
            "goal_type": "strategic",
            "parent_id": None,
            "completion_condition": "Player is in RUSTBORO_CITY",
            "directive": None,
            "metadata": {},
        },
    ]})


# ---------------------------------------------------------------------------
# TestBootstrapNextMilestonePrompt
# ---------------------------------------------------------------------------

class TestBootstrapNextMilestonePrompt:
    """_build_htn_generation_prompt includes a MANDATORY section when
    next_milestone is provided."""

    def _next_ms(self) -> dict:
        return {
            "milestone": "DAD_FIRST_MEETING",
            "description": "Enter gym to meet Dad",
            "target_location": "PETALBURG_CITY_GYM",
            "completion_type": "dialogue",
        }

    def test_mandatory_section_present(self):
        prompt = _build_htn_generation_prompt(
            "ctx", "PETALBURG_CITY", "PETALBURG_CITY", 0,
            next_milestone=self._next_ms(),
        )
        assert "MANDATORY" in prompt

    def test_milestone_id_in_prompt(self):
        prompt = _build_htn_generation_prompt(
            "ctx", "PETALBURG_CITY", "PETALBURG_CITY", 0,
            next_milestone=self._next_ms(),
        )
        assert "DAD_FIRST_MEETING" in prompt

    def test_target_location_in_prompt(self):
        prompt = _build_htn_generation_prompt(
            "ctx", "PETALBURG_CITY", "PETALBURG_CITY", 0,
            next_milestone=self._next_ms(),
        )
        assert "PETALBURG_CITY_GYM" in prompt

    def test_cannot_skip_milestone_warning_in_prompt(self):
        prompt = _build_htn_generation_prompt(
            "ctx", "PETALBURG_CITY", "PETALBURG_CITY", 0,
            next_milestone=self._next_ms(),
        )
        assert "skip" in prompt.lower() or "cannot" in prompt.lower()

    def test_no_mandatory_section_without_next_milestone(self):
        prompt = _build_htn_generation_prompt(
            "ctx", "PETALBURG_CITY", "PETALBURG_CITY", 0,
        )
        assert "MANDATORY" not in prompt

    def test_none_target_location_uses_placeholder(self):
        """next_milestone with no target_location renders gracefully."""
        ms = {"milestone": "GYM_EXPLANATION", "description": "Watch Wally tutorial",
              "target_location": None, "completion_type": "dialogue"}
        prompt = _build_htn_generation_prompt(
            "ctx", "PETALBURG_CITY_GYM", "GYM_EXPLANATION", 0,
            next_milestone=ms,
        )
        assert "GYM_EXPLANATION" in prompt
        assert "MANDATORY" in prompt


# ---------------------------------------------------------------------------
# TestBootstrapValidationFallback
# ---------------------------------------------------------------------------

class TestBootstrapValidationFallback:
    """When the LLM ignores the MANDATORY section and targets the wrong location,
    _bootstrap_stack falls back to the deterministic milestone stack."""

    def _chunks(self):
        return [
            {"text": "Enter Petalburg City Gym to meet your father Norman.",
             "metadata": {}, "distance": 0.2},
        ]

    def test_wrong_target_triggers_fallback(self):
        """LLM returns ROUTE_104_SOUTH instead of PETALBURG_CITY_GYM — must fall back."""
        db = _make_walkthrough_db(self._chunks())
        vlm = _make_vlm(_route104_response())
        stack = _bootstrap_stack(_PETALBURG_CITY_STATE_DATA, db, vlm)
        # Fallback produces milestone stack targeting PETALBURG_CITY_GYM
        assert len(stack) >= 1
        assert stack[0].directive is not None
        assert stack[0].directive.get("goal_location") == "PETALBURG_CITY_GYM"

    def test_correct_target_accepted(self):
        """LLM correctly targets PETALBURG_CITY_GYM — stack accepted as-is."""
        db = _make_walkthrough_db(self._chunks())
        vlm = _make_vlm(_gym_response())
        stack = _bootstrap_stack(_PETALBURG_CITY_STATE_DATA, db, vlm)
        assert len(stack) >= 1
        assert stack[0].goal_id == "enter_petalburg_gym"
        assert stack[0].directive["goal_location"] == "PETALBURG_CITY_GYM"

    def test_fallback_stack_targets_gym_for_petalburg_city(self):
        """Milestone fallback for PETALBURG_CITY milestones points at GYM."""
        from agent.objective_manager import MILESTONE_PROGRESSION
        stack = _milestone_fallback_stack(_PETALBURG_CITY_MILESTONES, _PETALBURG_CITY_STATE_DATA)
        assert len(stack) >= 1
        assert stack[0].directive is not None
        assert stack[0].directive.get("goal_location") == "PETALBURG_CITY_GYM"

    def test_no_validation_when_target_is_none(self):
        """next_milestone with target_location=None skips validation — no spurious fallback."""
        # GYM_EXPLANATION has target_location=None; any goal_location is accepted
        gym_exp_milestones = {
            **_PETALBURG_CITY_MILESTONES,
            "DAD_FIRST_MEETING": True,
        }
        state = {**_PETALBURG_CITY_STATE_DATA, "milestones": gym_exp_milestones,
                 "player": {"location": "PETALBURG_CITY_GYM", "position": {"x": 14, "y": 8}}}
        db = _make_walkthrough_db([{"text": "Watch the Wally tutorial.", "metadata": {}, "distance": 0.1}])
        # LLM returns any valid immediate goal — should be accepted without validation
        vlm = _make_vlm(_VALID_HTN_RESPONSE)  # points at PETALBURG_WOODS — fine since no required target
        stack = _bootstrap_stack(state, db, vlm)
        assert isinstance(stack, list)
        assert len(stack) >= 1


# ---------------------------------------------------------------------------
# TestBuildRagQueryNextMilestone
# ---------------------------------------------------------------------------

class TestBuildRagQueryNextMilestone:
    """_build_rag_query uses next_milestone to focus the query."""

    def _next_ms(self) -> dict:
        return {
            "milestone": "DAD_FIRST_MEETING",
            "description": "Enter gym to meet Dad",
            "target_location": "PETALBURG_CITY_GYM",
        }

    def test_query_targets_gym_when_next_milestone_given(self):
        query = _build_rag_query(
            "PETALBURG_CITY", "PETALBURG_CITY", "Petalburg City",
            next_milestone=self._next_ms(),
        )
        assert "Petalburg City Gym" in query

    def test_query_contains_next_milestone_description(self):
        query = _build_rag_query(
            "PETALBURG_CITY", "PETALBURG_CITY", "Petalburg City",
            next_milestone=self._next_ms(),
        )
        assert "Enter gym to meet Dad" in query

    def test_query_without_next_milestone_still_works(self):
        """No next_milestone falls back to old behaviour gracefully."""
        query = _build_rag_query("PETALBURG_CITY", "PETALBURG_CITY", "Petalburg City")
        assert "Petalburg City" in query
        assert isinstance(query, str) and len(query) > 0

    def test_query_excludes_route104_when_gym_is_next(self):
        """With next_milestone=DAD_FIRST_MEETING, Route 104 should NOT appear
        in the description text (we only include the single next description)."""
        query = _build_rag_query(
            "PETALBURG_CITY", "PETALBURG_CITY", "Petalburg City",
            next_milestone=self._next_ms(),
        )
        assert "Route 104" not in query

    def test_bootstrap_passes_next_milestone_to_rag(self):
        """_bootstrap_stack passes next_milestone to db.query so the RAG call
        is focused on the gym, not on later milestones."""
        db = _make_walkthrough_db([])
        vlm = _make_vlm(_gym_response())
        _bootstrap_stack(_PETALBURG_CITY_STATE_DATA, db, vlm)
        call_args = db.query.call_args
        query_text = call_args[0][0] if call_args[0] else call_args[1].get("query_text", "")
        assert "Petalburg City Gym" in query_text
