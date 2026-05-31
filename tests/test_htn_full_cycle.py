"""
tests/test_htn_full_cycle.py — Phase 7.2 automated tests for HTN full cycle.

Covers:
  TestBuildGraphUsehtn          — build_graph(use_htn=True) compiles without error
  TestFullCycleNavHandoff       — use_htn=True: goal_coords comes from Stack[0].directive
  TestFullCycleBattleHandoff    — nav_bot → battle_bot → nav_bot handoff sequence:
                                  Supervisor issues CONTINUE (goal still navigation)
  TestFullCycleDialogueHandoff  — nav_bot → coms_bot → nav_bot handoff sequence:
                                  Supervisor issues POP; Stack[0] advances
  TestMilestoneIndexInit        — milestone_index is set from save-state milestones on
                                  step 0, not always 0

All tests use mocked VLM / episodic memory so no LLM calls are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.graph.goal_stack import GoalNode, stack_push
from agent.graph.nodes.executive_supervisor import (
    _apply_immediate_directive,
    make_executive_supervisor_node,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MODULE = "agent.graph.nodes.executive_supervisor"


def _goal(
    goal_id: str,
    goal_type: str = "immediate",
    goal_location: str | None = None,
    goal_coords: list | None = None,
    **kwargs,
) -> GoalNode:
    directive = None
    if goal_type == "immediate":
        directive = {
            "action": "NAVIGATE",
            "goal_coords": goal_coords,
            "goal_location": goal_location,
            "should_interact": False,
            "npc_coords": None,
            "description": f"Navigate to {goal_location or 'target'}",
        }
    return GoalNode(
        goal_id=goal_id,
        description=f"desc:{goal_id}",
        goal_type=goal_type,
        directive=directive,
        **kwargs,
    )


def _stack_dicts(*nodes: GoalNode) -> list[dict]:
    return [n.to_dict() for n in nodes]


def _base_state(**overrides) -> dict:
    base = {
        "step_count": 5,
        "goal_stack": [],
        "supervisor_pending": True,
        "supervisor_last_operation": None,
        "supervisor_last_reasoning": None,
        "state_data": {
            "player": {"position": {"x": 10, "y": 20}, "location": "PETALBURG_CITY"},
            "game": {"in_battle": False, "in_dialog": False, "game_state": "overworld",
                     "badges": 0},
            "party": [{"hp": 30, "max_hp": 40}],
        },
        "last_action": "NAVIGATE",
        "last_node_fired": "nav_bot",
        "_boot_timestamp": 0.0,
        "milestone_index": 17,
        "goal_location": "PETALBURG_CITY_GYM",
        "goal_coords": None,
    }
    base.update(overrides)
    return base


def _llm_response(operation: str, reasoning: str = "test", new_goals: list | None = None) -> dict:
    return {"operation": operation, "reasoning": reasoning, "new_goals": new_goals or []}


def _make_node(use_htn: bool = False):
    return make_executive_supervisor_node(
        vlm=None,
        episodic_memory=None,
        walkthrough_db=None,
        use_htn=use_htn,
    )


# ---------------------------------------------------------------------------
# TestBuildGraphUsehtn
# ---------------------------------------------------------------------------

class TestBuildGraphUsehtn:
    """build_graph(use_htn=True) compiles without error and has all nodes."""

    def test_build_graph_use_htn_true_compiles(self):
        from agent.graph.graph import build_graph
        graph = build_graph(MagicMock(), MagicMock(), use_htn=True)
        assert graph is not None
        assert callable(getattr(graph, "invoke", None))

    def test_build_graph_use_htn_false_compiles(self):
        from agent.graph.graph import build_graph
        graph = build_graph(MagicMock(), MagicMock(), use_htn=False)
        assert graph is not None

    def test_build_graph_default_use_htn_false(self):
        from agent.graph.graph import build_graph
        graph = build_graph(MagicMock(), MagicMock())
        assert graph is not None

    def test_both_graphs_have_same_nodes(self):
        from agent.graph.graph import build_graph
        g_shadow = build_graph(MagicMock(), MagicMock(), use_htn=False)
        g_active = build_graph(MagicMock(), MagicMock(), use_htn=True)
        shadow_nodes = set(g_shadow.get_graph().nodes.keys())
        active_nodes = set(g_active.get_graph().nodes.keys())
        assert shadow_nodes == active_nodes, (
            f"Node sets differ: shadow={shadow_nodes - active_nodes} "
            f"active={active_nodes - shadow_nodes}"
        )


# ---------------------------------------------------------------------------
# TestFullCycleNavHandoff
# ---------------------------------------------------------------------------

class TestFullCycleNavHandoff:
    """use_htn=True: Stack[0].directive overwrites goal_location / goal_coords."""

    def test_htn_directive_sets_goal_location(self):
        node = _make_node(use_htn=True)
        gym_goal = _goal("enter_gym", goal_location="PETALBURG_CITY_GYM")
        state = _base_state(
            goal_stack=_stack_dicts(gym_goal),
            goal_location="PETALBURG_CITY",  # old FSM value — should be overwritten
        )
        with patch(f"{_MODULE}._call_supervisor_llm", return_value=_llm_response("CONTINUE")):
            result = node(state)
        assert result["goal_location"] == "PETALBURG_CITY_GYM"

    def test_htn_directive_sets_goal_coords(self):
        node = _make_node(use_htn=True)
        coords = [14, 8, "PETALBURG_CITY_GYM"]
        gym_goal = _goal("enter_gym", goal_location="PETALBURG_CITY_GYM", goal_coords=coords)
        state = _base_state(
            goal_stack=_stack_dicts(gym_goal),
            goal_coords=None,
        )
        with patch(f"{_MODULE}._call_supervisor_llm", return_value=_llm_response("CONTINUE")):
            result = node(state)
        assert result["goal_coords"] == coords

    def test_shadow_mode_does_not_overwrite_goal_location(self):
        """use_htn=False: HTN builds stack but leaves goal_location from FSM."""
        node = _make_node(use_htn=False)
        gym_goal = _goal("enter_gym", goal_location="PETALBURG_CITY_GYM")
        fsm_location = "PETALBURG_CITY"
        state = _base_state(
            goal_stack=_stack_dicts(gym_goal),
            goal_location=fsm_location,
        )
        with patch(f"{_MODULE}._call_supervisor_llm", return_value=_llm_response("CONTINUE")):
            result = node(state)
        assert result["goal_location"] == fsm_location

    def test_milestone_index_preserved_in_result(self):
        node = _make_node(use_htn=True)
        gym_goal = _goal("enter_gym", goal_location="PETALBURG_CITY_GYM")
        state = _base_state(goal_stack=_stack_dicts(gym_goal), milestone_index=17)
        with patch(f"{_MODULE}._call_supervisor_llm", return_value=_llm_response("CONTINUE")):
            result = node(state)
        assert result["milestone_index"] == 17

    def test_htn_no_directive_leaves_goal_location_unchanged(self):
        """When Stack[0] has no directive, _apply_immediate_directive is a no-op."""
        node = _make_node(use_htn=True)
        # Strategic goal has no directive
        strategic_goal = _goal("earn_badge", goal_type="strategic")
        state = _base_state(
            goal_stack=_stack_dicts(strategic_goal),
            goal_location="PETALBURG_CITY",
        )
        with patch(f"{_MODULE}._call_supervisor_llm", return_value=_llm_response("CONTINUE")):
            result = node(state)
        assert result.get("goal_location") == "PETALBURG_CITY"


# ---------------------------------------------------------------------------
# TestFullCycleBattleHandoff
# ---------------------------------------------------------------------------

class TestFullCycleBattleHandoff:
    """nav_bot → battle_bot → nav_bot handoff: Supervisor issues CONTINUE."""

    def test_supervisor_continues_after_battle_handoff(self):
        """After a battle ends, the nav goal is still valid — expect CONTINUE."""
        node = _make_node(use_htn=True)
        nav_goal = _goal("navigate_to_gym", goal_location="PETALBURG_CITY_GYM")
        state = _base_state(
            goal_stack=_stack_dicts(nav_goal),
            last_node_fired="nav_bot",
            state_data={
                "player": {"position": {"x": 10, "y": 20}, "location": "PETALBURG_CITY"},
                "game": {"in_battle": False, "in_dialog": False, "game_state": "overworld",
                         "badges": 0},
                "party": [{"hp": 30, "max_hp": 40}],
            },
        )
        with patch(f"{_MODULE}._call_supervisor_llm", return_value=_llm_response("CONTINUE", "Battle ended, nav goal still valid")):
            result = node(state)

        assert result["supervisor_last_operation"] == "CONTINUE"
        assert len(result["goal_stack"]) == 1
        assert result["goal_stack"][0]["goal_id"] == "navigate_to_gym"

    def test_goal_coords_reapplied_after_battle_continue(self):
        """use_htn=True: after CONTINUE, Stack[0] directive still drives nav fields."""
        node = _make_node(use_htn=True)
        coords = [14, 8, "PETALBURG_CITY_GYM"]
        nav_goal = _goal("navigate_to_gym", goal_location="PETALBURG_CITY_GYM", goal_coords=coords)
        state = _base_state(
            goal_stack=_stack_dicts(nav_goal),
            goal_coords=None,
            goal_location="PETALBURG_CITY",
        )
        with patch(f"{_MODULE}._call_supervisor_llm", return_value=_llm_response("CONTINUE")):
            result = node(state)

        assert result["goal_location"] == "PETALBURG_CITY_GYM"
        assert result["goal_coords"] == coords

    def test_supervisor_pending_false_after_continue(self):
        node = _make_node(use_htn=False)
        nav_goal = _goal("navigate_to_gym", goal_location="PETALBURG_CITY_GYM")
        state = _base_state(goal_stack=_stack_dicts(nav_goal), supervisor_pending=True)
        with patch(f"{_MODULE}._call_supervisor_llm", return_value=_llm_response("CONTINUE")):
            result = node(state)
        assert result["supervisor_pending"] is False


# ---------------------------------------------------------------------------
# TestFullCycleDialogueHandoff
# ---------------------------------------------------------------------------

class TestFullCycleDialogueHandoff:
    """nav_bot → coms_bot → nav_bot handoff: Supervisor POP advances the stack."""

    def _dialogue_state(self, **overrides) -> dict:
        """State after Norman dialogue completes (coms_bot → nav_bot handoff)."""
        base = _base_state(
            last_node_fired="coms_bot",
            state_data={
                "player": {"position": {"x": 14, "y": 8}, "location": "PETALBURG_CITY_GYM"},
                "game": {"in_battle": False, "in_dialog": False, "game_state": "overworld",
                         "badges": 0},
                "party": [{"hp": 30, "max_hp": 40}],
            },
        )
        base.update(overrides)
        return base

    def test_pop_after_dialogue_shrinks_stack(self):
        """Supervisor POP removes the completed dialogue goal from the stack."""
        node = _make_node(use_htn=True)
        dialogue_goal = _goal("talk_to_norman", goal_location="PETALBURG_CITY_GYM")
        nav_goal = _goal("head_to_route_104", goal_location="ROUTE_104_SOUTH", goal_type="tactical")
        state = self._dialogue_state(
            goal_stack=_stack_dicts(dialogue_goal, nav_goal),
        )
        with patch(f"{_MODULE}._call_supervisor_llm",
                   return_value=_llm_response("POP", "Norman dialogue complete — transcript confirmed")):
            result = node(state)

        assert result["supervisor_last_operation"] == "POP"
        assert len(result["goal_stack"]) == 1
        assert result["goal_stack"][0]["goal_id"] == "head_to_route_104"

    def test_pop_advances_to_next_tactical_goal(self):
        """After POP, the new Stack[0] directive is applied when use_htn=True."""
        node = _make_node(use_htn=True)
        dialogue_goal = _goal("talk_to_norman", goal_location="PETALBURG_CITY_GYM")
        nav_goal = _goal(
            "head_to_route_104",
            goal_type="immediate",
            goal_location="ROUTE_104_SOUTH",
        )
        state = self._dialogue_state(
            goal_stack=_stack_dicts(dialogue_goal, nav_goal),
            goal_location="PETALBURG_CITY_GYM",  # old location before POP
        )
        with patch(f"{_MODULE}._call_supervisor_llm",
                   return_value=_llm_response("POP", "Norman dialogue complete")):
            result = node(state)

        assert result["goal_location"] == "ROUTE_104_SOUTH"

    def test_continue_during_ongoing_dialogue_keeps_stack(self):
        """CONTINUE during dialogue: stack is unchanged, goal_location not altered."""
        node = _make_node(use_htn=True)
        dialogue_goal = _goal("talk_to_norman", goal_location="PETALBURG_CITY_GYM")
        state = self._dialogue_state(
            goal_stack=_stack_dicts(dialogue_goal),
            goal_location="PETALBURG_CITY_GYM",
        )
        with patch(f"{_MODULE}._call_supervisor_llm",
                   return_value=_llm_response("CONTINUE", "Norman dialogue not yet complete")):
            result = node(state)

        assert result["supervisor_last_operation"] == "CONTINUE"
        assert len(result["goal_stack"]) == 1
        assert result["goal_stack"][0]["goal_id"] == "talk_to_norman"

    def test_single_item_stack_pop_leaves_empty_stack(self):
        """POP on a one-item stack results in an empty stack (game completed goal)."""
        node = _make_node(use_htn=True)
        dialogue_goal = _goal("talk_to_norman", goal_location="PETALBURG_CITY_GYM")
        state = self._dialogue_state(goal_stack=_stack_dicts(dialogue_goal))
        with patch(f"{_MODULE}._call_supervisor_llm",
                   return_value=_llm_response("POP", "Goal completed")):
            result = node(state)

        assert result["supervisor_last_operation"] == "POP"
        assert result["goal_stack"] == []


# ---------------------------------------------------------------------------
# TestMilestoneIndexInit
# ---------------------------------------------------------------------------

class TestMilestoneIndexInit:
    """Phase 7.2: milestone_index is derived from save-state milestones on step 0.

    These tests exercise the lazy-init logic in Agent.step() indirectly by
    verifying the underlying helper ``get_highest_milestone_index`` and the
    initialisation formula ``max(0, highest + 1)``.
    """

    def test_empty_milestones_gives_index_zero(self):
        from agent.objective_manager import get_highest_milestone_index, MILESTONE_PROGRESSION
        milestones = {}
        highest = get_highest_milestone_index(milestones)
        init_index = max(0, highest + 1)
        assert init_index == 0

    def test_all_incomplete_gives_index_zero(self):
        from agent.objective_manager import get_highest_milestone_index, MILESTONE_PROGRESSION
        milestones = {
            MILESTONE_PROGRESSION[0]["milestone"]: {"completed": False},
            MILESTONE_PROGRESSION[1]["milestone"]: {"completed": False},
        }
        highest = get_highest_milestone_index(milestones)
        init_index = max(0, highest + 1)
        assert init_index == 0

    def test_first_milestone_complete_gives_index_one(self):
        from agent.objective_manager import get_highest_milestone_index, MILESTONE_PROGRESSION
        milestones = {
            MILESTONE_PROGRESSION[0]["milestone"]: {"completed": True},
        }
        highest = get_highest_milestone_index(milestones)
        init_index = max(0, highest + 1)
        assert init_index == 1

    def test_sixteen_milestones_complete_gives_index_seventeen(self):
        """Scenario: save loaded with 16 completed milestones (0..15) → index=16."""
        from agent.objective_manager import get_highest_milestone_index, MILESTONE_PROGRESSION
        milestones = {
            MILESTONE_PROGRESSION[i]["milestone"]: {"completed": True}
            for i in range(16)
        }
        highest = get_highest_milestone_index(milestones)
        init_index = max(0, highest + 1)
        assert init_index == 16

    def test_dad_first_meeting_state_gives_index_seventeen(self):
        """Petalburg City entrance save: milestones 0..16 done → index=17 (DAD_FIRST_MEETING)."""
        from agent.objective_manager import get_highest_milestone_index, MILESTONE_PROGRESSION
        # milestones 0..16 completed
        milestones = {
            MILESTONE_PROGRESSION[i]["milestone"]: {"completed": True}
            for i in range(17)
        }
        highest = get_highest_milestone_index(milestones)
        init_index = max(0, highest + 1)
        assert init_index == 17

    def test_milestone_target_non_null_for_dad_first_meeting(self):
        """With milestone_index=17, shadow log reads PETALBURG_CITY_GYM from MILESTONE_PROGRESSION."""
        from agent.objective_manager import MILESTONE_PROGRESSION
        idx = 17
        target = MILESTONE_PROGRESSION[idx].get("target_location")
        assert target == "PETALBURG_CITY_GYM"

    def test_milestone_target_none_for_gym_explanation(self):
        """GYM_EXPLANATION (index 18) has no target_location — milestone_target=None by design."""
        from agent.objective_manager import MILESTONE_PROGRESSION
        idx = 18
        target = MILESTONE_PROGRESSION[idx].get("target_location")
        assert target is None
