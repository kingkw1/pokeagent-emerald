"""
Tests for Phase 2 — NavBot node.

Covers:
  TestNavBotNodeBasic      — valid goal_coords → NAVIGATE action
  TestNavBotNoGoal         — goal_coords=None → PASS action
  TestNavBotNpcObstacles   — NPC at target tile is excluded from obstacles
  TestNavBotInteract       — should_interact=True appends A when adjacent
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.graph.nodes.nav_bot import nav_bot_node
from agent.graph.state import AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> AgentState:
    """Return a minimal valid AgentState dict."""
    base: AgentState = {
        "frame": None,
        "state_data": {
            "player": {
                "position": {"x": 5, "y": 5},
                "location": "ROUTE_101",
            },
            "active_npcs": [],
            "game": {},
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
        "last_action": None,
        "last_buttons": [],
        "step_count": 1,
        "telemetry": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TestNavBotNodeBasic
# ---------------------------------------------------------------------------


class TestNavBotNodeBasic:
    def test_returns_navigate_action(self):
        """Valid goal_coords → last_action='NAVIGATE'."""
        state = _make_state(goal_coords=(10, 3))
        with patch(
            "agent.graph.nodes.nav_bot.pathfind_to_goal",
            return_value=["RIGHT", "RIGHT"],
        ):
            result = nav_bot_node(state)
        assert result["last_action"] == "NAVIGATE"

    def test_returns_buttons_from_pathfind(self):
        """last_buttons equals the list returned by pathfind_to_goal."""
        state = _make_state(goal_coords=(10, 3))
        with patch(
            "agent.graph.nodes.nav_bot.pathfind_to_goal",
            return_value=["RIGHT", "RIGHT", "DOWN"],
        ):
            result = nav_bot_node(state)
        assert result["last_buttons"] == ["RIGHT", "RIGHT", "DOWN"]

    def test_state_data_passthrough(self):
        """All non-output fields in state are unchanged after nav_bot_node."""
        state = _make_state(goal_coords=(10, 3), milestone_index=5, step_count=42)
        with patch(
            "agent.graph.nodes.nav_bot.pathfind_to_goal",
            return_value=["RIGHT"],
        ):
            result = nav_bot_node(state)
        assert result["milestone_index"] == 5
        assert result["step_count"] == 42
        assert result["context"] == "navigation"

    def test_pathfind_returns_none_yields_empty_buttons(self):
        """If pathfind_to_goal returns None, nav_bot uses a directional fallback.

        The fallback moves one step toward the goal to keep the agent moving and
        expand the explored map even when A* has no solution.
        The default _make_state places the player at (5, 5); goal is (10, 3).
        dx=5, dy=-2 → abs(dx) > abs(dy) → fallback direction is 'RIGHT'.
        """
        state = _make_state(goal_coords=(10, 3))
        with patch(
            "agent.graph.nodes.nav_bot.pathfind_to_goal",
            return_value=None,
        ):
            result = nav_bot_node(state)
        assert result["last_action"] == "NAVIGATE"
        assert result["last_buttons"] == ["RIGHT"]


# ---------------------------------------------------------------------------
# TestNavBotNoGoal
# ---------------------------------------------------------------------------


class TestNavBotNoGoal:
    def test_none_goal_returns_pass(self):
        """goal_coords=None → last_action='PASS'."""
        state = _make_state(goal_coords=None)
        result = nav_bot_node(state)
        assert result["last_action"] == "PASS"

    def test_none_goal_returns_empty_buttons(self):
        """goal_coords=None → last_buttons=[]."""
        state = _make_state(goal_coords=None)
        result = nav_bot_node(state)
        assert result["last_buttons"] == []

    def test_empty_tuple_goal_returns_pass(self):
        """goal_coords=() (falsy) → last_action='PASS'."""
        state = _make_state(goal_coords=())
        result = nav_bot_node(state)
        assert result["last_action"] == "PASS"

    def test_passthrough_fields_preserved_when_no_goal(self):
        """State fields other than last_action/last_buttons unchanged on PASS."""
        state = _make_state(goal_coords=None, milestone_index=3)
        result = nav_bot_node(state)
        assert result["milestone_index"] == 3


# ---------------------------------------------------------------------------
# TestNavBotNpcObstacles
# ---------------------------------------------------------------------------


class TestNavBotNpcObstacles:
    def test_npc_coords_passed_to_pathfind(self):
        """npc_coords from state is forwarded to pathfind_to_goal."""
        npc = (10, 3)
        state = _make_state(
            goal_coords=(10, 3),
            npc_coords=npc,
            state_data={
                "player": {"position": {"x": 5, "y": 3}, "location": "ROUTE_101"},
                "active_npcs": [{"x": 10, "y": 3, "location": "ROUTE_101"}],
                "game": {},
            },
        )
        with patch(
            "agent.graph.nodes.nav_bot.pathfind_to_goal",
            return_value=["RIGHT"],
        ) as mock_pf:
            nav_bot_node(state)
        mock_pf.assert_called_once()
        _, _, _, kwargs = (
            mock_pf.call_args.args[0],
            mock_pf.call_args.args[1],
            mock_pf.call_args.args[2],
            mock_pf.call_args.kwargs,
        )
        assert kwargs.get("npc_coords") == npc

    def test_no_npc_coords_passes_none_to_pathfind(self):
        """When npc_coords is absent, pathfind_to_goal receives npc_coords=None."""
        state = _make_state(goal_coords=(10, 3), npc_coords=None)
        with patch(
            "agent.graph.nodes.nav_bot.pathfind_to_goal",
            return_value=["RIGHT"],
        ) as mock_pf:
            nav_bot_node(state)
        assert mock_pf.call_args.kwargs.get("npc_coords") is None


# ---------------------------------------------------------------------------
# TestNavBotInteract
# ---------------------------------------------------------------------------


class TestNavBotInteract:
    def test_adjacent_npc_appends_a(self):
        """should_interact=True + player adjacent to npc_coords → 'A' appended."""
        # Player at (9, 3), NPC at (10, 3) — Manhattan distance = 1
        state = _make_state(
            goal_coords=(10, 3),
            npc_coords=(10, 3),
            should_interact=True,
            state_data={
                "player": {"position": {"x": 9, "y": 3}, "location": "ROUTE_101"},
                "active_npcs": [],
                "game": {},
            },
        )
        with patch(
            "agent.graph.nodes.nav_bot.pathfind_to_goal",
            return_value=[],
        ):
            result = nav_bot_node(state)
        assert "A" in result["last_buttons"]
        assert result["last_buttons"][-1] == "A"

    def test_adjacent_appends_a_after_movement_buttons(self):
        """When path is non-empty and player is adjacent, 'A' is last button."""
        state = _make_state(
            goal_coords=(10, 3),
            npc_coords=(10, 3),
            should_interact=True,
            state_data={
                "player": {"position": {"x": 9, "y": 3}, "location": "ROUTE_101"},
                "active_npcs": [],
                "game": {},
            },
        )
        with patch(
            "agent.graph.nodes.nav_bot.pathfind_to_goal",
            return_value=["RIGHT"],
        ):
            result = nav_bot_node(state)
        assert result["last_buttons"] == ["RIGHT", "A"]

    def test_not_adjacent_no_a_appended(self):
        """should_interact=True but player not adjacent → 'A' NOT appended."""
        # Player at (5, 3), NPC at (10, 3) — distance = 5
        state = _make_state(
            goal_coords=(10, 3),
            npc_coords=(10, 3),
            should_interact=True,
            state_data={
                "player": {"position": {"x": 5, "y": 3}, "location": "ROUTE_101"},
                "active_npcs": [],
                "game": {},
            },
        )
        with patch(
            "agent.graph.nodes.nav_bot.pathfind_to_goal",
            return_value=["RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"],
        ):
            result = nav_bot_node(state)
        assert "A" not in result["last_buttons"]

    def test_should_interact_false_no_a(self):
        """should_interact=False → 'A' never appended even when adjacent."""
        state = _make_state(
            goal_coords=(10, 3),
            npc_coords=(10, 3),
            should_interact=False,
            state_data={
                "player": {"position": {"x": 9, "y": 3}, "location": "ROUTE_101"},
                "active_npcs": [],
                "game": {},
            },
        )
        with patch(
            "agent.graph.nodes.nav_bot.pathfind_to_goal",
            return_value=[],
        ):
            result = nav_bot_node(state)
        assert "A" not in result["last_buttons"]

    def test_no_npc_coords_no_a(self):
        """should_interact=True but npc_coords=None → 'A' NOT appended."""
        state = _make_state(
            goal_coords=(10, 3),
            npc_coords=None,
            should_interact=True,
            state_data={
                "player": {"position": {"x": 9, "y": 3}, "location": "ROUTE_101"},
                "active_npcs": [],
                "game": {},
            },
        )
        with patch(
            "agent.graph.nodes.nav_bot.pathfind_to_goal",
            return_value=[],
        ):
            result = nav_bot_node(state)
        assert "A" not in result["last_buttons"]
