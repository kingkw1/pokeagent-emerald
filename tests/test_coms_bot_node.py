"""
Tests for Phase 3 — ComsBot node.

Covers:
  TestDialogueAdvance      — normal NPC dialogue → ['A'] with script-idle guard
  TestOpenerBotDelegation  — opener-bot trigger → custom buttons / goal_coords
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.graph.nodes.coms_bot import coms_bot_node
from agent.graph.state import AgentState
from agent.opener_bot import ForceDialogueGoal, NavigationGoal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> AgentState:
    """Return a minimal valid AgentState for dialogue scenarios."""
    base: AgentState = {
        "frame": None,
        "state_data": {
            "player": {"position": {"x": 5, "y": 5}, "location": "ROUTE_101"},
            "game": {"game_state": "dialog", "in_dialog": True},
        },
        "perception": {},
        "goal_coords": None,
        "goal_location": None,
        "npc_coords": None,
        "should_interact": False,
        "milestone_index": 0,
        "context": "dialogue",
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
# TestDialogueAdvance
# ---------------------------------------------------------------------------


class TestDialogueAdvance:
    def test_returns_dialogue_action(self):
        """coms_bot_node always sets last_action='DIALOGUE'."""
        state = _make_state()
        mock_opener = MagicMock()
        mock_opener.should_handle.return_value = False
        with (
            patch("agent.graph.nodes.coms_bot.get_opener_bot", return_value=mock_opener),
            patch("agent.graph.nodes.coms_bot.wait_for_script_idle"),
        ):
            result = coms_bot_node(state)
        assert result["last_action"] == "DIALOGUE"

    def test_normal_dialogue_returns_a_button(self):
        """When OpenerBot doesn't handle, node presses A."""
        state = _make_state()
        mock_opener = MagicMock()
        mock_opener.should_handle.return_value = False
        with (
            patch("agent.graph.nodes.coms_bot.get_opener_bot", return_value=mock_opener),
            patch("agent.graph.nodes.coms_bot.wait_for_script_idle"),
        ):
            result = coms_bot_node(state)
        assert result["last_buttons"] == ["A"]

    def test_wait_for_script_idle_called_for_normal_dialogue(self):
        """wait_for_script_idle is called for regular overworld NPC dialogue."""
        state = _make_state()
        mock_opener = MagicMock()
        mock_opener.should_handle.return_value = False
        mock_idle = MagicMock()
        with (
            patch("agent.graph.nodes.coms_bot.get_opener_bot", return_value=mock_opener),
            patch("agent.graph.nodes.coms_bot.wait_for_script_idle", mock_idle),
        ):
            coms_bot_node(state)
        mock_idle.assert_called_once()

    def test_title_sequence_skips_script_idle(self):
        """TITLE_SEQUENCE location skips wait_for_script_idle."""
        state = _make_state(
            state_data={
                "player": {"position": {"x": 0, "y": 0}, "location": "TITLE_SEQUENCE"},
                "game": {"game_state": "title"},
            }
        )
        mock_opener = MagicMock()
        mock_opener.should_handle.return_value = False
        mock_idle = MagicMock()
        with (
            patch("agent.graph.nodes.coms_bot.get_opener_bot", return_value=mock_opener),
            patch("agent.graph.nodes.coms_bot.wait_for_script_idle", mock_idle),
        ):
            result = coms_bot_node(state)
        mock_idle.assert_not_called()
        assert result["last_buttons"] == ["A"]

    def test_moving_van_skips_script_idle(self):
        """MOVING_VAN location also skips wait_for_script_idle."""
        state = _make_state(
            state_data={
                "player": {"position": {"x": 8, "y": 1}, "location": "MOVING_VAN"},
                "game": {"game_state": "dialog"},
            }
        )
        mock_opener = MagicMock()
        mock_opener.should_handle.return_value = False
        mock_idle = MagicMock()
        with (
            patch("agent.graph.nodes.coms_bot.get_opener_bot", return_value=mock_opener),
            patch("agent.graph.nodes.coms_bot.wait_for_script_idle", mock_idle),
        ):
            result = coms_bot_node(state)
        mock_idle.assert_not_called()

    def test_script_idle_exception_does_not_raise(self):
        """If wait_for_script_idle raises, node still returns ['A'] silently."""
        state = _make_state()
        mock_opener = MagicMock()
        mock_opener.should_handle.return_value = False
        with (
            patch("agent.graph.nodes.coms_bot.get_opener_bot", return_value=mock_opener),
            patch(
                "agent.graph.nodes.coms_bot.wait_for_script_idle",
                side_effect=ConnectionError("server down"),
            ),
        ):
            result = coms_bot_node(state)
        assert result["last_buttons"] == ["A"]


# ---------------------------------------------------------------------------
# TestOpenerBotDelegation
# ---------------------------------------------------------------------------


class TestOpenerBotDelegation:
    def test_list_result_used_directly(self):
        """When OpenerBot returns a list of buttons, use them directly."""
        state = _make_state()
        mock_opener = MagicMock()
        mock_opener.should_handle.return_value = True
        mock_opener.get_action.return_value = ["START", "A"]
        with patch("agent.graph.nodes.coms_bot.get_opener_bot", return_value=mock_opener):
            result = coms_bot_node(state)
        assert result["last_buttons"] == ["START", "A"]

    def test_force_dialogue_goal_presses_a(self):
        """ForceDialogueGoal → ['A'] to dismiss misclassified dialogue."""
        state = _make_state()
        mock_opener = MagicMock()
        mock_opener.should_handle.return_value = True
        mock_opener.get_action.return_value = ForceDialogueGoal(reason="test")
        with patch("agent.graph.nodes.coms_bot.get_opener_bot", return_value=mock_opener):
            result = coms_bot_node(state)
        assert result["last_buttons"] == ["A"]

    def test_navigation_goal_sets_goal_coords(self):
        """NavigationGoal → goal_coords updated, last_buttons empty."""
        state = _make_state()
        mock_opener = MagicMock()
        mock_opener.should_handle.return_value = True
        mock_opener.get_action.return_value = NavigationGoal(
            x=7, y=3, map_location="TEST_MAP", description="test nav goal"
        )
        with patch("agent.graph.nodes.coms_bot.get_opener_bot", return_value=mock_opener):
            result = coms_bot_node(state)
        assert result["goal_coords"] == (7, 3)
        assert result["last_buttons"] == []

    def test_navigation_goal_with_interact_flag(self):
        """NavigationGoal.should_interact is propagated to state."""
        state = _make_state()
        mock_opener = MagicMock()
        mock_opener.should_handle.return_value = True
        mock_opener.get_action.return_value = NavigationGoal(
            x=5, y=1, map_location="HOUSE", description="clock", should_interact=True
        )
        with patch("agent.graph.nodes.coms_bot.get_opener_bot", return_value=mock_opener):
            result = coms_bot_node(state)
        assert result["should_interact"] is True

    def test_opener_none_result_falls_back_to_a(self):
        """OpenerBot returning None → fall through to ['A']."""
        state = _make_state()
        mock_opener = MagicMock()
        mock_opener.should_handle.return_value = True
        mock_opener.get_action.return_value = None
        with patch("agent.graph.nodes.coms_bot.get_opener_bot", return_value=mock_opener):
            result = coms_bot_node(state)
        assert result["last_buttons"] == ["A"]

    def test_opener_delegation_not_a_only(self):
        """When OpenerBot handles with custom buttons, result differs from ['A']."""
        state = _make_state()
        mock_opener = MagicMock()
        mock_opener.should_handle.return_value = True
        mock_opener.get_action.return_value = ["DOWN", "A"]
        with patch("agent.graph.nodes.coms_bot.get_opener_bot", return_value=mock_opener):
            result = coms_bot_node(state)
        assert result["last_buttons"] != ["A"]
        assert result["last_buttons"] == ["DOWN", "A"]

    def test_non_output_fields_unchanged(self):
        """Non-output fields in state are unchanged after coms_bot_node."""
        state = _make_state(milestone_index=2, step_count=17, context="dialogue")
        mock_opener = MagicMock()
        mock_opener.should_handle.return_value = False
        with (
            patch("agent.graph.nodes.coms_bot.get_opener_bot", return_value=mock_opener),
            patch("agent.graph.nodes.coms_bot.wait_for_script_idle"),
        ):
            result = coms_bot_node(state)
        assert result["milestone_index"] == 2
        assert result["step_count"] == 17
        assert result["context"] == "dialogue"
