"""
Tests for Phase 3 — BattleBot node.

Covers:
  TestWildBattle        — battle_bot_node returns BATTLE action and buttons
  TestTrainerBattle     — decision-to-buttons mapping for various decisions
  TestBattlePassthrough — non-output state fields are unchanged
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.graph.nodes.battle_bot import battle_bot_node
from agent.graph.state import AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> AgentState:
    """Return a minimal valid AgentState for battle scenarios."""
    base: AgentState = {
        "frame": None,
        "state_data": {
            "player": {"position": {"x": 5, "y": 5}, "location": "ROUTE_101"},
            "game": {"game_state": "battle", "in_battle": True},
        },
        "perception": {},
        "goal_coords": None,
        "goal_location": None,
        "npc_coords": None,
        "should_interact": False,
        "milestone_index": 0,
        "context": "battle",
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
# TestWildBattle
# ---------------------------------------------------------------------------


class TestWildBattle:
    def test_returns_battle_action(self):
        """battle_bot_node always sets last_action='BATTLE'."""
        state = _make_state()
        mock_bot = MagicMock()
        mock_bot.get_action.return_value = "SELECT_RUN"
        with patch("agent.graph.nodes.battle_bot.get_battle_bot", return_value=mock_bot):
            result = battle_bot_node(state)
        assert result["last_action"] == "BATTLE"

    def test_run_decision_maps_to_a(self):
        """SELECT_RUN → ['A'] (confirm the Run option in the battle menu)."""
        state = _make_state()
        mock_bot = MagicMock()
        mock_bot.get_action.return_value = "SELECT_RUN"
        with patch("agent.graph.nodes.battle_bot.get_battle_bot", return_value=mock_bot):
            result = battle_bot_node(state)
        assert result["last_buttons"] == ["A"]

    def test_vlm_select_run_maps_correctly(self):
        """VLM_SELECT_RUN → ['DOWN', 'RIGHT', 'A']."""
        state = _make_state()
        mock_bot = MagicMock()
        mock_bot.get_action.return_value = "VLM_SELECT_RUN"
        with patch("agent.graph.nodes.battle_bot.get_battle_bot", return_value=mock_bot):
            result = battle_bot_node(state)
        assert result["last_buttons"] == ["DOWN", "RIGHT", "A"]


# ---------------------------------------------------------------------------
# TestTrainerBattle
# ---------------------------------------------------------------------------


class TestTrainerBattle:
    def test_advance_battle_dialogue_maps_to_b_b(self):
        """ADVANCE_BATTLE_DIALOGUE → ['B', 'B']."""
        state = _make_state()
        mock_bot = MagicMock()
        mock_bot.get_action.return_value = "ADVANCE_BATTLE_DIALOGUE"
        with patch("agent.graph.nodes.battle_bot.get_battle_bot", return_value=mock_bot):
            result = battle_bot_node(state)
        assert result["last_buttons"] == ["B", "B"]

    def test_use_move_absorb_maps_correctly(self):
        """USE_MOVE_ABSORB → ['B', 'UP', 'LEFT', 'A', 'DOWN', 'A']."""
        state = _make_state()
        mock_bot = MagicMock()
        mock_bot.get_action.return_value = "USE_MOVE_ABSORB"
        with patch("agent.graph.nodes.battle_bot.get_battle_bot", return_value=mock_bot):
            result = battle_bot_node(state)
        assert result["last_buttons"] == ["B", "UP", "LEFT", "A", "DOWN", "A"]

    def test_use_move_pound_maps_correctly(self):
        """USE_MOVE_POUND → ['B', 'UP', 'LEFT', 'A', 'UP', 'A']."""
        state = _make_state()
        mock_bot = MagicMock()
        mock_bot.get_action.return_value = "USE_MOVE_POUND"
        with patch("agent.graph.nodes.battle_bot.get_battle_bot", return_value=mock_bot):
            result = battle_bot_node(state)
        assert result["last_buttons"] == ["B", "UP", "LEFT", "A", "UP", "A"]

    def test_recover_from_run_failure_maps_to_b(self):
        """RECOVER_FROM_RUN_FAILURE → ['B']."""
        state = _make_state()
        mock_bot = MagicMock()
        mock_bot.get_action.return_value = "RECOVER_FROM_RUN_FAILURE"
        with patch("agent.graph.nodes.battle_bot.get_battle_bot", return_value=mock_bot):
            result = battle_bot_node(state)
        assert result["last_buttons"] == ["B"]

    def test_press_b_maps_to_b(self):
        """PRESS_B → ['B']."""
        state = _make_state()
        mock_bot = MagicMock()
        mock_bot.get_action.return_value = "PRESS_B"
        with patch("agent.graph.nodes.battle_bot.get_battle_bot", return_value=mock_bot):
            result = battle_bot_node(state)
        assert result["last_buttons"] == ["B"]


# ---------------------------------------------------------------------------
# TestBattlePassthrough
# ---------------------------------------------------------------------------


class TestBattlePassthrough:
    def test_non_output_fields_unchanged(self):
        """All non-output fields in state are unchanged after battle_bot_node."""
        state = _make_state(milestone_index=3, step_count=42, context="battle")
        mock_bot = MagicMock()
        mock_bot.get_action.return_value = "PRESS_B"
        with patch("agent.graph.nodes.battle_bot.get_battle_bot", return_value=mock_bot):
            result = battle_bot_node(state)
        assert result["milestone_index"] == 3
        assert result["step_count"] == 42
        assert result["context"] == "battle"

    def test_none_decision_falls_back_to_a(self):
        """BattleBot returning None falls back to ['A']."""
        state = _make_state()
        mock_bot = MagicMock()
        mock_bot.get_action.return_value = None
        with patch("agent.graph.nodes.battle_bot.get_battle_bot", return_value=mock_bot):
            result = battle_bot_node(state)
        assert result["last_buttons"] == ["A"]

    def test_unknown_decision_falls_back_to_a(self):
        """Unrecognised symbolic decision falls back to ['A']."""
        state = _make_state()
        mock_bot = MagicMock()
        mock_bot.get_action.return_value = "UNKNOWN_DECISION"
        with patch("agent.graph.nodes.battle_bot.get_battle_bot", return_value=mock_bot):
            result = battle_bot_node(state)
        assert result["last_buttons"] == ["A"]

    def test_perception_injected_into_state_data(self):
        """latest_observation from perception is passed to BattleBot."""
        obs = {"text_box_visible": False, "game_state": "battle"}
        state = _make_state(perception={"latest_observation": obs})
        mock_bot = MagicMock()
        mock_bot.get_action.return_value = "SELECT_FIGHT"
        with patch("agent.graph.nodes.battle_bot.get_battle_bot", return_value=mock_bot):
            battle_bot_node(state)
        call_kwargs = mock_bot.get_action.call_args
        state_data_arg = call_kwargs[0][0]
        assert state_data_arg.get("latest_observation") == obs
