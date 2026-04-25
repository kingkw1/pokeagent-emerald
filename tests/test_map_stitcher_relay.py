"""
Tests for Phase 3 — MapStitcherRelay node.

Covers:
  TestPixelToTile       — pixel-to-tile coordinate conversion
  TestOffCenterPokeCenter — off-centre pixel offsets translate correctly
  TestContextReset      — output state fields are set correctly
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.graph.nodes.map_stitcher_relay import make_map_stitcher_relay_node
from agent.graph.state import AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> AgentState:
    """Return a minimal valid AgentState for map-stitcher-relay scenarios."""
    base: AgentState = {
        "frame": MagicMock(),   # simulate a PIL Image
        "state_data": {
            "player": {"position": {"x": 10, "y": 10}, "location": "ROUTE_101"},
            "game": {},
        },
        "perception": {},
        "goal_coords": None,
        "goal_location": None,
        "npc_coords": None,
        "should_interact": False,
        "milestone_index": 0,
        "context": "healing_needed",
        "reward": None,
        "prev_state_snapshot": None,
        "last_action": None,
        "last_buttons": [],
        "step_count": 1,
        "telemetry": None,
    }
    base.update(overrides)
    return base


def _mock_stitcher_no_overhead() -> MagicMock:
    """Return a stitcher mock that has no get_overhead_image method."""
    return MagicMock(spec=[])   # spec=[] → no attributes → hasattr returns False


# ---------------------------------------------------------------------------
# TestPixelToTile
# ---------------------------------------------------------------------------


class TestPixelToTile:
    def test_center_pixel_maps_to_player_coords(self):
        """VLM returns screen centre (120, 80) → goal_coords == player pos."""
        mock_vlm = MagicMock()
        mock_vlm.get_query.return_value = {"center_x": 120, "center_y": 80}
        node = make_map_stitcher_relay_node(mock_vlm)
        state = _make_state()   # player at (10, 10)
        with patch(
            "agent.graph.nodes.map_stitcher_relay.get_instance",
            return_value=_mock_stitcher_no_overhead(),
        ):
            result = node(state)
        assert result["goal_coords"] == (10, 10)

    def test_returns_heal_route_action(self):
        """last_action is HEAL_ROUTE after relay fires."""
        mock_vlm = MagicMock()
        mock_vlm.get_query.return_value = {"center_x": 120, "center_y": 80}
        node = make_map_stitcher_relay_node(mock_vlm)
        state = _make_state()
        with patch(
            "agent.graph.nodes.map_stitcher_relay.get_instance",
            return_value=_mock_stitcher_no_overhead(),
        ):
            result = node(state)
        assert result["last_action"] == "HEAL_ROUTE"

    def test_goal_coords_is_tuple(self):
        """goal_coords is a tuple, not a list."""
        mock_vlm = MagicMock()
        mock_vlm.get_query.return_value = {"center_x": 120, "center_y": 80}
        node = make_map_stitcher_relay_node(mock_vlm)
        state = _make_state()
        with patch(
            "agent.graph.nodes.map_stitcher_relay.get_instance",
            return_value=_mock_stitcher_no_overhead(),
        ):
            result = node(state)
        assert isinstance(result["goal_coords"], tuple)


# ---------------------------------------------------------------------------
# TestOffCenterPokeCenter
# ---------------------------------------------------------------------------


class TestOffCenterPokeCenter:
    def test_one_tile_right(self):
        """VLM returns (136, 80) → goal is 1 tile right of player."""
        mock_vlm = MagicMock()
        mock_vlm.get_query.return_value = {"center_x": 136, "center_y": 80}
        node = make_map_stitcher_relay_node(mock_vlm)
        state = _make_state()   # player at (10, 10)
        with patch(
            "agent.graph.nodes.map_stitcher_relay.get_instance",
            return_value=_mock_stitcher_no_overhead(),
        ):
            result = node(state)
        assert result["goal_coords"] == (11, 10)

    def test_one_tile_down(self):
        """VLM returns (120, 96) → goal is 1 tile below player."""
        mock_vlm = MagicMock()
        mock_vlm.get_query.return_value = {"center_x": 120, "center_y": 96}
        node = make_map_stitcher_relay_node(mock_vlm)
        state = _make_state()
        with patch(
            "agent.graph.nodes.map_stitcher_relay.get_instance",
            return_value=_mock_stitcher_no_overhead(),
        ):
            result = node(state)
        assert result["goal_coords"] == (10, 11)

    def test_negative_pixel_offset(self):
        """Pixel left of centre → goal tile left of player."""
        mock_vlm = MagicMock()
        # 120 - 16 = 104 → one tile left
        mock_vlm.get_query.return_value = {"center_x": 104, "center_y": 80}
        node = make_map_stitcher_relay_node(mock_vlm)
        state = _make_state()   # player at (10, 10)
        with patch(
            "agent.graph.nodes.map_stitcher_relay.get_instance",
            return_value=_mock_stitcher_no_overhead(),
        ):
            result = node(state)
        assert result["goal_coords"] == (9, 10)

    def test_large_offset(self):
        """Several tiles away from player is computed correctly."""
        mock_vlm = MagicMock()
        # 3 tiles right (120 + 3*16 = 168), 2 tiles up (80 - 2*16 = 48)
        mock_vlm.get_query.return_value = {"center_x": 168, "center_y": 48}
        node = make_map_stitcher_relay_node(mock_vlm)
        state = _make_state()   # player at (10, 10)
        with patch(
            "agent.graph.nodes.map_stitcher_relay.get_instance",
            return_value=_mock_stitcher_no_overhead(),
        ):
            result = node(state)
        assert result["goal_coords"] == (13, 8)


# ---------------------------------------------------------------------------
# TestContextReset
# ---------------------------------------------------------------------------


class TestContextReset:
    def test_context_set_to_navigation(self):
        """After relay, context is 'navigation' regardless of prior value."""
        mock_vlm = MagicMock()
        mock_vlm.get_query.return_value = {"center_x": 120, "center_y": 80}
        node = make_map_stitcher_relay_node(mock_vlm)
        state = _make_state(context="healing_needed")
        with patch(
            "agent.graph.nodes.map_stitcher_relay.get_instance",
            return_value=_mock_stitcher_no_overhead(),
        ):
            result = node(state)
        assert result["context"] == "navigation"

    def test_no_frame_and_no_overhead_returns_unchanged(self):
        """Without frame or overhead image state is returned unchanged."""
        mock_vlm = MagicMock()
        node = make_map_stitcher_relay_node(mock_vlm)
        state = _make_state(frame=None)
        with patch(
            "agent.graph.nodes.map_stitcher_relay.get_instance",
            return_value=_mock_stitcher_no_overhead(),
        ):
            result = node(state)
        assert result["goal_coords"] is None
        mock_vlm.get_query.assert_not_called()

    def test_non_output_fields_unchanged(self):
        """milestone_index and step_count are untouched after relay."""
        mock_vlm = MagicMock()
        mock_vlm.get_query.return_value = {"center_x": 120, "center_y": 80}
        node = make_map_stitcher_relay_node(mock_vlm)
        state = _make_state(milestone_index=5, step_count=100)
        with patch(
            "agent.graph.nodes.map_stitcher_relay.get_instance",
            return_value=_mock_stitcher_no_overhead(),
        ):
            result = node(state)
        assert result["milestone_index"] == 5
        assert result["step_count"] == 100

    def test_vlm_failure_returns_state_unchanged(self):
        """VLM exception → state returned unchanged."""
        mock_vlm = MagicMock()
        mock_vlm.get_query.side_effect = RuntimeError("timeout")
        node = make_map_stitcher_relay_node(mock_vlm)
        state = _make_state()
        with patch(
            "agent.graph.nodes.map_stitcher_relay.get_instance",
            return_value=_mock_stitcher_no_overhead(),
        ):
            result = node(state)
        assert result["goal_coords"] is None
        assert result["last_action"] is None

    def test_vlm_bad_response_returns_state_unchanged(self):
        """Malformed VLM response → state returned unchanged."""
        mock_vlm = MagicMock()
        mock_vlm.get_query.return_value = "not a dict"
        node = make_map_stitcher_relay_node(mock_vlm)
        state = _make_state()
        with patch(
            "agent.graph.nodes.map_stitcher_relay.get_instance",
            return_value=_mock_stitcher_no_overhead(),
        ):
            result = node(state)
        assert result["goal_coords"] is None
