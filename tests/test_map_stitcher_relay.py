"""
Tests for Phase 3 — MapStitcherRelay node.

Covers:
  TestLocationGraphLookup — location_graph coordinate path (no VLM)
  TestPixelToTile         — VLM fallback pixel-to-tile coordinate conversion
  TestOffCenterPokeCenter — VLM fallback off-centre pixel offsets
  TestContextReset        — output state fields are set correctly
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
    """Return a minimal valid AgentState for map-stitcher-relay scenarios.

    Default location is "UNKNOWN_CITY_XYZ" — deliberately absent from
    location_graph so VLM-fallback tests exercise the VLM path.
    Use ``state_data={"player": {"location": "PETALBURG CITY", ...}}``
    overrides for location_graph path tests.
    """
    base: AgentState = {
        "frame": MagicMock(),   # simulate a PIL Image
        "state_data": {
            "player": {"position": {"x": 10, "y": 10}, "location": "UNKNOWN_CITY_XYZ"},
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
# TestLocationGraphLookup
# ---------------------------------------------------------------------------


def _make_city_state(location: str, player_x: int = 10, player_y: int = 10) -> AgentState:
    """State helper with a real city location for location_graph path tests."""
    state = _make_state()
    state["state_data"] = {
        "player": {"position": {"x": player_x, "y": player_y}, "location": location},
        "game": {},
    }
    return state


class TestLocationGraphLookup:
    """location_graph path: no VLM call, deterministic coords from graph data."""

    def test_petalburg_city_returns_pc_entrance(self):
        """Petalburg City → PC entrance at (6, 8)."""
        mock_vlm = MagicMock()
        node = make_map_stitcher_relay_node(mock_vlm)
        state = _make_city_state("Petalburg City")
        result = node(state)
        assert result["goal_coords"] == (6, 8)
        assert result["last_action"] == "HEAL_ROUTE"
        assert result["context"] == "navigation"
        mock_vlm.get_query.assert_not_called()

    def test_petalburg_city_normalises_spaces(self):
        """Location with spaces is normalised before lookup."""
        mock_vlm = MagicMock()
        node = make_map_stitcher_relay_node(mock_vlm)
        # Game state returns location with spaces and mixed case
        state = _make_city_state("PETALBURG CITY")
        result = node(state)
        assert result["goal_coords"] == (6, 8)
        mock_vlm.get_query.assert_not_called()

    def test_petalburg_city_sets_goal_location(self):
        """goal_location is the PC graph key after location_graph lookup."""
        mock_vlm = MagicMock()
        node = make_map_stitcher_relay_node(mock_vlm)
        state = _make_city_state("Petalburg City")
        result = node(state)
        assert result["goal_location"] == "PETALBURG_CITY_POKEMON_CENTER_1F"

    def test_oldale_town_returns_pc_entrance(self):
        """Oldale Town → PC entrance at (6, 16)."""
        mock_vlm = MagicMock()
        node = make_map_stitcher_relay_node(mock_vlm)
        state = _make_city_state("OLDALE TOWN")
        result = node(state)
        assert result["goal_coords"] == (6, 16)
        mock_vlm.get_query.assert_not_called()

    def test_rustboro_city_returns_pc_entrance(self):
        """Rustboro City → PC entrance at (16, 38)."""
        mock_vlm = MagicMock()
        node = make_map_stitcher_relay_node(mock_vlm)
        state = _make_city_state("RUSTBORO CITY")
        result = node(state)
        assert result["goal_coords"] == (16, 38)
        mock_vlm.get_query.assert_not_called()

    def test_player_position_not_used_for_coords(self):
        """goal_coords comes from graph, not relative to player position."""
        mock_vlm = MagicMock()
        node = make_map_stitcher_relay_node(mock_vlm)
        # Player at a wildly different position — coords should still be (6, 8)
        state = _make_city_state("PETALBURG CITY", player_x=99, player_y=99)
        result = node(state)
        assert result["goal_coords"] == (6, 8)

    def test_goal_coords_is_tuple(self):
        """goal_coords from location_graph is a tuple, not a list."""
        mock_vlm = MagicMock()
        node = make_map_stitcher_relay_node(mock_vlm)
        state = _make_city_state("PETALBURG CITY")
        result = node(state)
        assert isinstance(result["goal_coords"], tuple)

    def test_non_output_fields_unchanged(self):
        """milestone_index and step_count are untouched."""
        mock_vlm = MagicMock()
        node = make_map_stitcher_relay_node(mock_vlm)
        state = _make_city_state("PETALBURG CITY")
        state["milestone_index"] = 7
        state["step_count"] = 42
        result = node(state)
        assert result["milestone_index"] == 7
        assert result["step_count"] == 42


# ---------------------------------------------------------------------------
# TestPixelToTile  (VLM fallback — location not in graph)
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
# TestOffCenterPokeCenter  (VLM fallback — location not in graph)
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
# TestContextReset  (covers both paths)
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
