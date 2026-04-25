"""
tests/test_graph_router.py — Phase 4 tests for routing_condition and graph assembly.

Test classes:
  TestRoutingCondition  — unit tests for routing_condition()
  TestGraphCompiles     — build_graph() succeeds and returns a compiled graph
  TestGraphEdges        — compiled graph contains the expected nodes and edges
  TestFullGraphInvoke   — end-to-end invocation with mock specialist nodes
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.graph.router import routing_condition
from agent.graph.state import AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**game_overrides) -> AgentState:
    """Return a minimal AgentState with game overrides applied."""
    game = {
        "in_battle": False,
        "in_dialog": False,
        "game_state": "overworld",
    }
    game.update(game_overrides)
    return AgentState(
        frame=None,
        state_data={"game": game, "player": {}, "party": []},
        perception={},
        goal_coords=None,
        goal_location=None,
        npc_coords=None,
        should_interact=False,
        milestone_index=0,
        context="navigation",
        reward=None,
        prev_state_snapshot=None,
        last_action=None,
        last_buttons=[],
        step_count=0,
        telemetry=None,
    )


# ---------------------------------------------------------------------------
# TestRoutingCondition
# ---------------------------------------------------------------------------

class TestRoutingCondition:
    def test_battle_flag_routes_to_battle_bot(self):
        state = _make_state(in_battle=True)
        assert routing_condition(state) == "battle_bot"

    def test_context_dialogue_routes_to_coms_bot(self):
        """VLM-derived context='dialogue' should route to coms_bot."""
        state = _make_state()
        state = {**state, "context": "dialogue"}
        assert routing_condition(state) == "coms_bot"

    def test_ram_in_dialog_alone_does_not_route_to_coms_bot(self):
        """RAM in_dialog flag must NOT drive routing (unreliable in save states)."""
        state = _make_state(in_dialog=True)  # RAM says dialog but context='navigation'
        assert routing_condition(state) == "nav_bot"

    def test_game_state_battle_routes_to_battle_bot(self):
        state = _make_state(game_state="battle")
        assert routing_condition(state) == "battle_bot"

    def test_game_state_dialog_alone_does_not_route_to_coms_bot(self):
        """game_state='dialog' alone (no context='dialogue') falls through to nav_bot."""
        state = _make_state(game_state="dialog")
        assert routing_condition(state) == "nav_bot"

    def test_normal_state_routes_to_nav_bot(self):
        state = _make_state()
        assert routing_condition(state) == "nav_bot"

    def test_battle_takes_priority_over_dialogue_context(self):
        """When in_battle and context='dialogue', battle wins."""
        state = _make_state(in_battle=True)
        state = {**state, "context": "dialogue"}
        assert routing_condition(state) == "battle_bot"

    def test_healing_context_routes_to_map_stitcher_relay(self):
        state = _make_state()
        state = {**state, "context": "healing_needed"}
        assert routing_condition(state) == "map_stitcher_relay"

    def test_healing_takes_priority_over_battle(self):
        """healing_needed context wins over in_battle."""
        state = _make_state(in_battle=True)
        state = {**state, "context": "healing_needed"}
        assert routing_condition(state) == "map_stitcher_relay"


# ---------------------------------------------------------------------------
# TestGraphCompiles
# ---------------------------------------------------------------------------

class TestGraphCompiles:
    def test_build_graph_returns_without_exception(self):
        from agent.graph.graph import build_graph
        mock_obj = MagicMock()
        mock_vlm = MagicMock()
        graph = build_graph(mock_obj, mock_vlm)
        assert graph is not None

    def test_compiled_graph_has_invoke_method(self):
        from agent.graph.graph import build_graph
        graph = build_graph(MagicMock(), MagicMock())
        assert callable(getattr(graph, "invoke", None))


# ---------------------------------------------------------------------------
# TestGraphEdges
# ---------------------------------------------------------------------------

class TestGraphEdges:
    """Verify the compiled graph exposes the expected nodes."""

    @pytest.fixture(scope="class")
    def compiled_graph(self):
        from agent.graph.graph import build_graph
        return build_graph(MagicMock(), MagicMock())

    def test_all_nodes_present(self, compiled_graph):
        node_names = set(compiled_graph.get_graph().nodes.keys())
        expected = {"dispatch", "nav_bot", "battle_bot", "coms_bot",
                    "map_stitcher_relay", "verification", "__start__", "__end__"}
        assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"


# ---------------------------------------------------------------------------
# TestFullGraphInvoke
# ---------------------------------------------------------------------------

class TestFullGraphInvoke:
    """End-to-end graph.invoke() with real nodes (mock external I/O)."""

    @pytest.fixture(autouse=True)
    def _patch_externals(self):
        """Patch heavy external calls so tests run without GBA hardware."""
        with (
            patch("agent.graph.nodes.battle_bot.get_battle_bot") as mock_bb,
            patch("agent.graph.nodes.nav_bot.pathfind_to_goal", return_value=["RIGHT"]),
            patch("agent.graph.nodes.coms_bot.get_opener_bot") as mock_ob,
            patch("agent.graph.nodes.coms_bot.wait_for_script_idle", return_value=None),
            patch("agent.graph.nodes.verification.MILESTONE_PROGRESSION", []),
        ):
            # Configure mock battle bot
            bb_instance = MagicMock()
            bb_instance.get_action.return_value = "SELECT_RUN"
            mock_bb.return_value = bb_instance

            # Configure mock opener bot
            ob_instance = MagicMock()
            ob_instance.should_handle.return_value = False
            mock_ob.return_value = ob_instance

            yield

    @pytest.fixture(scope="class")
    def graph(self):
        from agent.graph.graph import build_graph
        return build_graph(MagicMock(), MagicMock())

    def _base_state(self, **overrides) -> AgentState:
        base = AgentState(
            frame=None,
            state_data={"game": {"in_battle": False, "in_dialog": False,
                                 "game_state": "overworld"},
                        "player": {"position": {"x": 10, "y": 10}, "location": "ROUTE_102"},
                        "party": []},
            perception={},
            goal_coords=(8, 8),
            goal_location="OLDALE_TOWN",
            npc_coords=None,
            should_interact=False,
            milestone_index=0,
            context="navigation",
            reward=None,
            prev_state_snapshot=None,
            last_action=None,
            last_buttons=[],
            step_count=0,
            telemetry=None,
        )
        return {**base, **overrides}

    def test_battle_state_fires_battle_node(self, graph):
        state = self._base_state(
            state_data={"game": {"in_battle": True, "in_dialog": False,
                                 "game_state": "battle"},
                        "player": {}, "party": []}
        )
        result = graph.invoke(state)
        assert result["last_action"] == "BATTLE"

    def test_normal_state_fires_nav_node(self, graph):
        state = self._base_state()
        result = graph.invoke(state)
        assert result["last_action"] == "NAVIGATE"

    def test_last_buttons_populated_for_battle(self, graph):
        state = self._base_state(
            state_data={"game": {"in_battle": True, "in_dialog": False,
                                 "game_state": "battle"},
                        "player": {}, "party": []}
        )
        result = graph.invoke(state)
        assert isinstance(result["last_buttons"], list)
        assert len(result["last_buttons"]) > 0

    def test_last_buttons_populated_for_nav(self, graph):
        state = self._base_state()
        result = graph.invoke(state)
        assert isinstance(result["last_buttons"], list)
