"""
Phase 4.4e — NPC Obstacle Tests

Tests for ``update_npc_obstacles()`` in ``agent/pathfinding.py`` and its
integration with the ``is_walkable()`` checks inside pathfinding functions.

Verifies that:
- NPC positions are correctly added to ``_npc_occupied_tiles``
- Player, invisible, and off-screen NPCs are excluded
- Target NPC exclusion (``exclude_coords``) works
- ``is_walkable`` functions correctly block NPC-occupied tiles
- Obstacle set is rebuilt (ephemeral) on every call
"""

import pytest
from unittest.mock import patch

import agent.pathfinding as pf
from agent.pathfinding import update_npc_obstacles


# ===========================================================================
# Sample NPC data
# ===========================================================================

SAMPLE_NPCS = [
    {
        "slot": 0, "graphics_id": 0, "local_id": 0,
        "current_x": 5, "current_y": 3,
        "is_player": True, "invisible": False, "off_screen": False,
    },
    {
        "slot": 1, "graphics_id": 105, "local_id": 2,
        "current_x": 10, "current_y": 3,
        "is_player": False, "invisible": False, "off_screen": False,
    },
    {
        "slot": 2, "graphics_id": 29, "local_id": 1,
        "current_x": 7, "current_y": 4,
        "is_player": False, "invisible": False, "off_screen": False,
    },
]


def _state(location="ROUTE_103", npcs=None):
    return {
        "active_npcs": npcs if npcs is not None else SAMPLE_NPCS,
        "player": {"position": {"x": 5, "y": 3}, "location": location},
    }


# ===========================================================================
# Core update_npc_obstacles tests
# ===========================================================================

class TestUpdateNpcObstacles:
    """Unit tests for the update_npc_obstacles function."""

    def teardown_method(self):
        """Reset module-level state after each test."""
        pf._npc_occupied_tiles = set()

    def test_populates_npc_tiles(self):
        """Visible non-player NPCs should appear in _npc_occupied_tiles."""
        update_npc_obstacles(_state())

        assert (10, 3, "ROUTE_103") in pf._npc_occupied_tiles
        assert (7, 4, "ROUTE_103") in pf._npc_occupied_tiles
        assert len(pf._npc_occupied_tiles) == 2

    def test_skips_player(self):
        """The player's own object event should never be an obstacle."""
        update_npc_obstacles(_state())

        assert (5, 3, "ROUTE_103") not in pf._npc_occupied_tiles

    def test_skips_invisible_npc(self):
        """Invisible NPCs should not block tiles."""
        npcs = [
            {
                "slot": 1, "graphics_id": 50, "local_id": 3,
                "current_x": 20, "current_y": 20,
                "is_player": False, "invisible": True, "off_screen": False,
            },
        ]
        update_npc_obstacles(_state(npcs=npcs))

        assert len(pf._npc_occupied_tiles) == 0

    def test_skips_off_screen_npc(self):
        """Off-screen NPCs should not block tiles."""
        npcs = [
            {
                "slot": 1, "graphics_id": 50, "local_id": 3,
                "current_x": 20, "current_y": 20,
                "is_player": False, "invisible": False, "off_screen": True,
            },
        ]
        update_npc_obstacles(_state(npcs=npcs))

        assert len(pf._npc_occupied_tiles) == 0

    def test_exclude_coords_target_npc(self):
        """Target NPC tile should be excluded when navigating TO that NPC."""
        update_npc_obstacles(_state(), exclude_coords=(10, 3))

        # Rival at (10, 3) excluded, nurse at (7, 4) still blocked
        assert (10, 3, "ROUTE_103") not in pf._npc_occupied_tiles
        assert (7, 4, "ROUTE_103") in pf._npc_occupied_tiles
        assert len(pf._npc_occupied_tiles) == 1

    def test_exclude_coords_no_false_exclusion(self):
        """Exclude only matches exact coords, not other NPCs."""
        update_npc_obstacles(_state(), exclude_coords=(99, 99))

        # No NPC at (99, 99), so both NPCs are still blocked
        assert len(pf._npc_occupied_tiles) == 2

    def test_ephemeral_rebuild(self):
        """Calling update_npc_obstacles again replaces previous set entirely."""
        update_npc_obstacles(_state())
        assert len(pf._npc_occupied_tiles) == 2

        # Second call with no NPCs → set should be empty
        update_npc_obstacles(_state(npcs=[]))
        assert len(pf._npc_occupied_tiles) == 0

    def test_empty_active_npcs(self):
        """Empty NPC list → empty obstacle set."""
        update_npc_obstacles(_state(npcs=[]))

        assert len(pf._npc_occupied_tiles) == 0

    def test_no_active_npcs_key(self):
        """Missing active_npcs key entirely → empty obstacle set."""
        update_npc_obstacles({"player": {"position": {"x": 5, "y": 3}, "location": "TEST"}})

        assert len(pf._npc_occupied_tiles) == 0

    def test_npc_missing_coords_skipped(self):
        """NPC entries without current_x/current_y should be skipped."""
        npcs = [
            {
                "slot": 1, "graphics_id": 50, "local_id": 3,
                "is_player": False, "invisible": False, "off_screen": False,
                # No current_x, current_y
            },
        ]
        update_npc_obstacles(_state(npcs=npcs))

        assert len(pf._npc_occupied_tiles) == 0

    def test_location_tag_matches_player_location(self):
        """Obstacle tiles should be tagged with the player's current location."""
        update_npc_obstacles(_state(location="OLDALE_TOWN"))

        assert (10, 3, "OLDALE_TOWN") in pf._npc_occupied_tiles
        assert (10, 3, "ROUTE_103") not in pf._npc_occupied_tiles

    def test_multiple_npcs_same_tile(self):
        """Two NPCs at same position should produce one obstacle entry (set dedup)."""
        npcs = [
            {
                "slot": 1, "graphics_id": 50, "local_id": 1,
                "current_x": 10, "current_y": 5,
                "is_player": False, "invisible": False, "off_screen": False,
            },
            {
                "slot": 2, "graphics_id": 60, "local_id": 2,
                "current_x": 10, "current_y": 5,
                "is_player": False, "invisible": False, "off_screen": False,
            },
        ]
        update_npc_obstacles(_state(npcs=npcs))

        assert (10, 5, "ROUTE_103") in pf._npc_occupied_tiles
        assert len(pf._npc_occupied_tiles) == 1


# ===========================================================================
# A* routing integration — NPC obstacle avoidance
# ===========================================================================

class TestAstarNpcAvoidance:
    """Verify A* actually routes AROUND NPC-occupied tiles."""

    def teardown_method(self):
        pf._npc_occupied_tiles = set()
        pf._dynamically_blocked_tiles = set()

    def _make_corridor_grid(self, blocked_tile=None):
        """
        Build a 5-wide, 5-tall corridor as a location_grid in world coords.

        Layout (walkable '.' everywhere):
            (0,0) (1,0) (2,0) (3,0) (4,0)
            (0,1) (1,1) (2,1) (3,1) (4,1)
            (0,2) (1,2) (2,2) (3,2) (4,2)   ← player at (0,2)
            (0,3) (1,3) (2,3) (3,3) (4,3)
            (0,4) (1,4) (2,4) (3,4) (4,4)

        Goal: (4,2) — due east of player.
        If blocked_tile is set, that world coord becomes '#' (wall).
        """
        grid = {}
        for x in range(5):
            for y in range(5):
                grid[(x, y)] = '.'
        if blocked_tile:
            grid[blocked_tile] = '#'
        bounds = {"min_x": 0, "max_x": 4, "min_y": 0, "max_y": 4}
        return grid, bounds

    def test_straight_path_no_npc(self):
        """Without NPC obstacles, A* should go east directly."""
        grid, bounds = self._make_corridor_grid()
        result = pf._astar_pathfind_with_grid_data(
            location_grid=grid,
            bounds=bounds,
            current_pos=(0, 2),
            location="TEST",
            goal_direction="east",
            goal_coords=(4, 2),
        )
        # Returns batched steps list; all should be RIGHT (direct path)
        assert isinstance(result, list)
        assert all(s == "RIGHT" for s in result), f"Expected all RIGHT, got {result}"
        assert len(result) == 4  # 4 tiles east

    def test_npc_blocks_direct_path(self):
        """NPC on (1,2) should force A* to detour up or down, not RIGHT into NPC."""
        grid, bounds = self._make_corridor_grid()
        # Place NPC obstacle at (1, 2) — directly east of player at (0, 2)
        pf._npc_occupied_tiles = {(1, 2, "TEST")}

        result = pf._astar_pathfind_with_grid_data(
            location_grid=grid,
            bounds=bounds,
            current_pos=(0, 2),
            location="TEST",
            goal_direction="east",
            goal_coords=(4, 2),
        )
        # Must detour: first step UP or DOWN, and path must NOT pass through (1,2)
        assert isinstance(result, list)
        assert result[0] in ("UP", "DOWN"), (
            f"Expected first step UP or DOWN to detour around NPC at (1,2), got {result[0]}"
        )
        # Verify the full path avoids the NPC tile by simulating
        x, y = 0, 2
        for step in result:
            if step == "UP": y -= 1
            elif step == "DOWN": y += 1
            elif step == "LEFT": x -= 1
            elif step == "RIGHT": x += 1
            assert (x, y) != (1, 2), f"Path walks through NPC at (1,2)! Step sequence: {result}"
        # Should arrive at goal
        assert (x, y) == (4, 2), f"Path didn't reach goal (4,2), ended at ({x},{y})"

    def test_npc_on_goal_tile_excluded(self):
        """If the goal tile has an NPC, A* should still route to it (talk-to-NPC case)."""
        grid, bounds = self._make_corridor_grid()
        # Place NPC obstacle on the goal tile (4, 2)
        pf._npc_occupied_tiles = {(4, 2, "TEST")}
        result = pf._astar_pathfind_with_grid_data(
            location_grid=grid,
            bounds=bounds,
            current_pos=(0, 2),
            location="TEST",
            goal_direction="east",
            goal_coords=(4, 2),
        )
        # A* has special logic allowing non-walkable goal tiles — should still go RIGHT
        assert isinstance(result, list)
        assert result[0] == "RIGHT"
