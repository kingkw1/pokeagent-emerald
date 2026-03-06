"""
Phase 4.4b — NPC Coordinate Resolution Tests

Tests for ``ObjectiveManager._resolve_npc_coords()`` — the dynamic NPC
coordinate lookup that replaced hardcoded ``npc_coords`` values.

Two test classes:

- ``TestResolveNpcCoordsPure``:  Unit tests using synthetic NPC lists
  (no emulator needed).
- ``TestResolveNpcCoordsIntegration``:  Integration tests against the
  rival_battle_save state to verify the full pipeline from
  ``read_active_npcs()`` → ``active_npcs`` in state_data → resolver.
"""

import os
import pytest
from pathlib import Path

from agent.objective_manager import ObjectiveManager


# ===========================================================================
# Unit Tests — _resolve_npc_coords with synthetic data
# ===========================================================================

class TestResolveNpcCoordsPure:
    """Unit tests for _resolve_npc_coords (no emulator required)."""

    # Sample NPC data mimicking read_active_npcs() output
    SAMPLE_NPCS = [
        {
            "slot": 8, "graphics_id": 0, "local_id": 0,
            "current_x": 5, "current_y": 3,
            "is_player": True, "invisible": False, "off_screen": False,
        },
        {
            "slot": 9, "graphics_id": 105, "local_id": 2,
            "current_x": 10, "current_y": 3,
            "is_player": False, "invisible": False, "off_screen": False,
        },
        {
            "slot": 10, "graphics_id": 29, "local_id": 1,
            "current_x": 7, "current_y": 3,
            "is_player": False, "invisible": False, "off_screen": False,
        },
    ]

    @pytest.fixture
    def om(self):
        """ObjectiveManager instance with no external dependencies."""
        return ObjectiveManager()

    def _state(self, npcs=None):
        return {"active_npcs": npcs if npcs is not None else self.SAMPLE_NPCS}

    # --- Basic matching ---

    def test_find_by_graphics_id(self, om):
        coords = om._resolve_npc_coords(
            self._state(), graphics_id=105
        )
        assert coords == (10, 3)

    def test_find_by_local_id(self, om):
        coords = om._resolve_npc_coords(
            self._state(), local_id=2
        )
        assert coords == (10, 3)

    def test_find_by_both_criteria(self, om):
        coords = om._resolve_npc_coords(
            self._state(), graphics_id=105, local_id=2
        )
        assert coords == (10, 3)

    def test_mismatch_criteria_uses_fallback(self, om):
        """When graphics_id and local_id don't match the same NPC, return fallback."""
        coords = om._resolve_npc_coords(
            self._state(), graphics_id=105, local_id=1, fallback=(99, 99)
        )
        assert coords == (99, 99)

    # --- Player is always skipped ---

    def test_skips_player(self, om):
        """Player NPC (is_player=True) should never be returned even if criteria match."""
        coords = om._resolve_npc_coords(
            self._state(), graphics_id=0, local_id=0, fallback=(0, 0)
        )
        # Player has gfx=0, local_id=0.  Should be skipped → fallback
        assert coords == (0, 0)

    # --- Invisible / off-screen filtering ---

    def test_skips_invisible_npc(self, om):
        npcs = [
            {
                "slot": 1, "graphics_id": 50, "local_id": 3,
                "current_x": 20, "current_y": 20,
                "is_player": False, "invisible": True, "off_screen": False,
            },
        ]
        coords = om._resolve_npc_coords(
            {"active_npcs": npcs}, graphics_id=50, fallback=(0, 0)
        )
        assert coords == (0, 0)

    def test_skips_off_screen_npc(self, om):
        npcs = [
            {
                "slot": 1, "graphics_id": 50, "local_id": 3,
                "current_x": 20, "current_y": 20,
                "is_player": False, "invisible": False, "off_screen": True,
            },
        ]
        coords = om._resolve_npc_coords(
            {"active_npcs": npcs}, graphics_id=50, fallback=(0, 0)
        )
        assert coords == (0, 0)

    # --- Fallback behaviour ---

    def test_returns_fallback_when_not_found(self, om):
        coords = om._resolve_npc_coords(
            self._state(), graphics_id=999, fallback=(42, 42)
        )
        assert coords == (42, 42)

    def test_returns_none_when_not_found_no_fallback(self, om):
        coords = om._resolve_npc_coords(
            self._state(), graphics_id=999
        )
        assert coords is None

    def test_returns_fallback_when_no_npcs(self, om):
        coords = om._resolve_npc_coords(
            {"active_npcs": []}, graphics_id=105, fallback=(10, 3)
        )
        assert coords == (10, 3)

    def test_returns_fallback_when_active_npcs_missing(self, om):
        """state_data with no 'active_npcs' key at all."""
        coords = om._resolve_npc_coords(
            {}, graphics_id=105, fallback=(10, 3)
        )
        assert coords == (10, 3)

    # --- Edge: no criteria given ---

    def test_no_criteria_returns_fallback(self, om):
        """If neither graphics_id nor local_id is specified, return fallback."""
        coords = om._resolve_npc_coords(
            self._state(), fallback=(0, 0)
        )
        assert coords == (0, 0)

    # --- Second NPC in list ---

    def test_find_nurse_by_graphics_id(self, om):
        coords = om._resolve_npc_coords(
            self._state(), graphics_id=29
        )
        assert coords == (7, 3)


# ===========================================================================
# Integration Tests — real emulator + rival_battle_save.state
# ===========================================================================

ROM_PATH = Path(__file__).parent.parent / "Emerald-GBAdvance" / "rom.gba"


@pytest.mark.skipif(
    not ROM_PATH.exists(),
    reason="GBA ROM not found — integration tests require Emerald-GBAdvance/rom.gba",
)
class TestResolveNpcCoordsIntegration:
    """Integration: read_active_npcs() flows through state_data to _resolve_npc_coords()."""

    @pytest.fixture(scope="class")
    def emulator(self):
        from pokemon_env.emulator import EmeraldEmulator
        emu = EmeraldEmulator(str(ROM_PATH), headless=True, sound=False)
        emu.initialize()
        yield emu
        emu.stop()

    @pytest.fixture
    def rival_state_data(self, emulator):
        """Load rival_battle_save, read comprehensive state including active_npcs."""
        emulator.load_state("tests/save_states/rival_battle_save.state")
        for _ in range(5):
            emulator.core.run_frame()
        state = emulator.memory_reader.get_comprehensive_state()
        return state

    def test_active_npcs_in_state(self, rival_state_data):
        """get_comprehensive_state() should include 'active_npcs' key."""
        assert "active_npcs" in rival_state_data
        assert isinstance(rival_state_data["active_npcs"], list)
        assert len(rival_state_data["active_npcs"]) >= 2  # player + rival

    def test_resolve_rival_from_state(self, rival_state_data):
        """_resolve_npc_coords should find the rival May (gfx=105)."""
        om = ObjectiveManager()
        coords = om._resolve_npc_coords(
            rival_state_data, graphics_id=105, local_id=2
        )
        assert coords is not None, "Rival not found in active_npcs"
        # Rival should be near (10, 3)
        assert abs(coords[0] - 10) <= 2, f"Rival X={coords[0]}, expected ~10"
        assert abs(coords[1] - 3) <= 2, f"Rival Y={coords[1]}, expected ~3"

    def test_resolve_nonexistent_npc_returns_fallback(self, rival_state_data):
        """NPC with non-matching criteria should return fallback."""
        om = ObjectiveManager()
        coords = om._resolve_npc_coords(
            rival_state_data, graphics_id=999, fallback=(99, 99)
        )
        assert coords == (99, 99)

    def test_rival_directive_uses_dynamic_coords(self, rival_state_data):
        """The rival battle directive should use resolved coords, not hardcoded."""
        # Set up milestones so ROUTE_103 is complete but RIVAL_BATTLE_1 is not
        rival_state_data['milestones'] = {
            'GAME_RUNNING': {'completed': True},
            'PLAYER_NAME_SET': {'completed': True},
            'INTRO_CUTSCENE_COMPLETE': {'completed': True},
            'LITTLEROOT_TOWN': {'completed': True},
            'PLAYER_HOUSE_ENTERED': {'completed': True},
            'PLAYER_BEDROOM': {'completed': True},
            'RIVAL_HOUSE': {'completed': True},
            'RIVAL_BEDROOM': {'completed': True},
            'ROUTE_101': {'completed': True},
            'STARTER_CHOSEN': {'completed': True},
            'BIRCH_LAB_VISITED': {'completed': True},
            'OLDALE_TOWN': {'completed': True},
            'ROUTE_103': {'completed': True},
            # RIVAL_BATTLE_1 NOT set → this is the next target
        }

        om = ObjectiveManager()
        directive = om.get_next_action_directive(rival_state_data)
        assert directive is not None, "Expected a rival battle directive"
        assert directive.get('should_interact') is True
        assert directive.get('npc_coords') is not None

        npc_coords = directive['npc_coords']
        # Should be the dynamically resolved coords (≈ 10, 3), not a fixed literal
        assert abs(npc_coords[0] - 10) <= 2, f"npc_coords X={npc_coords[0]}, expected ~10"
        assert abs(npc_coords[1] - 3) <= 2, f"npc_coords Y={npc_coords[1]}, expected ~3"

        # goal_coords should be one tile west of the rival
        goal_coords = directive.get('goal_coords')
        assert goal_coords is not None
        assert goal_coords[0] == npc_coords[0] - 1, \
            f"goal_x={goal_coords[0]} should be rival_x-1={npc_coords[0]-1}"
        assert goal_coords[1] == npc_coords[1], \
            f"goal_y={goal_coords[1]} should match rival_y={npc_coords[1]}"
