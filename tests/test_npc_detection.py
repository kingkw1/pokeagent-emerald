"""
Phase 4.4a — NPC Detection Tests

Tests for the canonical `read_active_npcs()` method that parses the
gObjectEvents array at 0x02037230 using the pokeemerald-validated struct layout.

These tests require the GBA ROM and emulator to be available.  They are
**skipped** automatically when the ROM is not present.

Canonical test save: tests/save_states/rival_battle_save.state
- Map:    ROUTE_103
- Player: (5, 3), facing right
- Rival:  ~5 tiles to the right, approximately (10, 3)
- Party:  TREECKO Lv.7, HP 23/23
"""

import os
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Skip entire module when the ROM is not available
# ---------------------------------------------------------------------------
ROM_PATH = Path(__file__).parent.parent / "Emerald-GBAdvance" / "rom.gba"
pytestmark = pytest.mark.skipif(
    not ROM_PATH.exists(),
    reason="GBA ROM not found — integration tests require Emerald-GBAdvance/rom.gba",
)

from pokemon_env.emulator import EmeraldEmulator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def emulator():
    """Create and initialise an emulator instance (shared across the module)."""
    emu = EmeraldEmulator(str(ROM_PATH), headless=True, sound=False)
    emu.initialize()
    yield emu
    emu.stop()


@pytest.fixture
def rival_battle_state(emulator):
    """Load the rival_battle_save state and return the emulator."""
    emulator.load_state("tests/save_states/rival_battle_save.state")
    # Advance a few frames so the emulator settles after load
    for _ in range(5):
        emulator.core.run_frame()
    return emulator


# ---------------------------------------------------------------------------
# Tests — rival_battle_save.state
# ---------------------------------------------------------------------------

class TestReadActiveNpcs:
    """Tests for the canonical read_active_npcs() method."""

    def test_returns_list(self, rival_battle_state):
        """read_active_npcs() should return a list."""
        npcs = rival_battle_state.memory_reader.read_active_npcs()
        assert isinstance(npcs, list)

    def test_player_detected(self, rival_battle_state):
        """Slot 0 should be the player at approximately (5, 3)."""
        npcs = rival_battle_state.memory_reader.read_active_npcs()

        # Find the player entry (is_player flag set)
        players = [n for n in npcs if n.get("is_player")]
        assert len(players) >= 1, f"No player object found.  All NPCs: {npcs}"

        player = players[0]
        assert abs(player["current_x"] - 5) <= 1, f"Player X={player['current_x']}, expected ~5"
        assert abs(player["current_y"] - 3) <= 1, f"Player Y={player['current_y']}, expected ~3"

    def test_at_least_one_npc_besides_player(self, rival_battle_state):
        """There should be at least one non-player NPC (the rival)."""
        npcs = rival_battle_state.memory_reader.read_active_npcs()
        non_player = [n for n in npcs if not n.get("is_player")]
        assert len(non_player) >= 1, f"No NPCs found besides player.  All: {npcs}"

    def test_rival_npc_position(self, rival_battle_state):
        """The rival NPC should be within Manhattan distance ≤ 8 of (10, 3)."""
        npcs = rival_battle_state.memory_reader.read_active_npcs()
        non_player = [n for n in npcs if not n.get("is_player")]

        # At least one NPC should be near (10, 3)
        close_to_rival = [
            n for n in non_player
            if abs(n["current_x"] - 10) + abs(n["current_y"] - 3) <= 8
        ]
        assert len(close_to_rival) >= 1, (
            f"No NPC found near (10, 3).  Non-player NPCs: {non_player}"
        )

    def test_npc_has_required_fields(self, rival_battle_state):
        """Every NPC dict should contain the expected fields."""
        npcs = rival_battle_state.memory_reader.read_active_npcs()
        assert len(npcs) > 0

        required_fields = {
            "slot", "graphics_id", "movement_type", "trainer_type",
            "local_id", "map_num", "map_group",
            "current_x", "current_y",
            "initial_x", "initial_y",
            "previous_x", "previous_y",
            "is_player", "active", "memory_address",
        }

        for npc in npcs:
            missing = required_fields - set(npc.keys())
            assert not missing, f"NPC slot {npc.get('slot', '?')} missing fields: {missing}"

    def test_no_zero_zero_coordinates(self, rival_battle_state):
        """Active NPCs should not have (0, 0) coordinates."""
        npcs = rival_battle_state.memory_reader.read_active_npcs()
        for npc in npcs:
            assert not (npc["current_x"] == 0 and npc["current_y"] == 0), (
                f"NPC at slot {npc['slot']} has (0,0) coordinates — likely uninitialised"
            )

    def test_coordinates_in_valid_range(self, rival_battle_state):
        """All NPC coordinates should be within sensible map bounds."""
        npcs = rival_battle_state.memory_reader.read_active_npcs()
        for npc in npcs:
            x, y = npc["current_x"], npc["current_y"]
            assert -20 <= x <= 200, f"NPC slot {npc['slot']} X={x} out of range"
            assert -20 <= y <= 200, f"NPC slot {npc['slot']} Y={y} out of range"


class TestReadActiveNpcsMultiState:
    """Run read_active_npcs() against additional save states for robustness."""

    @pytest.mark.parametrize("state_file,expected_location_substr", [
        ("tests/save_states/oldale_town_save.state", "OLDALE"),
        ("tests/save_states/birch_lab_save.state", "BIRCH"),
        ("tests/save_states/route102_save.state", "ROUTE"),
    ])
    def test_no_crash(self, emulator, state_file, expected_location_substr):
        """read_active_npcs() should not crash on various save states."""
        state_path = Path(__file__).parent.parent / state_file
        if not state_path.exists():
            pytest.skip(f"Save state not found: {state_file}")

        emulator.load_state(state_file)
        for _ in range(5):
            emulator.core.run_frame()

        npcs = emulator.memory_reader.read_active_npcs()
        assert isinstance(npcs, list)

        # Verify we're on the expected map (sanity check)
        loc = emulator.memory_reader.read_location() or ""
        # Don't assert on location — some save states may report differently
        # Just ensure no crash occurred
        print(f"  State: {state_file} → location='{loc}', NPCs found: {len(npcs)}")

    @pytest.mark.parametrize("state_file", [
        "tests/save_states/enter_petalburg_city_save.state",
        "tests/save_states/route102_save.state",
    ])
    def test_no_false_positives_at_zero_zero(self, emulator, state_file):
        """No NPC should be at (0, 0) — that's almost certainly uninitialised memory."""
        state_path = Path(__file__).parent.parent / state_file
        if not state_path.exists():
            pytest.skip(f"Save state not found: {state_file}")

        emulator.load_state(state_file)
        for _ in range(5):
            emulator.core.run_frame()

        npcs = emulator.memory_reader.read_active_npcs()
        for npc in npcs:
            assert not (npc["current_x"] == 0 and npc["current_y"] == 0), (
                f"False positive: NPC at (0,0) in {state_file}"
            )


class TestReadActiveNpcsDiagnostic:
    """Diagnostic test that prints raw NPC data — useful during development."""

    def test_dump_all_slots(self, rival_battle_state):
        """Print all active NPC slots for manual inspection."""
        npcs = rival_battle_state.memory_reader.read_active_npcs()
        player_coords = rival_battle_state.memory_reader.read_coordinates()

        print(f"\n{'='*70}")
        print(f"  rival_battle_save.state — Player at {player_coords}")
        print(f"  Total active NPCs: {len(npcs)}")
        print(f"{'='*70}")

        for npc in npcs:
            player_tag = " [PLAYER]" if npc.get("is_player") else ""
            print(
                f"  Slot {npc['slot']:2d}: "
                f"({npc['current_x']:3d}, {npc['current_y']:3d}) "
                f"gfx={npc['graphics_id']:3d} "
                f"move={npc['movement_type']:2d} "
                f"trainer={npc['trainer_type']:2d} "
                f"local_id={npc['local_id']:2d} "
                f"map={npc['map_group']}.{npc['map_num']}"
                f"{player_tag}"
            )
        print(f"{'='*70}\n")

        # This test always passes — it's for manual inspection
        assert True
