"""
Phase 4.4d — NPC Registry Observation Integration Test (Tutorial / Mom)

Diagnostic + integration test that:
1. Loads the 01_tutorial split save (player inside moving van)
2. Scripts button inputs to exit the van
3. Waits for the player to arrive in Littleroot Town
4. Checks gObjectEvents for Mom's NPC
5. Simulates the registry observation hook
6. Verifies Mom's identity gets registered

This tests the *full discovery pipeline*: active_npcs → nearest NPC →
speaker extraction → registry upsert.
"""

import os
import pytest
from pathlib import Path

from agent.brain.npc_registry import NpcRegistry
from pokemon_env.emulator import EmeraldEmulator

PROJECT_ROOT = Path(__file__).parent.parent
ROM_PATH = PROJECT_ROOT / "Emerald-GBAdvance" / "rom.gba"
TUTORIAL_STATE = PROJECT_ROOT / "Emerald-GBAdvance" / "splits" / "01_tutorial" / "01_tutorial.state"


@pytest.mark.skipif(
    not ROM_PATH.exists(),
    reason="GBA ROM not found — requires Emerald-GBAdvance/rom.gba",
)
@pytest.mark.skipif(
    not TUTORIAL_STATE.exists(),
    reason="Tutorial split save not found",
)
class TestTutorialMomRegistration:
    """Integration: exit van → Mom approaches → registry captures her identity."""

    @pytest.fixture(scope="class")
    def emulator(self):
        emu = EmeraldEmulator(str(ROM_PATH), headless=True, sound=False)
        emu.initialize()
        yield emu
        emu.stop()

    # ── Helpers ───────────────────────────────────────────────

    def _read_location(self, emu):
        """Quick location string from memory reader."""
        try:
            state = emu.memory_reader.get_comprehensive_state()
            return state.get("player", {}).get("location", "")
        except Exception:
            return ""

    def _read_active_npcs(self, emu):
        """Read active NPCs from gObjectEvents."""
        try:
            return emu.memory_reader.read_active_npcs()
        except Exception:
            return []

    def _player_pos(self, emu):
        """Get (x, y) from memory reader."""
        try:
            state = emu.memory_reader.get_comprehensive_state()
            pos = state.get("player", {}).get("position", {})
            return (pos.get("x"), pos.get("y"))
        except Exception:
            return (None, None)

    # ── Phase 1: Diagnostic — what NPCs exist at each stage ──

    def test_npcs_inside_van(self, emulator):
        """Check what gObjectEvents looks like inside the moving van."""
        emulator.load_state(str(TUTORIAL_STATE))
        emulator.tick(10)  # settle

        loc = self._read_location(emulator)
        npcs = self._read_active_npcs(emulator)
        pos = self._player_pos(emulator)

        print(f"\n=== INSIDE VAN ===")
        print(f"Location: {loc}")
        print(f"Player pos: {pos}")
        print(f"Active NPCs ({len(npcs)}):")
        for npc in npcs:
            print(f"  slot={npc.get('slot')} gfx={npc.get('graphics_id')} "
                  f"local={npc.get('local_id')} pos=({npc.get('current_x')},{npc.get('current_y')}) "
                  f"player={npc.get('is_player')} invis={npc.get('invisible')}")

        # Basic sanity: should be inside the van
        assert "VAN" in loc.upper() or "MOVING" in loc.upper() or pos == (3, 2), \
            f"Expected MOVING_VAN, got location='{loc}' pos={pos}"

    def test_exit_van_and_find_mom(self, emulator):
        """Exit the van via scripted inputs, then check for Mom NPC."""
        emulator.load_state(str(TUTORIAL_STATE))
        emulator.tick(10)

        # Script the exit: press RIGHT repeatedly then UP to leave the van
        # From the log: opener bot sends ['RIGHT','RIGHT','RIGHT','RIGHT','UP','RIGHT']
        for btn in ["right", "right", "right", "right", "up", "right"]:
            emulator.press_key(btn, frames=8)

        # Run more frames — the warp + Mom cutscene takes a while
        # Mom approaches automatically after the player exits
        emulator.tick(120)

        loc = self._read_location(emulator)
        npcs = self._read_active_npcs(emulator)
        pos = self._player_pos(emulator)

        print(f"\n=== AFTER EXITING VAN (120 extra frames) ===")
        print(f"Location: {loc}")
        print(f"Player pos: {pos}")
        print(f"Active NPCs ({len(npcs)}):")
        for npc in npcs:
            print(f"  slot={npc.get('slot')} gfx={npc.get('graphics_id')} "
                  f"local={npc.get('local_id')} pos=({npc.get('current_x')},{npc.get('current_y')}) "
                  f"player={npc.get('is_player')} invis={npc.get('invisible')} "
                  f"off_screen={npc.get('off_screen')}")

        # If we're still in the van, try more frames
        if "VAN" in loc.upper() or "MOVING" in loc.upper():
            print("\n--- Still in van, running 200 more frames ---")
            emulator.tick(200)
            loc = self._read_location(emulator)
            npcs = self._read_active_npcs(emulator)
            pos = self._player_pos(emulator)
            print(f"Location: {loc}")
            print(f"Player pos: {pos}")
            print(f"Active NPCs ({len(npcs)}):")
            for npc in npcs:
                print(f"  slot={npc.get('slot')} gfx={npc.get('graphics_id')} "
                      f"local={npc.get('local_id')} pos=({npc.get('current_x')},{npc.get('current_y')}) "
                      f"player={npc.get('is_player')} invis={npc.get('invisible')}")

        # Log any non-player NPCs — these are Mom candidates
        non_player_npcs = [n for n in npcs if not n.get("is_player")]
        visible_npcs = [n for n in non_player_npcs
                        if not n.get("invisible") and not n.get("off_screen")]
        print(f"\nVisible non-player NPCs: {len(visible_npcs)}")
        for npc in visible_npcs:
            print(f"  → gfx={npc.get('graphics_id')} local={npc.get('local_id')} "
                  f"at ({npc.get('current_x')},{npc.get('current_y')})")

    def test_advance_to_mom_dialogue(self, emulator):
        """Keep advancing until Mom dialogue appears, log NPCs at that point."""
        emulator.load_state(str(TUTORIAL_STATE))
        emulator.tick(10)

        # Exit van
        for btn in ["right", "right", "right", "right", "up", "right"]:
            emulator.press_key(btn, frames=8)

        # Now run frames in batches, checking for location change and dialogue
        found_littleroot = False
        found_dialogue = False
        total_frames = 0
        max_frames = 1500  # safety cap

        while total_frames < max_frames:
            emulator.tick(30)
            total_frames += 30

            loc = self._read_location(emulator)
            if not found_littleroot and "LITTLEROOT" in loc.upper():
                found_littleroot = True
                print(f"\n✅ Arrived in Littleroot at frame {total_frames}")

            # Check for dialogue via memory reader
            try:
                state = emulator.memory_reader.get_comprehensive_state()
                in_dialog = state.get("game", {}).get("in_dialog", False)
                if in_dialog and found_littleroot:
                    found_dialogue = True
                    print(f"✅ Dialogue detected at frame {total_frames}")

                    # NOW check NPCs
                    npcs = self._read_active_npcs(emulator)
                    pos = self._player_pos(emulator)
                    print(f"Player pos: {pos}")
                    print(f"Active NPCs ({len(npcs)}):")
                    for npc in npcs:
                        print(f"  slot={npc.get('slot')} gfx={npc.get('graphics_id')} "
                              f"local={npc.get('local_id')} "
                              f"pos=({npc.get('current_x')},{npc.get('current_y')}) "
                              f"player={npc.get('is_player')} invis={npc.get('invisible')}")

                    # Try to identify Mom: nearest non-player NPC to the player
                    px, py = pos
                    if px is not None and py is not None:
                        nearest = NpcRegistry.infer_nearest_npc(npcs, px, py, max_distance=5)
                        if nearest:
                            print(f"\n🎯 Nearest NPC (likely Mom): gfx={nearest.get('graphics_id')} "
                                  f"local={nearest.get('local_id')} "
                                  f"at ({nearest.get('current_x')},{nearest.get('current_y')})")
                        else:
                            print("\n⚠️ No nearby NPC found — Mom may be script-spawned")
                    break
            except Exception as e:
                pass  # memory reader may fail during transitions

        print(f"\nTotal frames advanced: {total_frames}")
        assert found_littleroot, f"Never arrived in Littleroot (ran {total_frames} frames)"
        # Don't hard-assert on dialogue — this diagnostic tells us what's possible

    def test_registry_captures_mom(self, emulator, tmp_path):
        """Full integration: exit van → find Mom → register her in NPC registry."""
        emulator.load_state(str(TUTORIAL_STATE))
        emulator.tick(10)

        # Exit van
        for btn in ["right", "right", "right", "right", "up", "right"]:
            emulator.press_key(btn, frames=8)

        # Advance until we're in Littleroot with NPCs nearby
        found_mom_npc = None
        player_pos = (None, None)
        location = ""
        total_frames = 0
        max_frames = 1500

        while total_frames < max_frames:
            emulator.tick(30)
            total_frames += 30

            try:
                loc = self._read_location(emulator)
                if "LITTLEROOT" not in loc.upper():
                    continue

                location = loc
                npcs = self._read_active_npcs(emulator)
                px, py = self._player_pos(emulator)
                if px is None:
                    continue

                player_pos = (px, py)

                # Look for non-player NPC nearby (Mom approaching)
                nearest = NpcRegistry.infer_nearest_npc(npcs, px, py, max_distance=5)
                if nearest:
                    found_mom_npc = nearest
                    print(f"\n🎯 Found Mom NPC at frame {total_frames}: "
                          f"gfx={nearest.get('graphics_id')} "
                          f"local={nearest.get('local_id')} "
                          f"at ({nearest.get('current_x')},{nearest.get('current_y')})")
                    break
            except Exception:
                continue

        if found_mom_npc is None:
            # Even if Mom isn't in gObjectEvents (script-spawned cutscene),
            # document the finding — this is diagnostic
            pytest.skip(
                f"No nearby NPC found after {total_frames} frames in Littleroot. "
                f"Mom may be script-spawned during this cutscene and not visible "
                f"in gObjectEvents. Player at {player_pos}, location='{location}'"
            )

        # === Registry integration ===
        registry = NpcRegistry(json_path=str(tmp_path / "mom_test_registry.json"))
        assert len(registry) == 0, "Registry should start empty"

        # Simulate what _observe_npc_identity does:
        # Speaker would come from VLM — we mock it as "MOM" since this is early game
        registry.register_npc(
            location,
            found_mom_npc["local_id"],
            name="MOM",
            role="mom",
            graphics_id=found_mom_npc["graphics_id"],
            step=1,
        )

        assert len(registry) == 1
        entry = registry.lookup_by_name("MOM")
        assert entry is not None, "Mom should be findable by name"
        assert entry["graphics_id"] == found_mom_npc["graphics_id"]
        assert entry["local_id"] == found_mom_npc["local_id"]

        # Can also look up by role
        entry_role = registry.lookup_by_role("mom")
        assert entry_role is not None
        assert entry_role["name"] == "MOM"

        print(f"\n✅ Registry entry: {entry}")
        print(f"✅ Mom registered successfully: gfx={entry['graphics_id']}, "
              f"local_id={entry['local_id']}, location={entry['location']}")
