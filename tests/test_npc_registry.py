"""
Phase 4.4d — NPC Identity Registry Tests

Tests for ``NpcRegistry`` — the adaptive NPC identity discovery system
that replaces hardcoded ``graphics_id`` constants.

Test classes:
- ``TestNpcRegistryCore``:       CRUD, persistence, lookup by role/name
- ``TestNpcRegistryColdStart``:  Cold-start heuristic (nearest NPC inference)
- ``TestRegistryResolveIntegration``: Registry wired into ``_resolve_npc_coords``
"""

import json
import os
import tempfile
import pytest

from agent.brain.npc_registry import NpcRegistry, _canon_role, _canon_name, ROLE_ALIASES
from agent.objective_manager import ObjectiveManager


# ===========================================================================
# Unit Tests — NpcRegistry core
# ===========================================================================

class TestNpcRegistryCore:
    """CRUD, persistence, and lookup tests (no emulator)."""

    @pytest.fixture
    def tmp_json(self, tmp_path):
        """Provide a temp path for the registry JSON file."""
        return str(tmp_path / "test_registry.json")

    @pytest.fixture
    def registry(self, tmp_json):
        return NpcRegistry(json_path=tmp_json)

    # --- Registration ---

    def test_register_and_lookup_by_role(self, registry):
        registry.register_npc(
            "ROUTE_103", 2,
            name="May", role="rival", graphics_id=105, step=10,
        )
        entry = registry.lookup_by_role("rival")
        assert entry is not None
        assert entry["name"] == "May"
        assert entry["graphics_id"] == 105
        assert entry["local_id"] == 2
        assert entry["location"] == "ROUTE_103"

    def test_register_and_lookup_by_name(self, registry):
        registry.register_npc(
            "PETALBURG_CITY_GYM", 1,
            name="Norman", role="gym_leader_norman", graphics_id=88,
        )
        entry = registry.lookup_by_name("Norman")
        assert entry is not None
        assert entry["role"] == "gym_leader_norman"

    def test_lookup_by_name_case_insensitive(self, registry):
        registry.register_npc("X", 1, name="Nurse Joy", role="nurse", graphics_id=29)
        assert registry.lookup_by_name("NURSE JOY") is not None
        assert registry.lookup_by_name("nurse joy") is not None

    def test_lookup_miss_returns_none(self, registry):
        assert registry.lookup_by_role("unknown_role") is None
        assert registry.lookup_by_name("Ghost") is None

    def test_role_aliases(self, registry):
        registry.register_npc("X", 1, name="May", role="rival", graphics_id=105)
        # "may" is an alias for "rival"
        assert registry.lookup_by_role("may") is not None
        assert registry.lookup_by_role("the rival") is not None

    # --- Upsert / merge semantics ---

    def test_upsert_preserves_existing_fields(self, registry):
        """Second registration with None fields should NOT overwrite existing values."""
        registry.register_npc("R103", 2, name="May", role="rival", graphics_id=105)
        # Update without name — should preserve "May"
        registry.register_npc("R103", 2, graphics_id=105)
        entry = registry.lookup_by_role("rival")
        assert entry["name"] == "May"

    def test_upsert_overwrites_with_new_values(self, registry):
        registry.register_npc("R103", 2, name="May", role="rival", graphics_id=105)
        registry.register_npc("R103", 2, name="May (updated)", role="rival", graphics_id=105)
        entry = registry.lookup_by_role("rival")
        assert entry["name"] == "May (updated)"

    # --- Location-scoped lookup ---

    def test_lookup_by_role_prefers_same_location(self, registry):
        registry.register_npc("OLDALE_POKECENTER", 1, name="Nurse A", role="nurse", graphics_id=29)
        registry.register_npc("PETALBURG_POKECENTER", 1, name="Nurse B", role="nurse", graphics_id=29)

        # When location matches, prefer that entry
        entry = registry.lookup_by_role("nurse", location="PETALBURG_POKECENTER")
        assert entry["name"] == "Nurse B"

        # When no location, return first match
        entry_any = registry.lookup_by_role("nurse")
        assert entry_any is not None

    # --- Persistence ---

    def test_persistence_across_instances(self, tmp_json):
        """Registry should survive being recreated from the same JSON file."""
        reg1 = NpcRegistry(json_path=tmp_json)
        reg1.register_npc("ROUTE_103", 2, name="May", role="rival", graphics_id=105)

        reg2 = NpcRegistry(json_path=tmp_json)
        entry = reg2.lookup_by_role("rival")
        assert entry is not None
        assert entry["name"] == "May"

    def test_clear_wipes_all(self, registry, tmp_json):
        registry.register_npc("X", 1, name="Test", role="test", graphics_id=1)
        assert len(registry) == 1
        registry.clear()
        assert len(registry) == 0
        # JSON file should be empty too
        reg2 = NpcRegistry(json_path=tmp_json)
        assert len(reg2) == 0

    def test_len_and_contains(self, registry):
        assert len(registry) == 0
        registry.register_npc("X", 1, name="A", role="a", graphics_id=1)
        assert len(registry) == 1
        assert "X:1" in registry

    # --- Canon helpers ---

    def test_canon_role(self):
        assert _canon_role("Rival") == "rival"
        assert _canon_role("May") == "rival"
        assert _canon_role("nurse joy") == "nurse"
        assert _canon_role("Norman") == "gym_leader_norman"
        assert _canon_role("unknown_thing") == "unknown_thing"

    def test_canon_name(self):
        assert _canon_name("  Nurse  Joy  ") == "nurse joy"
        assert _canon_name("MAY") == "may"


# ===========================================================================
# Cold-Start Heuristic Tests
# ===========================================================================

class TestNpcRegistryColdStart:
    """Tests for ``infer_nearest_npc`` cold-start fallback."""

    NPCS = [
        {"slot": 0, "graphics_id": 0, "local_id": 0,
         "current_x": 5, "current_y": 3,
         "is_player": True, "invisible": False, "off_screen": False},
        {"slot": 1, "graphics_id": 105, "local_id": 2,
         "current_x": 10, "current_y": 3,
         "is_player": False, "invisible": False, "off_screen": False},
        {"slot": 2, "graphics_id": 50, "local_id": 3,
         "current_x": 20, "current_y": 20,
         "is_player": False, "invisible": False, "off_screen": False},
    ]

    def test_finds_nearest_npc(self):
        result = NpcRegistry.infer_nearest_npc(self.NPCS, 5, 3)
        assert result is not None
        assert result["graphics_id"] == 105  # closest to player at (5,3)

    def test_skips_player(self):
        result = NpcRegistry.infer_nearest_npc(self.NPCS, 5, 3)
        assert result is not None
        assert not result.get("is_player")

    def test_skips_invisible(self):
        npcs = [
            {"slot": 1, "graphics_id": 50, "local_id": 1,
             "current_x": 6, "current_y": 3,
             "is_player": False, "invisible": True, "off_screen": False},
        ]
        assert NpcRegistry.infer_nearest_npc(npcs, 5, 3) is None

    def test_respects_max_distance(self):
        result = NpcRegistry.infer_nearest_npc(self.NPCS, 5, 3, max_distance=3)
        # NPC at (10,3) is 5 tiles away — exceeds max_distance=3
        assert result is None

    def test_empty_list_returns_none(self):
        assert NpcRegistry.infer_nearest_npc([], 5, 3) is None


# ===========================================================================
# Integration: Registry + _resolve_npc_coords
# ===========================================================================

class TestRegistryResolveIntegration:
    """Test that _resolve_npc_coords uses the registry when available."""

    SAMPLE_NPCS = [
        {"slot": 0, "graphics_id": 0, "local_id": 0,
         "current_x": 5, "current_y": 3,
         "is_player": True, "invisible": False, "off_screen": False},
        {"slot": 1, "graphics_id": 105, "local_id": 2,
         "current_x": 10, "current_y": 3,
         "is_player": False, "invisible": False, "off_screen": False},
        {"slot": 2, "graphics_id": 29, "local_id": 1,
         "current_x": 7, "current_y": 3,
         "is_player": False, "invisible": False, "off_screen": False},
    ]

    def _state(self, location="ROUTE_103", npcs=None):
        return {
            "active_npcs": npcs if npcs is not None else self.SAMPLE_NPCS,
            "player": {"position": {"x": 5, "y": 3}, "location": location},
        }

    @pytest.fixture
    def registry(self, tmp_path):
        reg = NpcRegistry(json_path=str(tmp_path / "test_reg.json"))
        reg.register_npc("ROUTE_103", 2, name="May", role="rival", graphics_id=105)
        reg.register_npc("OLDALE_POKECENTER", 1, name="Nurse Joy", role="nurse", graphics_id=29)
        return reg

    @pytest.fixture
    def om(self, registry):
        return ObjectiveManager(npc_registry=registry)

    def test_resolve_by_role_uses_registry(self, om):
        """npc_role='rival' should hit the registry and resolve to (10, 3)."""
        coords = om._resolve_npc_coords(
            self._state(), npc_role="rival"
        )
        assert coords == (10, 3)

    def test_resolve_by_role_with_alias(self, om):
        """'may' is an alias for 'rival' — should resolve the same."""
        coords = om._resolve_npc_coords(
            self._state(), npc_role="may"
        )
        assert coords == (10, 3)

    def test_resolve_by_role_registry_miss_uses_explicit(self, om):
        """Role not in registry → fall back to explicit graphics_id."""
        coords = om._resolve_npc_coords(
            self._state(), npc_role="unknown_npc", graphics_id=105, fallback=(0, 0)
        )
        assert coords == (10, 3)  # Found via explicit graphics_id

    def test_resolve_by_role_no_registry(self):
        """ObjectiveManager without registry → uses explicit criteria directly."""
        om = ObjectiveManager()  # no registry
        coords = om._resolve_npc_coords(
            self._state(), npc_role="rival", graphics_id=105
        )
        assert coords == (10, 3)

    def test_cold_start_nearest_npc(self, tmp_path):
        """With empty registry and no explicit criteria, use nearest NPC."""
        empty_reg = NpcRegistry(json_path=str(tmp_path / "empty.json"))
        om = ObjectiveManager(npc_registry=empty_reg)
        coords = om._resolve_npc_coords(
            self._state(), npc_role="rival",
        )
        # No registry match, no graphics_id → cold start → nearest NPC
        assert coords is not None
        # Nearest non-player NPC is at (7, 3) — nurse, dist=2
        assert coords == (7, 3)

    def test_hardcoded_plus_registry(self, om):
        """When both npc_role and graphics_id are given, registry should win."""
        # Registry says rival has gfx=105, even if caller passes gfx=999
        # Registry overrides the explicit value
        coords = om._resolve_npc_coords(
            self._state(), npc_role="rival", graphics_id=999, fallback=(0, 0)
        )
        assert coords == (10, 3)  # Registry overrode gfx=999 with gfx=105

    def test_nurse_role_different_location(self, om):
        """Nurse registered for OLDALE_POKECENTER — should still match if on different map."""
        # State says we're on ROUTE_103, but nurse is registered for OLDALE
        coords = om._resolve_npc_coords(
            self._state(location="ROUTE_103"),
            npc_role="nurse",
        )
        # Registry has nurse with gfx=29, which is in active_npcs at (7, 3)
        assert coords == (7, 3)
