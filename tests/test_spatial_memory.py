# tests/test_spatial_memory.py
"""Phase 3 — Spatial Awareness: unit tests for coordinate-tagged episodic memory."""
import unittest
import tempfile
import shutil

from agent.brain.memory import EpisodicMemory, extract_spatial_metadata


# ---------------------------------------------------------------------------
# Helper: fake state_data dicts (mirrors the shape of real emulator output)
# ---------------------------------------------------------------------------
def _make_state(x=5, y=12, location="ROUTE_102"):
    return {
        "player": {
            "position": {"x": x, "y": y},
            "location": location,
        },
        "game": {"in_battle": False},
    }


class TestExtractSpatialMetadata(unittest.TestCase):
    """Pure-function tests for extract_spatial_metadata()."""

    def test_normal_state(self):
        meta = extract_spatial_metadata(_make_state(5, 12, "ROUTE_102"))
        self.assertEqual(meta, {"pos_x": 5, "pos_y": 12, "location": "ROUTE_102"})

    def test_zero_coords_are_valid(self):
        meta = extract_spatial_metadata(_make_state(0, 0, "LITTLEROOT_TOWN"))
        self.assertEqual(meta["pos_x"], 0)
        self.assertEqual(meta["pos_y"], 0)

    def test_missing_position_key(self):
        meta = extract_spatial_metadata({"player": {"location": "OLDALE_TOWN"}})
        self.assertNotIn("pos_x", meta)
        self.assertNotIn("pos_y", meta)
        self.assertEqual(meta["location"], "OLDALE_TOWN")

    def test_none_state_data(self):
        meta = extract_spatial_metadata(None)
        self.assertEqual(meta, {})

    def test_empty_state_data(self):
        meta = extract_spatial_metadata({})
        self.assertEqual(meta, {})

    def test_tuple_position(self):
        """Some internal callers might pass position as a tuple."""
        state = {"player": {"position": (7, 3), "location": "PETALBURG_CITY"}}
        meta = extract_spatial_metadata(state)
        self.assertEqual(meta["pos_x"], 7)
        self.assertEqual(meta["pos_y"], 3)

    def test_missing_location(self):
        state = {"player": {"position": {"x": 1, "y": 2}}}
        meta = extract_spatial_metadata(state)
        self.assertIn("pos_x", meta)
        self.assertNotIn("location", meta)


class TestLogEventSpatial(unittest.TestCase):
    """Verify that log_event() with state_data persists spatial metadata."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.mem = EpisodicMemory(db_path=self._tmpdir)

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_log_event_stores_coords(self):
        self.mem.log_event(
            "Heard dialogue: 'Wait!'",
            {"type": "dialogue"},
            state_data=_make_state(5, 12, "ROUTE_102"),
        )
        results = self.mem.retrieve_raw("Wait", n_results=1)
        self.assertEqual(len(results), 1)
        meta = results[0]["metadata"]
        self.assertEqual(meta["pos_x"], 5)
        self.assertEqual(meta["pos_y"], 12)
        self.assertEqual(meta["location"], "ROUTE_102")
        self.assertEqual(meta["type"], "dialogue")

    def test_log_event_without_state_data(self):
        """Backwards-compatible: omitting state_data must still work."""
        self.mem.log_event("Some event", {"type": "mechanic"})
        results = self.mem.retrieve_raw("event", n_results=1)
        self.assertEqual(len(results), 1)
        meta = results[0]["metadata"]
        self.assertNotIn("pos_x", meta)
        self.assertEqual(meta["type"], "mechanic")

    def test_caller_metadata_wins(self):
        """If the caller already set 'location', state_data must not overwrite."""
        self.mem.log_event(
            "Battle started",
            {"type": "battle_start", "location": "CALLER_SET"},
            state_data=_make_state(1, 2, "STATE_DATA_LOCATION"),
        )
        results = self.mem.retrieve_raw("Battle", n_results=1)
        meta = results[0]["metadata"]
        # Caller's value wins (setdefault semantics)
        self.assertEqual(meta["location"], "CALLER_SET")
        # But spatial coords still come through (no conflict)
        self.assertEqual(meta["pos_x"], 1)


class TestRetrieveNear(unittest.TestCase):
    """Test spatial filtering via retrieve_near()."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.mem = EpisodicMemory(db_path=self._tmpdir)
        # Seed several events at known locations
        self.mem.log_event(
            "Trainer battle on Route 102",
            {"type": "battle_start"},
            state_data=_make_state(10, 20, "ROUTE_102"),
        )
        self.mem.log_event(
            "NPC said route is closed",
            {"type": "dialogue"},
            state_data=_make_state(50, 60, "ROUTE_104"),
        )
        self.mem.log_event(
            "Healed at Pokemon Center",
            {"type": "heal"},
            state_data=_make_state(12, 22, "ROUTE_102"),
        )
        # Event without spatial data
        self.mem.log_event(
            "Cut trees block the path",
            {"type": "mechanic"},
        )

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_filter_by_location(self):
        entries = self.mem.retrieve_near(
            "battle", location="ROUTE_102", n_results=10,
        )
        # All returned entries should be from ROUTE_102 or have no location
        for e in entries:
            loc = e["metadata"].get("location")
            if loc is not None:
                self.assertEqual(loc, "ROUTE_102")

    def test_filter_by_radius(self):
        entries = self.mem.retrieve_near(
            "battle",
            x=10, y=20, radius=5,
            n_results=10,
        )
        # "Trainer battle on Route 102" (10,20) should be in
        # "Healed at Pokemon Center" (12,22) should be in (within radius 5)
        # "NPC said route is closed" (50,60) should NOT be in
        texts = [e["text"] for e in entries]
        self.assertIn("Trainer battle on Route 102", texts)
        self.assertIn("Healed at Pokemon Center", texts)
        self.assertNotIn("NPC said route is closed", texts)
        # Events without coords are kept
        self.assertIn("Cut trees block the path", texts)

    def test_filter_by_location_and_radius(self):
        entries = self.mem.retrieve_near(
            "path",
            location="ROUTE_104",
            x=50, y=60, radius=3,
            n_results=10,
        )
        # Only ROUTE_104 events, and only within radius
        for e in entries:
            loc = e["metadata"].get("location")
            if loc is not None:
                self.assertEqual(loc, "ROUTE_104")

    def test_empty_db(self):
        empty_mem = EpisodicMemory(db_path=tempfile.mkdtemp())
        entries = empty_mem.retrieve_near("anything", location="ROUTE_102")
        self.assertEqual(entries, [])


if __name__ == "__main__":
    unittest.main()
