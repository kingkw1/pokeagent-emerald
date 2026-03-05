"""
Tests for Phase 4: Walkthrough RAG — Proactive Strategic Planner

Covers:
- LocationResolver  (alias resolution, fuzzy matching, edge cases)
- WalkthroughDB     (chunking, wikitext cleaning, query/retrieval)
- StrategicPlanner  (LLM response parsing, directive generation, shadow comparison)
- build_walkthrough_db  (offline build, chunk counts)
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
from agent.brain.location_resolver import (
    LOCATION_ALIASES,
    get_display_name,
    list_known_locations,
    resolve_location,
    resolve_location_key,
)
from agent.brain.walkthrough_db import (
    WalkthroughDB,
    chunk_wikitext,
    clean_wikitext,
)
from agent.brain.strategic_planner import StrategicPlanner


# ===========================================================================
# LocationResolver Tests
# ===========================================================================

class TestResolveLocation(unittest.TestCase):
    """Test location name resolution (Phase 4.4)."""

    def test_exact_alias_match(self):
        result = resolve_location("Route 101")
        self.assertIsNotNone(result)
        self.assertEqual(result["key"], "ROUTE_101")

    def test_case_insensitive(self):
        result = resolve_location("route 101")
        self.assertIsNotNone(result)
        self.assertEqual(result["key"], "ROUTE_101")

    def test_direct_graph_key(self):
        result = resolve_location("LITTLEROOT_TOWN")
        self.assertIsNotNone(result)
        self.assertEqual(result["key"], "LITTLEROOT_TOWN")

    def test_fuzzy_match(self):
        # "Petalburg" is close to "Petalburg City"
        result = resolve_location("Petalburg")
        self.assertIsNotNone(result)
        self.assertIn("PETALBURG", result["key"])

    def test_returns_portals(self):
        result = resolve_location("Oldale Town")
        self.assertIsNotNone(result)
        self.assertIn("portals", result)
        self.assertIn("ROUTE_101", result["portals"])

    def test_none_input(self):
        self.assertIsNone(resolve_location(None))

    def test_empty_string(self):
        self.assertIsNone(resolve_location(""))

    def test_nonexistent_location(self):
        self.assertIsNone(resolve_location("Hogwarts Castle"))

    def test_future_location_not_in_graph(self):
        # "Dewford Town" is in aliases but not in LOCATION_GRAPH
        result = resolve_location("Dewford Town")
        self.assertIsNone(result)  # key exists in aliases but not in graph


class TestResolveLocationKey(unittest.TestCase):
    """Test the convenience key-only resolver."""

    def test_returns_key_string(self):
        self.assertEqual(resolve_location_key("Petalburg City"), "PETALBURG_CITY")

    def test_returns_none_on_miss(self):
        self.assertIsNone(resolve_location_key("Nowhere Town"))


class TestGetDisplayName(unittest.TestCase):
    """Test reverse lookup: graph key → display name."""

    def test_known_key(self):
        name = get_display_name("ROUTE_101")
        self.assertEqual(name, "Route 101")

    def test_unknown_key_falls_back(self):
        name = get_display_name("UNKNOWN_ZONE_99")
        self.assertEqual(name, "UNKNOWN_ZONE_99")


class TestListKnownLocations(unittest.TestCase):
    """Test listing resolvable locations."""

    def test_returns_sorted_list(self):
        locations = list_known_locations()
        self.assertIsInstance(locations, list)
        self.assertTrue(len(locations) > 0)
        # Should be sorted
        self.assertEqual(locations, sorted(locations))

    def test_only_includes_graph_locations(self):
        from agent.location_graph import LOCATION_GRAPH
        locations = list_known_locations()
        for loc_name in locations:
            key = LOCATION_ALIASES[loc_name]
            self.assertIn(key, LOCATION_GRAPH,
                          f"Alias '{loc_name}' → '{key}' not in LOCATION_GRAPH")


# ===========================================================================
# WalkthroughDB Tests
# ===========================================================================

class TestCleanWikitext(unittest.TestCase):
    """Test MediaWiki markup stripping."""

    def test_removes_templates(self):
        raw = "Hello {{template|arg}} world"
        self.assertEqual(clean_wikitext(raw), "Hello  world")

    def test_removes_file_links(self):
        raw = "Text [[File:Example.png|thumb|Caption]] more"
        self.assertEqual(clean_wikitext(raw), "Text  more")

    def test_resolves_wiki_links(self):
        raw = "Visit [[Rustboro City|the city]] now"
        self.assertEqual(clean_wikitext(raw), "Visit the city now")

    def test_resolves_simple_wiki_links(self):
        raw = "Go to [[Route 101]] next"
        self.assertEqual(clean_wikitext(raw), "Go to Route 101 next")

    def test_removes_bold_italic(self):
        raw = "'''Bold''' and ''italic'' text"
        self.assertEqual(clean_wikitext(raw), "Bold and italic text")

    def test_removes_html_tags(self):
        raw = "Text <br/> more <ref>citation</ref>"
        self.assertNotIn("<", clean_wikitext(raw))

    def test_removes_categories(self):
        raw = "Text [[Category:Walkthroughs]] end"
        self.assertEqual(clean_wikitext(raw), "Text  end")

    def test_handles_empty_input(self):
        self.assertEqual(clean_wikitext(""), "")


class TestChunkWikitext(unittest.TestCase):
    """Test walkthrough text chunking."""

    def test_basic_chunking(self):
        raw = """
==Littleroot Town==
Start here. Begin the game.

==Route 101==
Walk north through tall grass to reach Oldale Town.
"""
        chunks = chunk_wikitext(raw, part_number=1)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["metadata"]["location"], "Littleroot Town")
        self.assertEqual(chunks[1]["metadata"]["location"], "Route 101")

    def test_part_number_in_metadata(self):
        raw = "==Test Location==\nSome walkthrough text here and more."
        chunks = chunk_wikitext(raw, part_number=7)
        self.assertEqual(chunks[0]["metadata"]["part"], 7)

    def test_section_order(self):
        raw = """
==First==
Text first section here content.

==Second==
Text second section here content.

==Third==
Text third section here content.
"""
        chunks = chunk_wikitext(raw, part_number=1)
        orders = [c["metadata"]["section_order"] for c in chunks]
        self.assertEqual(orders, sorted(orders))

    def test_battle_detection(self):
        raw = "==Route 102==\nThere is a trainer battle here. Defeat the trainer."
        chunks = chunk_wikitext(raw, part_number=1)
        self.assertTrue(chunks[0]["metadata"]["has_battle"])

    def test_no_battle_detection(self):
        raw = "==Oldale Town==\nHeal your Pokemon at the center and buy items."
        chunks = chunk_wikitext(raw, part_number=1)
        self.assertFalse(chunks[0]["metadata"]["has_battle"])

    def test_skips_short_sections(self):
        raw = "==Short==\nToo short."
        chunks = chunk_wikitext(raw, part_number=1)
        self.assertEqual(len(chunks), 0)  # body < 20 chars


class TestWalkthroughDB(unittest.TestCase):
    """Integration tests for WalkthroughDB (uses tmp dir)."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db = WalkthroughDB(db_path=self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_add_and_count(self):
        self.db.add_chunk("Walk north to Oldale Town.", {"location": "Route 101", "part": 1})
        self.assertEqual(self.db.count(), 1)

    def test_add_chunks_batch(self):
        chunks = [
            {"text": "Start in Littleroot Town. Your adventure begins here.", "metadata": {"location": "Littleroot Town", "part": 1}},
            {"text": "Walk north through Route 101 tall grass to reach Oldale.", "metadata": {"location": "Route 101", "part": 1}},
        ]
        added = self.db.add_chunks(chunks)
        self.assertEqual(added, 2)
        self.assertEqual(self.db.count(), 2)

    def test_query_returns_results(self):
        self.db.add_chunk(
            "Walk north through Route 101 tall grass to reach Oldale Town.",
            {"location": "Route 101", "part": 1, "section_order": 1},
        )
        self.db.add_chunk(
            "Enter Rustboro Gym and battle Roxanne the rock type gym leader.",
            {"location": "Rustboro City", "part": 3, "section_order": 5},
        )
        results = self.db.query("I am on Route 101, what do I do?")
        self.assertTrue(len(results) > 0)
        # Route 101 chunk should rank higher
        self.assertIn("Route 101", results[0]["text"])

    def test_query_empty_db(self):
        results = self.db.query("anything at all")
        self.assertEqual(results, [])

    def test_clear(self):
        self.db.add_chunk("Something here.", {"location": "Test"})
        self.assertEqual(self.db.count(), 1)
        self.db.clear()
        self.assertEqual(self.db.count(), 0)

    def test_query_next_steps(self):
        self.db.add_chunk(
            "In Petalburg City, visit the gym to meet your father Norman. He will explain gym battles.",
            {"location": "Petalburg City", "part": 2, "section_order": 3},
        )
        results = self.db.query_next_steps("Petalburg City")
        self.assertTrue(len(results) > 0)

    def test_peek(self):
        self.db.add_chunk("Peek test chunk entry.", {"location": "TestLoc"})
        peeked = self.db.peek(n=1)
        self.assertEqual(len(peeked), 1)
        self.assertEqual(peeked[0]["text"], "Peek test chunk entry.")


# ===========================================================================
# StrategicPlanner Tests
# ===========================================================================

class TestStrategicPlannerParsing(unittest.TestCase):
    """Test LLM response parsing in isolation."""

    def test_parse_valid_json(self):
        response = json.dumps({
            "target_location": "Route 102",
            "description": "Travel west to Petalburg City.",
            "priority_actions": ["Battle trainers", "Catch a Ralts"],
        })
        plan = StrategicPlanner._parse_response(response)
        self.assertEqual(plan["target_location"], "Route 102")
        self.assertEqual(len(plan["priority_actions"]), 2)

    def test_parse_markdown_fenced_json(self):
        response = '```json\n{"target_location": "Oldale Town", "description": "Heal.", "priority_actions": ["Heal"]}\n```'
        plan = StrategicPlanner._parse_response(response)
        self.assertEqual(plan["target_location"], "Oldale Town")

    def test_parse_empty_response(self):
        plan = StrategicPlanner._parse_response("")
        self.assertIsNone(plan["target_location"])
        self.assertIsInstance(plan["priority_actions"], list)

    def test_parse_garbage_response(self):
        plan = StrategicPlanner._parse_response("I don't know what to do!")
        self.assertIsNone(plan["target_location"])

    def test_parse_missing_keys(self):
        response = json.dumps({"target_location": "Route 101"})
        plan = StrategicPlanner._parse_response(response)
        self.assertEqual(plan["target_location"], "Route 101")
        self.assertIn("description", plan)
        self.assertIn("priority_actions", plan)


class TestStrategicPlannerDirective(unittest.TestCase):
    """Test end-to-end directive generation (no real VLM)."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db = WalkthroughDB(db_path=self.tmp_dir)
        # Seed with walkthrough data — chunks must be semantically close
        # to their query locations so they pass the distance threshold.
        # The enriched RAG query is: "I am in <location> with N badges.
        # Last completed milestone: X. What should I do next?"
        # Chunks need enough context for the embedding model to score
        # them below the 0.50 distance threshold.
        self.db.add_chunk(
            "Littleroot Town\n\n"
            "Littleroot Town is the starting town in Pokemon Emerald. "
            "You begin your adventure here with 0 badges. "
            "After choosing your starter Pokemon from Professor Birch, "
            "you should head north from Littleroot Town through Route 101 "
            "to reach Oldale Town. This is the first thing you should do "
            "next after starting the game in Littleroot Town.",
            {"location": "Littleroot Town", "part": 1, "section_order": 1},
        )
        self.db.add_chunk(
            "Route 102\n\n"
            "When you are in Route 102 with 0 badges, you should head west "
            "to reach Petalburg City. What should you do next in Route 102? "
            "Go west. Battle trainers along the way through Route 102 "
            "to gain experience. After arriving in Petalburg City, visit the "
            "gym to meet your father Norman. Route 102 is the path from "
            "Oldale Town to Petalburg City.",
            {"location": "Route 102", "part": 1, "section_order": 2},
        )
        # No VLM → mock fallback
        self.planner = StrategicPlanner(vlm=None, walkthrough_db=self.db)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_returns_directive_dict(self):
        result = self.planner.get_next_directive(
            current_location="LITTLEROOT_TOWN",
            badge_count=0,
            pokemon_summary="Treecko Lv.5",
            last_milestone="STARTER_CHOSEN",
        )
        self.assertIn("target_location", result)
        self.assertIn("description", result)
        self.assertIn("source", result)
        self.assertEqual(result["source"], "walkthrough_rag")

    def test_mock_vlm_returns_route_101(self):
        """The mock VLM hardcodes Route 101 as the target."""
        result = self.planner.get_next_directive(
            current_location="LITTLEROOT_TOWN",
        )
        self.assertEqual(result["target_location"], "ROUTE_101")

    def test_goal_coords_present(self):
        """When target resolves, goal_coords should be populated."""
        result = self.planner.get_next_directive(
            current_location="LITTLEROOT_TOWN",
        )
        self.assertIn("goal_coords", result)
        self.assertIsNotNone(result["goal_coords"])
        self.assertEqual(len(result["goal_coords"]), 3)  # (x, y, 'LOCATION')

    def test_with_real_llm_mock(self):
        """Test with a mocked VLM that returns structured JSON."""
        mock_vlm = MagicMock()
        mock_vlm.get_text_query.return_value = json.dumps({
            "target_location": "Petalburg City",
            "description": "Head to Petalburg to meet Dad at the gym.",
            "priority_actions": ["Visit Petalburg Gym"],
        })
        planner = StrategicPlanner(vlm=mock_vlm, walkthrough_db=self.db)
        result = planner.get_next_directive(current_location="ROUTE_102")
        self.assertEqual(result["target_location"], "PETALBURG_CITY")
        self.assertIn("goal_coords", result)

    def test_empty_context_defers_to_milestone(self):
        """When all chunks are filtered out, planner returns None target (milestone fallback)."""
        # Query a location with no relevant chunks in the DB
        result = self.planner.get_next_directive(
            current_location="RUSTBORO_CITY",
            badge_count=0,
            last_milestone="STARTER_CHOSEN",
        )
        # No LLM call should happen — target_location should be None
        self.assertIsNone(result["target_location"])
        self.assertEqual(result["source"], "walkthrough_rag")


class TestShadowCompare(unittest.TestCase):
    """Test shadow-mode comparison (Phase 4.3a)."""

    def setUp(self):
        self.planner = StrategicPlanner()

    def test_agree_same_target(self):
        result = self.planner.shadow_compare("ROUTE_101", "ROUTE_101")
        self.assertTrue(result["agree"])

    def test_disagree_different_target(self):
        result = self.planner.shadow_compare("ROUTE_101", "ROUTE_102")
        self.assertFalse(result["agree"])

    def test_both_none(self):
        result = self.planner.shadow_compare(None, None)
        self.assertTrue(result["agree"])

    def test_one_none(self):
        result = self.planner.shadow_compare("ROUTE_101", None)
        self.assertFalse(result["agree"])

    def test_case_insensitive(self):
        result = self.planner.shadow_compare("route_101", "ROUTE_101")
        self.assertTrue(result["agree"])


# ===========================================================================
# Build Script Tests
# ===========================================================================

class TestBuildWalkthroughDB(unittest.TestCase):
    """Test the build script in offline mode."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_offline_build_produces_chunks(self):
        from scripts.build_walkthrough_db import build_database
        total = build_database(
            parts=[1, 2, 3],
            offline=True,
            rebuild=True,
            db_path=self.tmp_dir,
        )
        self.assertGreater(total, 0)
        # Verify DB has entries
        db = WalkthroughDB(db_path=self.tmp_dir)
        self.assertGreater(db.count(), 0)

    def test_dry_run_does_not_embed(self):
        from scripts.build_walkthrough_db import build_database
        total = build_database(
            parts=[1],
            offline=True,
            dry_run=True,
            db_path=self.tmp_dir,
        )
        self.assertGreater(total, 0)
        # DB should remain empty in dry-run
        db = WalkthroughDB(db_path=self.tmp_dir)
        self.assertEqual(db.count(), 0)


if __name__ == "__main__":
    unittest.main()
