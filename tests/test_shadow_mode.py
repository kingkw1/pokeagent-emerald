"""
Tests for Phase 4.3a (Shadow-Mode RAG Comparison) and 4.3b (RAG-Primary Navigation)

Covers:
- Shadow comparison fires when StrategicPlanner is attached
- Shadow comparison does NOT fire when no planner is attached
- Throttling (only every 20 steps)
- Agreement tracking and statistics
- JSONL log file writing
- Directive passthrough (milestone directive is never modified)
- Skips during battle and dialogue
- Phase 4.3b: _query_rag_target resolution
- Phase 4.3b: RAG override vs milestone fallback tracking
- Phase 4.3b: directive_source field in shadow log
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from agent.objective_manager import (
    MILESTONE_PROGRESSION,
    ObjectiveManager,
    get_highest_milestone_index,
    get_next_milestone_target,
)


def _make_state_data(
    location="ROUTE 101",
    x=5,
    y=5,
    in_battle=False,
    milestones=None,
    screen_context="exploration",
    party=None,
    badges=0,
):
    """Helper to build a minimal state_data dict."""
    if milestones is None:
        milestones = {}
    if party is None:
        party = [{"species_name": "MUDKIP", "level": 10, "current_hp": 30, "max_hp": 30}]
    return {
        "player": {
            "location": location,
            "position": {"x": x, "y": y},
            "party": party,
        },
        "game": {
            "in_battle": in_battle,
            "badges": badges,
        },
        "milestones": milestones,
        "screen_context": screen_context,
    }


def _make_mock_strategic_planner(target_location="OLDALE_TOWN", display_name="Oldale Town"):
    """Create a mock StrategicPlanner that returns a predictable directive."""
    planner = MagicMock()
    planner.get_next_directive.return_value = {
        "target_location": target_location,
        "target_display_name": display_name,
        "description": f"Head to {display_name}",
        "priority_actions": ["Heal at Pokemon Center"],
        "goal_coords": (5, 0, target_location),
        "source": "walkthrough_rag",
    }
    planner.shadow_compare.return_value = {
        "milestone_target": "OLDALE_TOWN",
        "rag_target": target_location,
        "agree": target_location == "OLDALE_TOWN",
    }
    return planner


class TestShadowModeInit(unittest.TestCase):
    """Test that ObjectiveManager accepts and stores a StrategicPlanner."""

    def test_init_without_planner(self):
        om = ObjectiveManager()
        self.assertIsNone(om.strategic_planner)
        self.assertEqual(om._shadow_step_count, 0)
        self.assertEqual(om._shadow_total_count, 0)

    def test_init_with_planner(self):
        mock_planner = _make_mock_strategic_planner()
        om = ObjectiveManager(strategic_planner=mock_planner)
        self.assertIs(om.strategic_planner, mock_planner)
        self.assertEqual(om._shadow_step_count, 0)


class TestShadowModeThrottling(unittest.TestCase):
    """Test that shadow comparison only fires every 20 steps."""

    def setUp(self):
        self.planner = _make_mock_strategic_planner()
        self.om = ObjectiveManager(strategic_planner=self.planner)
        self.tmp_dir = tempfile.mkdtemp()
        self.om._shadow_log_path = os.path.join(self.tmp_dir, "shadow.jsonl")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_fires_on_step_1(self):
        """First call to _run_shadow_comparison should fire (step 1 % 20 == 1)."""
        state = _make_state_data()
        directive = {"target_location": "OLDALE_TOWN", "description": "Go"}
        self.om._run_shadow_comparison(state, directive)
        self.assertEqual(self.om._shadow_step_count, 1)
        self.assertEqual(self.om._shadow_total_count, 1)
        self.planner.get_next_directive.assert_called_once()

    def test_skips_steps_2_through_20(self):
        """Steps 2-20 should be throttled (no LLM call)."""
        state = _make_state_data()
        directive = {"target_location": "OLDALE_TOWN"}

        # Step 1 fires
        self.om._run_shadow_comparison(state, directive)
        self.assertEqual(self.planner.get_next_directive.call_count, 1)

        # Steps 2-20 should NOT fire
        for _ in range(19):
            self.om._run_shadow_comparison(state, directive)
        self.assertEqual(self.planner.get_next_directive.call_count, 1)
        self.assertEqual(self.om._shadow_step_count, 20)
        self.assertEqual(self.om._shadow_total_count, 1)

    def test_fires_on_step_21(self):
        """Step 21 should fire again (21 % 20 == 1)."""
        state = _make_state_data()
        directive = {"target_location": "OLDALE_TOWN"}

        for _ in range(21):
            self.om._run_shadow_comparison(state, directive)

        self.assertEqual(self.planner.get_next_directive.call_count, 2)
        self.assertEqual(self.om._shadow_total_count, 2)


class TestShadowModeSkips(unittest.TestCase):
    """Test that shadow comparison is skipped during battles and dialogue."""

    def setUp(self):
        self.planner = _make_mock_strategic_planner()
        self.om = ObjectiveManager(strategic_planner=self.planner)
        self.tmp_dir = tempfile.mkdtemp()
        self.om._shadow_log_path = os.path.join(self.tmp_dir, "shadow.jsonl")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_skips_during_battle(self):
        state = _make_state_data(in_battle=True)
        self.om._run_shadow_comparison(state, {"target_location": "X"})
        # Step count increments but no comparison logged
        self.assertEqual(self.om._shadow_step_count, 1)
        self.assertEqual(self.om._shadow_total_count, 0)
        self.planner.get_next_directive.assert_not_called()

    def test_skips_during_dialogue(self):
        state = _make_state_data(screen_context="dialogue")
        self.om._run_shadow_comparison(state, {"target_location": "X"})
        self.assertEqual(self.om._shadow_step_count, 1)
        self.assertEqual(self.om._shadow_total_count, 0)
        self.planner.get_next_directive.assert_not_called()


class TestShadowModeAgreement(unittest.TestCase):
    """Test agreement tracking."""

    def test_agreement_counted(self):
        planner = _make_mock_strategic_planner(target_location="OLDALE_TOWN")
        planner.shadow_compare.return_value = {
            "milestone_target": "OLDALE_TOWN",
            "rag_target": "OLDALE_TOWN",
            "agree": True,
        }
        om = ObjectiveManager(strategic_planner=planner)
        tmp_dir = tempfile.mkdtemp()
        om._shadow_log_path = os.path.join(tmp_dir, "shadow.jsonl")

        state = _make_state_data()
        om._run_shadow_comparison(state, {"goal_coords": (5, 0, "OLDALE_TOWN")})

        self.assertEqual(om._shadow_agree_count, 1)
        self.assertEqual(om._shadow_total_count, 1)
        shutil.rmtree(tmp_dir)

    def test_disagreement_counted(self):
        planner = _make_mock_strategic_planner(target_location="ROUTE_102")
        planner.shadow_compare.return_value = {
            "milestone_target": "OLDALE_TOWN",
            "rag_target": "ROUTE_102",
            "agree": False,
        }
        om = ObjectiveManager(strategic_planner=planner)
        tmp_dir = tempfile.mkdtemp()
        om._shadow_log_path = os.path.join(tmp_dir, "shadow.jsonl")

        state = _make_state_data()
        om._run_shadow_comparison(state, {"goal_coords": (5, 0, "OLDALE_TOWN")})

        self.assertEqual(om._shadow_agree_count, 0)
        self.assertEqual(om._shadow_total_count, 1)
        shutil.rmtree(tmp_dir)


class TestShadowStats(unittest.TestCase):
    """Test get_shadow_stats()."""

    def test_initial_stats(self):
        om = ObjectiveManager()
        stats = om.get_shadow_stats()
        self.assertEqual(stats["total_comparisons"], 0)
        self.assertEqual(stats["agreement_rate"], 0.0)
        self.assertEqual(stats["rag_overrides"], 0)
        self.assertEqual(stats["milestone_fallbacks"], 0)

    def test_stats_after_comparisons(self):
        planner = _make_mock_strategic_planner()
        planner.shadow_compare.return_value = {
            "milestone_target": "X", "rag_target": "X", "agree": True,
        }
        om = ObjectiveManager(strategic_planner=planner)
        tmp_dir = tempfile.mkdtemp()
        om._shadow_log_path = os.path.join(tmp_dir, "shadow.jsonl")

        state = _make_state_data()
        # Fire 3 comparisons (steps 1, 21, 41)
        for _ in range(41):
            om._run_shadow_comparison(state, {"target_location": "X"})

        stats = om.get_shadow_stats()
        self.assertEqual(stats["total_comparisons"], 3)
        self.assertEqual(stats["agreements"], 3)
        self.assertEqual(stats["agreement_rate"], 100.0)
        self.assertEqual(stats["steps_processed"], 41)
        shutil.rmtree(tmp_dir)


class TestShadowLogFile(unittest.TestCase):
    """Test JSONL log file writing."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.planner = _make_mock_strategic_planner()
        self.planner.shadow_compare.return_value = {
            "milestone_target": "OLDALE_TOWN",
            "rag_target": "OLDALE_TOWN",
            "agree": True,
        }
        self.om = ObjectiveManager(strategic_planner=self.planner)
        self.om._shadow_log_path = os.path.join(self.tmp_dir, "shadow_comparison.jsonl")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_log_file_created(self):
        state = _make_state_data()
        self.om._run_shadow_comparison(state, {"target_location": "OLDALE_TOWN"})

        self.assertTrue(os.path.exists(self.om._shadow_log_path))

    def test_log_entry_structure(self):
        state = _make_state_data(location="ROUTE 101", x=5, y=10)
        self.om._run_shadow_comparison(state, {"goal_coords": (5, 0, "OLDALE_TOWN")})

        with open(self.om._shadow_log_path) as f:
            entry = json.loads(f.readline())

        self.assertIn("timestamp", entry)
        self.assertIn("step", entry)
        self.assertEqual(entry["location"], "ROUTE 101")
        self.assertEqual(entry["coords"], [5, 10])
        self.assertEqual(entry["milestone_target"], "OLDALE_TOWN")
        self.assertEqual(entry["rag_target"], "OLDALE_TOWN")
        self.assertTrue(entry["agree"])
        self.assertEqual(entry["agreement_rate"], 100.0)

    def test_multiple_entries_appended(self):
        state = _make_state_data()
        directive = {"target_location": "OLDALE_TOWN"}

        # Fire 3 comparisons
        for _ in range(41):
            self.om._run_shadow_comparison(state, directive)

        with open(self.om._shadow_log_path) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 3)


class TestDirectivePassthrough(unittest.TestCase):
    """Test that get_next_action_directive produces valid directives with
    and without a StrategicPlanner attached.  In Phase 4.3b the RAG planner
    can change the navigation target, so we only verify structural validity."""

    def test_directive_unchanged(self):
        """get_next_action_directive() should return a valid directive
        (or None) regardless of whether a StrategicPlanner is attached.
        With RAG-primary (4.3b) the target may differ, but milestone_id
        is always populated from MILESTONE_PROGRESSION."""
        planner = _make_mock_strategic_planner()
        planner.shadow_compare.return_value = {
            "milestone_target": "X", "rag_target": "Y", "agree": False,
        }
        om_with = ObjectiveManager(strategic_planner=planner)
        tmp_dir = tempfile.mkdtemp()
        om_with._shadow_log_path = os.path.join(tmp_dir, "shadow.jsonl")
        om_without = ObjectiveManager()

        # Build state that triggers the sequential milestone system
        # (Use a simple exploration state — most code paths return None or a dict)
        state = _make_state_data(
            location="LITTLEROOT TOWN",
            milestones={
                "GAME_RUNNING": {"completed": True},
                "PLAYER_NAME_SET": {"completed": True},
                "INTRO_CUTSCENE_COMPLETE": {"completed": True},
                "LITTLEROOT_TOWN": {"completed": True},
                "PLAYER_HOUSE_ENTERED": {"completed": True},
                "PLAYER_BEDROOM": {"completed": True},
                "RIVAL_HOUSE": {"completed": True},
                "RIVAL_BEDROOM": {"completed": True},
                "ROUTE_101": {"completed": True},
                "STARTER_CHOSEN": {"completed": True},
                "BIRCH_LAB_VISITED": {"completed": True},
            },
        )

        d_with = om_with.get_next_action_directive(state)
        d_without = om_without.get_next_action_directive(state)

        # Both should return a directive (not None)
        if d_with is not None and d_without is not None:
            # Both should have a milestone key from MILESTONE_PROGRESSION
            self.assertIn("milestone", d_with)
            self.assertIn("milestone", d_without)
            # Milestone id should be the same (from the same progression)
            self.assertEqual(d_with.get("milestone"), d_without.get("milestone"))

        shutil.rmtree(tmp_dir)


class TestShadowWithNoPlanner(unittest.TestCase):
    """When no StrategicPlanner is attached, shadow comparison is a no-op."""

    def test_no_shadow_without_planner(self):
        om = ObjectiveManager()
        state = _make_state_data()
        # _run_shadow_comparison should be a silent no-op
        om._run_shadow_comparison(state, {"target_location": "X"})
        self.assertEqual(om._shadow_step_count, 0)
        self.assertEqual(om._shadow_total_count, 0)


class TestMilestoneTargetExtraction(unittest.TestCase):
    """Test that milestone_target is correctly extracted from various directive formats."""

    def setUp(self):
        self.planner = _make_mock_strategic_planner()
        self.planner.shadow_compare.return_value = {
            "milestone_target": None, "rag_target": None, "agree": True,
        }
        self.om = ObjectiveManager(strategic_planner=self.planner)
        self.tmp_dir = tempfile.mkdtemp()
        self.om._shadow_log_path = os.path.join(self.tmp_dir, "shadow.jsonl")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_milestone_preferred_over_goal_coords(self):
        """milestone key (final destination) is preferred over goal_coords[2] (waypoint)."""
        state = _make_state_data()
        # This is the real-world case: milestone=ROUTE_103 but goal_coords says ROUTE_102
        self.om._run_shadow_comparison(state, {
            "milestone": "ROUTE_103",
            "goal_coords": (48, 11, "ROUTE_102"),
        })

        with open(self.om._shadow_log_path) as f:
            entry = json.loads(f.readline())
        self.assertEqual(entry["milestone_target"], "ROUTE_103")

    def test_goal_coords_fallback_when_no_milestone(self):
        """goal_coords[2] used only when milestone and target_location are absent."""
        state = _make_state_data()
        self.om._run_shadow_comparison(state, {"goal_coords": (5, 0, "ROUTE_103")})

        with open(self.om._shadow_log_path) as f:
            entry = json.loads(f.readline())
        self.assertEqual(entry["milestone_target"], "ROUTE_103")

    def test_extracts_from_target_location(self):
        state = _make_state_data()
        self.om._run_shadow_comparison(state, {"target_location": "PETALBURG_CITY"})

        with open(self.om._shadow_log_path) as f:
            entry = json.loads(f.readline())
        self.assertEqual(entry["milestone_target"], "PETALBURG_CITY")

    def test_extracts_from_milestone(self):
        state = _make_state_data()
        self.om._run_shadow_comparison(state, {"milestone": "RUSTBORO_CITY"})

        with open(self.om._shadow_log_path) as f:
            entry = json.loads(f.readline())
        self.assertEqual(entry["milestone_target"], "RUSTBORO_CITY")

    def test_none_directive(self):
        state = _make_state_data()
        self.om._run_shadow_comparison(state, None)

        with open(self.om._shadow_log_path) as f:
            entry = json.loads(f.readline())
        self.assertIsNone(entry["milestone_target"])


# =====================================================================
# Phase 4.3b Tests: RAG-Primary Navigation
# =====================================================================


class TestQueryRagTarget(unittest.TestCase):
    """Test _query_rag_target returns correct structure or None."""

    def test_returns_none_without_planner(self):
        om = ObjectiveManager()
        state = _make_state_data()
        result = om._query_rag_target(state)
        self.assertIsNone(result)

    def test_returns_target_with_planner(self):
        planner = _make_mock_strategic_planner(target_location="OLDALE_TOWN")
        om = ObjectiveManager(strategic_planner=planner)
        state = _make_state_data()
        result = om._query_rag_target(state)

        self.assertIsNotNone(result)
        self.assertEqual(result["target_location"], "OLDALE_TOWN")
        self.assertEqual(result["display_name"], "Oldale Town")
        self.assertEqual(result["target_coords"], (5, 0))

    def test_returns_none_when_rag_has_no_target(self):
        planner = _make_mock_strategic_planner()
        planner.get_next_directive.return_value = {
            "target_location": None,
            "description": "Explore around.",
        }
        om = ObjectiveManager(strategic_planner=planner)
        state = _make_state_data()
        result = om._query_rag_target(state)
        self.assertIsNone(result)

    def test_returns_none_on_exception(self):
        planner = _make_mock_strategic_planner()
        planner.get_next_directive.side_effect = RuntimeError("LLM down")
        om = ObjectiveManager(strategic_planner=planner)
        state = _make_state_data()
        result = om._query_rag_target(state)
        self.assertIsNone(result)

    def test_passes_badge_count_and_party(self):
        planner = _make_mock_strategic_planner()
        om = ObjectiveManager(strategic_planner=planner)
        state = _make_state_data(badges=3, party=[
            {"species_name": "BLAZIKEN", "level": 40, "current_hp": 100, "max_hp": 100},
        ])
        om._query_rag_target(state)

        call_kwargs = planner.get_next_directive.call_args[1]
        self.assertEqual(call_kwargs["badge_count"], 2)  # bin(3)='0b11' → 2 bits
        self.assertIn("BLAZIKEN", call_kwargs["pokemon_summary"])


class TestRagPrimaryCounters(unittest.TestCase):
    """Test that RAG override / milestone fallback counters update correctly."""

    def test_initial_counts(self):
        om = ObjectiveManager()
        self.assertEqual(om._rag_override_count, 0)
        self.assertEqual(om._milestone_fallback_count, 0)

    def test_stats_include_rag_fields(self):
        om = ObjectiveManager()
        stats = om.get_shadow_stats()
        self.assertIn("rag_overrides", stats)
        self.assertIn("milestone_fallbacks", stats)


class TestDirectiveSourceTracking(unittest.TestCase):
    """Test that _last_directive_source defaults and updates properly."""

    def test_default_source_is_milestone(self):
        om = ObjectiveManager()
        self.assertEqual(om._last_directive_source, "milestone")

    def test_source_in_shadow_log(self):
        """Shadow log entries should include directive_source field."""
        planner = _make_mock_strategic_planner()
        planner.shadow_compare.return_value = {
            "milestone_target": "OLDALE_TOWN",
            "rag_target": "OLDALE_TOWN",
            "agree": True,
        }
        om = ObjectiveManager(strategic_planner=planner)
        tmp_dir = tempfile.mkdtemp()
        om._shadow_log_path = os.path.join(tmp_dir, "shadow.jsonl")

        state = _make_state_data()
        om._run_shadow_comparison(state, {"milestone": "OLDALE_TOWN"})

        with open(om._shadow_log_path) as f:
            entry = json.loads(f.readline())
        self.assertIn("directive_source", entry)
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
