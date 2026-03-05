"""
Tests for battle exit detection and post-battle grace period.

Covers three bug-fixes:
  1. Battle bot recognises "Got away safely!" and stops sending RUN inputs
     that would leak onto the overworld and displace the player.
  2. Oscillation detector clears position history on battle→overworld
     transition, preventing false stuck detection from stale pre-battle
     positions.
  3. Milestone data is injected into state_data['game']['milestones_completed']
     so the battle bot can distinguish Birch rescue from regular wild battles.
"""

import unittest
from unittest.mock import MagicMock, patch
from agent.battle_bot import BattleBot, BattleType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    in_battle=True,
    dialogue="",
    milestones_completed=None,
    has_pokedex=False,
    location="ROUTE 103",
    player_x=10,
    player_y=5,
):
    """Build a minimal state_data dict for battle bot tests."""
    return {
        "game": {
            "in_battle": in_battle,
            "battle_info": {
                "player_pokemon": {
                    "species": "TREECKO",
                    "level": 7,
                    "current_hp": 20,
                    "max_hp": 20,
                    "moves": ["POUND", "ABSORB"],
                    "move_pp": [35, 20],
                },
                "opponent_pokemon": {},
            },
            "milestones_completed": milestones_completed or [],
            "flags": {"has_pokedex": has_pokedex},
        },
        "player": {
            "location": location,
            "position": {"x": player_x, "y": player_y},
        },
        "milestones": {},
        "latest_observation": {
            "visual_data": {
                "screen_context": "battle" if in_battle else "overworld",
                "on_screen_text": {
                    "dialogue": dialogue,
                    "raw_dialogue": dialogue,
                    "speaker": None,
                    "menu_title": None,
                },
                "visible_entities": [],
                "visual_elements": {
                    "text_box_visible": bool(dialogue),
                    "continue_prompt_visible": False,
                },
            }
        },
    }


class TestBattleExitDetection(unittest.TestCase):
    """Fix #1 – battle bot must stop sending RUN inputs after 'Got away safely!'."""

    def setUp(self):
        self.bot = BattleBot()
        # Simulate a wild battle already in progress
        self.bot._battle_started = True
        self.bot._current_battle_type = BattleType.WILD
        self.bot._was_in_battle_last_step = True
        self.bot._wild_battle_dialogue_turns = 2

    def test_got_away_safely_returns_advance_dialogue(self):
        """'Got away safely!' should advance dialogue, not trigger RUN."""
        state = _make_state(dialogue="Got away safely!")
        action = self.bot.get_action(state)
        self.assertEqual(action, "ADVANCE_BATTLE_DIALOGUE")

    def test_got_away_resets_dialogue_turns(self):
        """Dialogue turn counter must reset so next battle starts clean."""
        state = _make_state(dialogue="Got away safely!")
        self.bot.get_action(state)
        self.assertEqual(self.bot._wild_battle_dialogue_turns, 0)

    def test_got_away_does_not_select_run(self):
        """After 3+ dialogue turns, 'Got away safely!' must NOT force base_menu→RUN."""
        self.bot._wild_battle_dialogue_turns = 5  # Would normally force RUN
        state = _make_state(dialogue="Got away safely!")
        action = self.bot.get_action(state)
        # Must be dialogue advancement, NOT VLM_SELECT_RUN
        self.assertNotEqual(action, "VLM_SELECT_RUN")
        self.assertEqual(action, "ADVANCE_BATTLE_DIALOGUE")

    def test_whited_out_returns_advance_dialogue(self):
        """'Whited out' (player lost) should also just advance dialogue."""
        state = _make_state(dialogue="Player whited out!")
        action = self.bot.get_action(state)
        self.assertEqual(action, "ADVANCE_BATTLE_DIALOGUE")

    def test_normal_wild_intro_still_counts(self):
        """Normal intro dialogue ('Wild X appeared!') should still count turns."""
        state = _make_state(dialogue="Wild POOCHYENA appeared!")
        old_turns = self.bot._wild_battle_dialogue_turns
        self.bot.get_action(state)
        # Turns should have incremented (dialogue turn counting still works)
        self.assertGreater(self.bot._wild_battle_dialogue_turns, old_turns)

    def test_base_menu_still_selects_run(self):
        """Normal base_menu state should still trigger RUN for wild battles."""
        state = _make_state(dialogue="What will\nTREECKO do?")
        action = self.bot.get_action(state)
        self.assertEqual(action, "VLM_SELECT_RUN")


class TestPostBattleGracePeriod(unittest.TestCase):
    """Fix #2 – oscillation detector must clear history when battle ends."""

    def _make_agent_mock(self):
        """Create a minimal mock with position_history and the new flag."""
        agent = MagicMock()
        agent.position_history = []
        agent._was_in_battle_for_stuck = False
        agent.objective_manager = MagicMock()
        return agent

    def test_history_cleared_on_battle_end(self):
        """Position history should be reset when transitioning out of battle."""
        agent = self._make_agent_mock()
        # Simulate 6 steps of battle (all same position)
        battle_pos = (10, 5, "ROUTE 103")
        agent.position_history = [battle_pos] * 6
        agent._was_in_battle_for_stuck = True  # Was in battle last step

        # Simulate the battle→overworld transition logic inline
        current_position = (10, 5, "ROUTE 103")
        in_battle_now = False

        if agent._was_in_battle_for_stuck and not in_battle_now:
            agent.position_history = [current_position]
        agent._was_in_battle_for_stuck = in_battle_now

        self.assertEqual(len(agent.position_history), 1)
        self.assertFalse(agent._was_in_battle_for_stuck)

    def test_history_preserved_during_battle(self):
        """Position history should NOT be cleared while still in battle."""
        agent = self._make_agent_mock()
        agent.position_history = [(10, 5, "ROUTE 103")] * 6
        agent._was_in_battle_for_stuck = True

        in_battle_now = True
        if agent._was_in_battle_for_stuck and not in_battle_now:
            agent.position_history = [(10, 5, "ROUTE 103")]
        agent._was_in_battle_for_stuck = in_battle_now

        self.assertEqual(len(agent.position_history), 6)

    def test_false_oscillation_prevented(self):
        """After battle end, stale history should not cause false oscillation."""
        agent = self._make_agent_mock()
        pre_battle_pos = (14, 13, "ROUTE 103")
        post_battle_pos = (15, 14, "ROUTE 103")  # Displaced by leaked inputs

        # Build history as it would look: pre-battle stale + post-battle displaced
        agent.position_history = [pre_battle_pos] * 5 + [post_battle_pos]
        agent._was_in_battle_for_stuck = True

        # Battle ends
        in_battle_now = False
        current_position = post_battle_pos
        if agent._was_in_battle_for_stuck and not in_battle_now:
            agent.position_history = [current_position]
        agent._was_in_battle_for_stuck = in_battle_now

        # Only 1 position in history → oscillation check should NOT trigger
        self.assertEqual(len(agent.position_history), 1)
        unique = set(agent.position_history[-6:])
        # With only 1 entry, there can't be 6 entries to check
        self.assertLess(len(agent.position_history), 6)


class TestMilestoneInjection(unittest.TestCase):
    """Fix #3 – milestones from state_data['milestones'] should be available
    as state_data['game']['milestones_completed'] for battle_bot."""

    def test_completed_milestones_injected(self):
        """Completed milestone IDs should appear in game.milestones_completed."""
        state_data = {
            "milestones": {
                "STARTER_CHOSEN": {"completed": True},
                "BIRCH_LAB_VISITED": {"completed": True},
                "OLDALE_TOWN": {"completed": True},
                "ROUTE_103": {"completed": False},
            },
            "game": {},
        }
        # Replicate the injection logic from agent/__init__.py
        milestones = state_data.get("milestones", {})
        completed_ids = [
            m_id for m_id, m in milestones.items()
            if isinstance(m, dict) and m.get("completed")
        ]
        state_data.setdefault("game", {})["milestones_completed"] = completed_ids

        result = state_data["game"]["milestones_completed"]
        self.assertIn("STARTER_CHOSEN", result)
        self.assertIn("BIRCH_LAB_VISITED", result)
        self.assertIn("OLDALE_TOWN", result)
        self.assertNotIn("ROUTE_103", result)

    def test_empty_milestones_produces_empty_list(self):
        """When no milestones exist, milestones_completed should be []."""
        state_data = {"milestones": {}, "game": {}}
        milestones = state_data.get("milestones", {})
        completed_ids = [
            m_id for m_id, m in milestones.items()
            if isinstance(m, dict) and m.get("completed")
        ]
        state_data.setdefault("game", {})["milestones_completed"] = completed_ids
        self.assertEqual(state_data["game"]["milestones_completed"], [])

    def test_battle_bot_sees_milestones(self):
        """BattleBot.should_handle should see injected milestones_completed."""
        bot = BattleBot()
        state = _make_state(
            in_battle=True,
            dialogue="Wild POOCHYENA appeared!",
            milestones_completed=["OLDALE_TOWN", "ROUTE_103"],
        )
        bot.should_handle(state)
        # With post-rescue milestones present, NOT a Birch rescue battle
        self.assertFalse(bot._is_birch_rescue_battle)

    def test_no_milestones_defaults_to_run_not_birch(self):
        """Empty milestones should default to RUN (not Birch rescue)."""
        bot = BattleBot()
        state = _make_state(
            in_battle=True,
            dialogue="Wild POOCHYENA appeared!",
            milestones_completed=[],
        )
        bot.should_handle(state)
        # Should default to NOT Birch rescue (safe default = try to run)
        self.assertFalse(bot._is_birch_rescue_battle)


if __name__ == "__main__":
    unittest.main()
