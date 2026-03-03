#!/usr/bin/env python3
"""
Opener Bot Test Suite

Tests the STATEFUL opener bot state machine for Pokemon Emerald opening sequence.
Updated to match the current stateful API (S0_TITLE_SCREEN through COMPLETED).
"""
import unittest
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.opener_bot import OpenerBot, get_opener_bot, NavigationGoal


class TestStateDetection(unittest.TestCase):
    """Test _detect_starting_state for various game scenarios"""

    def setUp(self):
        self.bot = OpenerBot()

    def test_default_title_screen(self):
        """No milestones → S0_TITLE_SCREEN"""
        state_data = {
            'game': {'state': 'title'},
            'player': {'name': '', 'location': '', 'position': {'x': 0, 'y': 0}},
            'milestones': {},
        }
        detected = self.bot._detect_starting_state(state_data)
        self.assertEqual(detected, 'S0_TITLE_SCREEN')

    def test_moving_van(self):
        """Location MOVING_VAN → S3_TRUCK_RIDE"""
        state_data = {
            'game': {'state': 'running'},
            'player': {'name': 'CASEY', 'location': 'MOVING_VAN', 'position': {'x': 8, 'y': 5}},
            'milestones': {'PLAYER_NAME_SET': {'completed': True}},
        }
        detected = self.bot._detect_starting_state(state_data)
        self.assertEqual(detected, 'S3_TRUCK_RIDE')

    def test_players_house_2f(self):
        """On 2F of player's house → S6_NAV_TO_CLOCK"""
        state_data = {
            'game': {'state': 'running'},
            'player': {'name': 'CASEY', 'location': 'PLAYERS_HOUSE_2F', 'position': {'x': 5, 'y': 3}},
            'milestones': {'PLAYER_NAME_SET': {'completed': True}},
        }
        detected = self.bot._detect_starting_state(state_data)
        self.assertEqual(detected, 'S6_NAV_TO_CLOCK')

    def test_players_house_1f_after_house_entered(self):
        """1F of house after already entering → S5_NAV_TO_STAIRS_1F"""
        state_data = {
            'game': {'state': 'running'},
            'player': {'name': 'CASEY', 'location': 'PLAYERS_HOUSE_1F', 'position': {'x': 8, 'y': 7}},
            'milestones': {
                'PLAYER_NAME_SET': {'completed': True},
                'PLAYER_HOUSE_ENTERED': {'completed': True},
            },
        }
        detected = self.bot._detect_starting_state(state_data)
        self.assertEqual(detected, 'S5_NAV_TO_STAIRS_1F')

    def test_littleroot_after_rival_bedroom(self):
        """In Littleroot after visiting rival's bedroom → S15_NAV_TO_NPC_NORTH"""
        state_data = {
            'game': {'state': 'running'},
            'player': {'name': 'CASEY', 'location': 'LITTLEROOT TOWN', 'position': {'x': 11, 'y': 10}},
            'milestones': {
                'PLAYER_NAME_SET': {'completed': True},
                'RIVAL_BEDROOM': {'completed': True},
            },
        }
        detected = self.bot._detect_starting_state(state_data)
        self.assertEqual(detected, 'S15_NAV_TO_NPC_NORTH')

    def test_route101_no_starter(self):
        """On Route 101 without starter near bag → S19_NAV_TO_BAG"""
        state_data = {
            'game': {'state': 'running'},
            'player': {'name': 'CASEY', 'location': 'ROUTE 101', 'position': {'x': 7, 'y': 12}},
            'milestones': {'PLAYER_NAME_SET': {'completed': True}},
        }
        detected = self.bot._detect_starting_state(state_data)
        self.assertEqual(detected, 'S19_NAV_TO_BAG')

    def test_birchs_lab_with_starter(self):
        """In Birch's Lab with starter → S23_BIRCH_DIALOG_2"""
        state_data = {
            'game': {'state': 'running'},
            'player': {'name': 'CASEY', 'location': 'PROFESSOR BIRCHS LAB', 'position': {'x': 5, 'y': 5}},
            'milestones': {'STARTER_CHOSEN': {'completed': True}},
            'party': [{'species': 'TREECKO', 'hp_current': 20}],
        }
        detected = self.bot._detect_starting_state(state_data)
        self.assertEqual(detected, 'S23_BIRCH_DIALOG_2')


class TestShouldHandle(unittest.TestCase):
    """Test should_handle decision logic"""

    def setUp(self):
        self.bot = OpenerBot()

    def _make_state(self, location='', milestones=None):
        return {
            'game': {'state': 'running'},
            'player': {'name': 'CASEY', 'location': location, 'position': {'x': 5, 'y': 5}},
            'milestones': milestones or {},
        }

    def test_active_before_starter(self):
        """Bot should be active before starter is chosen"""
        result = self.bot.should_handle(
            self._make_state(location='LITTLEROOT TOWN'),
            {},
        )
        self.assertTrue(result)

    def test_active_in_lab_with_starter(self):
        """Bot should still be active in lab with starter (handling S23/S24)"""
        result = self.bot.should_handle(
            self._make_state(
                location='PROFESSOR BIRCHS LAB',
                milestones={'STARTER_CHOSEN': {'completed': True}},
            ),
            {},
        )
        self.assertTrue(result)

    def test_inactive_outside_lab_with_starter(self):
        """Bot should hand off once outside lab after getting starter"""
        result = self.bot.should_handle(
            self._make_state(
                location='LITTLEROOT TOWN',
                milestones={'STARTER_CHOSEN': {'completed': True}},
            ),
            {},
        )
        self.assertFalse(result)
        self.assertEqual(self.bot.current_state_name, 'COMPLETED')

    def test_completed_is_permanent(self):
        """Once COMPLETED, should_handle always returns False (no reactivation)"""
        self.bot._transition_to_state('COMPLETED')

        # Even if still in lab with starter, COMPLETED is permanent
        result = self.bot.should_handle(
            self._make_state(
                location='PROFESSOR BIRCHS LAB',
                milestones={'STARTER_CHOSEN': {'completed': True}},
            ),
            {},
        )
        self.assertFalse(result)


class TestStateTransitions(unittest.TestCase):
    """Test state machine transitions"""

    def setUp(self):
        self.bot = OpenerBot()

    def test_s23_to_completed_no_nickname(self):
        """S23 transitions to COMPLETED when dialogue clears and no nickname"""
        self.bot._transition_to_state('S23_BIRCH_DIALOG_2')
        self.bot.initialized_state = True  # Prevent auto-detection override
        state_data = {
            'game': {'state': 'running', 'game_state': 'overworld'},
            'player': {'name': 'CASEY', 'location': 'PROFESSOR BIRCHS LAB', 'position': {'x': 5, 'y': 5}},
            'milestones': {'STARTER_CHOSEN': {'completed': True}},
            'party': [{'species': 'TREECKO', 'hp_current': 20}],
        }
        visual_data = {
            'screen_context': 'overworld',
            'visual_elements': {'text_box_visible': False, 'continue_prompt_visible': False},
            'on_screen_text': {'dialogue': '', 'menu_title': ''},
        }

        # Get action — transition logic should fire
        self.bot.get_action(state_data, visual_data)
        self.assertEqual(self.bot.current_state_name, 'COMPLETED')

    def test_s24_to_completed(self):
        """S24 transitions to COMPLETED when nickname text clears"""
        self.bot._transition_to_state('S24_NICKNAME')
        self.bot.initialized_state = True  # Prevent auto-detection override
        state_data = {
            'game': {'state': 'running', 'game_state': 'overworld'},
            'player': {'name': 'CASEY', 'location': 'PROFESSOR BIRCHS LAB', 'position': {'x': 5, 'y': 5}},
            'milestones': {'STARTER_CHOSEN': {'completed': True}},
            'party': [{'species': 'TREECKO', 'hp_current': 20}],
        }
        visual_data = {
            'screen_context': 'overworld',
            'visual_elements': {'text_box_visible': False, 'continue_prompt_visible': False},
            'on_screen_text': {'dialogue': '', 'menu_title': ''},
        }

        self.bot.get_action(state_data, visual_data)
        self.assertEqual(self.bot.current_state_name, 'COMPLETED')


class TestSafetyLimits(unittest.TestCase):
    """Test safety fallbacks (max attempts, timeouts)"""

    def setUp(self):
        self.bot = OpenerBot()

    def test_timeout_triggers_completed(self):
        """State timeout should hand off to VLM by transitioning to COMPLETED"""
        self.bot._transition_to_state('S0_TITLE_SCREEN')
        self.bot.state_entry_time = time.time() - 200  # 200 seconds ago (limit is 180)

        state_data = {
            'game': {'state': 'title', 'game_state': 'title'},
            'player': {'name': '', 'location': '', 'position': {'x': 0, 'y': 0}},
            'milestones': {},
        }
        visual_data = {
            'screen_context': 'title',
            'visual_elements': {'text_box_visible': False, 'continue_prompt_visible': False},
            'on_screen_text': {'dialogue': '', 'menu_title': ''},
        }

        action = self.bot.get_action(state_data, visual_data)
        self.assertIsNone(action)
        self.assertEqual(self.bot.current_state_name, 'COMPLETED')

    def test_max_attempts_triggers_completed(self):
        """Exceeding max_attempts should hand off to VLM"""
        self.bot._transition_to_state('S0_TITLE_SCREEN')
        state = self.bot.states['S0_TITLE_SCREEN']

        state_data = {
            'game': {'state': 'title', 'game_state': 'title'},
            'player': {'name': '', 'location': 'TITLE_SEQUENCE', 'position': {'x': 0, 'y': 0}},
            'milestones': {},
        }
        visual_data = {
            'screen_context': 'title',
            'visual_elements': {'text_box_visible': False, 'continue_prompt_visible': False},
            'on_screen_text': {'dialogue': '', 'menu_title': ''},
        }

        # Exhaust max attempts
        for _ in range(state.max_attempts + 1):
            self.bot.get_action(state_data, visual_data)

        self.assertEqual(self.bot.current_state_name, 'COMPLETED')


class TestGlobalInstance(unittest.TestCase):
    """Test singleton global instance management"""

    def test_same_instance(self):
        """get_opener_bot returns the same instance on repeated calls"""
        bot1 = get_opener_bot()
        bot2 = get_opener_bot()
        self.assertIs(bot1, bot2)


class TestNavigationGoal(unittest.TestCase):
    """Test NavigationGoal data class"""

    def test_basic_goal(self):
        goal = NavigationGoal(x=8, y=1, map_location='MOVING_VAN', description='Exit Van')
        self.assertEqual(goal.x, 8)
        self.assertEqual(goal.y, 1)
        self.assertIsNone(goal.should_interact)

    def test_interaction_goal(self):
        goal = NavigationGoal(x=5, y=1, map_location='HOUSE_2F', description='Clock', should_interact=True)
        self.assertTrue(goal.should_interact)


if __name__ == '__main__':
    unittest.main()
