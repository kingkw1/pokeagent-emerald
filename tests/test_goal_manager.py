# tests/test_goal_manager.py
# Now tests ObjectiveManager's blocker/recovery system (ported from GoalManager).
import unittest
from agent.objective_manager import ObjectiveManager


class TestBlockerDetection(unittest.TestCase):
    """Tests for _scan_dialogue_for_blockers and signal_blocker."""

    def setUp(self):
        self.om = ObjectiveManager()

    # ------------------------------------------------------------------
    # Defensive parsing — malformed / missing perception data
    # ------------------------------------------------------------------

    def test_safe_parsing_empty_data(self):
        """Empty dict (missing 'visual_data' entirely) must not crash."""
        self.om._scan_dialogue_for_blockers({})
        self.assertFalse(self.om.is_blocked)

    def test_safe_parsing_none_dialogue(self):
        """VLM explicitly sets dialogue=None on false-positive filtering."""
        mock = {
            "visual_data": {
                "screen_context": "overworld",
                "on_screen_text": {"dialogue": None, "speaker": None},
            }
        }
        self.om._scan_dialogue_for_blockers(mock)
        self.assertFalse(self.om.is_blocked)

    def test_safe_parsing_on_screen_text_is_string(self):
        """VLM sometimes returns on_screen_text as a raw string instead of dict."""
        mock = {
            "visual_data": {
                "screen_context": "dialogue",
                "on_screen_text": "Wait! Don't go out there!",
            }
        }
        self.om._scan_dialogue_for_blockers(mock)
        self.assertTrue(self.om.is_blocked)

    def test_safe_parsing_on_screen_text_is_none(self):
        """on_screen_text could be None rather than missing."""
        mock = {
            "visual_data": {
                "screen_context": "overworld",
                "on_screen_text": None,
            }
        }
        self.om._scan_dialogue_for_blockers(mock)
        self.assertFalse(self.om.is_blocked)

    # ------------------------------------------------------------------
    # Blocker detection
    # ------------------------------------------------------------------

    def test_blocker_detection_keyword(self):
        """Specific dialogue keywords trigger the BLOCKED state."""
        mock = {
            "visual_data": {
                "screen_context": "overworld",
                "on_screen_text": {
                    "dialogue": "Wait! Don't go out into the tall grass!",
                    "speaker": "Prof. Birch",
                },
            }
        }
        self.om._scan_dialogue_for_blockers(mock)
        self.assertTrue(self.om.is_blocked)

    def test_non_blocking_dialogue(self):
        """Normal dialogue must NOT trigger a block."""
        mock = {
            "visual_data": {
                "screen_context": "overworld",
                "on_screen_text": {
                    "dialogue": "Hello there! Nice weather today.",
                    "speaker": "Townsfolk",
                },
            }
        }
        self.om._scan_dialogue_for_blockers(mock)
        self.assertFalse(self.om.is_blocked)

    def test_repeated_blocking_is_idempotent(self):
        """Seeing the same blocker on consecutive frames must not double-trigger."""
        mock = {
            "visual_data": {
                "screen_context": "dialogue",
                "on_screen_text": {
                    "dialogue": "Wait! It's dangerous!",
                    "speaker": "Old Man",
                },
            }
        }
        self.om._scan_dialogue_for_blockers(mock)
        self.om._scan_dialogue_for_blockers(mock)
        self.om._scan_dialogue_for_blockers(mock)
        # Still blocked, no error from repeated calls
        self.assertTrue(self.om.is_blocked)

    def test_signal_blocker_external(self):
        """signal_blocker() triggers blocked state from external call."""
        self.om.signal_blocker(reason="Trainer Battle", context="Wild encounter")
        self.assertTrue(self.om.is_blocked)

    # ------------------------------------------------------------------
    # Recovery task stack
    # ------------------------------------------------------------------

    def test_add_recovery_task_directive(self):
        """After adding a recovery task, current_brain_directive reflects it."""
        self.om.signal_blocker(reason="test", context="test")
        self.om.add_recovery_task("Interact with Old Man for tutorial")
        self.assertIn("Interact with Old Man", self.om.current_brain_directive)

    def test_complete_recovery_task_pops_stack(self):
        """Completing recovery task removes it from the stack."""
        self.om.add_recovery_task("Talk to NPC")
        self.om.add_recovery_task("Watch tutorial")
        self.om.complete_recovery_task()
        # "Watch tutorial" was on top, now "Talk to NPC" is
        self.assertIn("Talk to NPC", self.om.current_brain_directive)

    def test_complete_recovery_on_empty_stack(self):
        """Completing when no recovery tasks exist does not crash."""
        self.om.complete_recovery_task()  # should be a no-op

    # ------------------------------------------------------------------
    # Clear blocker
    # ------------------------------------------------------------------

    def test_clear_blocker(self):
        """clear_blocker() exits the BLOCKED state."""
        self.om.signal_blocker(reason="test", context="test")
        self.assertTrue(self.om.is_blocked)
        self.om.clear_blocker()
        self.assertFalse(self.om.is_blocked)


if __name__ == "__main__":
    unittest.main()
