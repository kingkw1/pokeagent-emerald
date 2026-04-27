"""
tests/test_handoff_detector.py — Phase 1 unit tests for handoff_detector_node.

Covers:
  TestSignificantTransition   — listed node-type changes set supervisor_pending=True
  TestInsignificantTransition — same-node re-entries leave supervisor_pending=False
  TestFirstStep               — absent last_node_fired → supervisor_pending=True
  TestEmptyStack              — empty goal_stack → supervisor_pending=True
  TestLastNodeFiredUpdated    — last_node_fired written correctly from last_action
  TestNavStallDetection       — position stall triggers Supervisor at threshold

IMPORTANT: Every test that exercises the nav-stall globals must reset them in
a fixture or at the start of the test.  Use the module-level reset helper
``_reset_stall_state()`` defined below.
"""

from __future__ import annotations

import pytest

import agent.graph.nodes.handoff_detector as hd
from agent.graph.nodes.handoff_detector import handoff_detector_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_stall_state() -> None:
    """Reset module-level nav-stall globals to avoid inter-test contamination."""
    hd._consecutive_nav_stall_steps = 0
    hd._last_nav_position = None
    hd._prev_goal_stack_was_populated = False


def _state(
    last_action: str = "NAVIGATE",
    last_node_fired: str = "nav_bot",
    goal_stack: list | None = None,
    state_data: dict | None = None,
) -> dict:
    """Return a minimal AgentState-compatible dict."""
    return {
        "last_action":    last_action,
        "last_node_fired": last_node_fired,
        "goal_stack":     goal_stack if goal_stack is not None else [{"goal_id": "g"}],
        "state_data":     state_data or {},
        "step_count":     0,
    }


def _nav_state(x: int = 10, y: int = 20, location: str = "ROUTE_101") -> dict:
    """Return a state_data dict that mimics the game's player-position structure."""
    return {
        "player": {
            "position": {"x": x, "y": y},
            "location": location,
        }
    }


@pytest.fixture(autouse=True)
def reset_stall_globals():
    """Auto-reset nav-stall module state before every test."""
    _reset_stall_state()
    yield
    _reset_stall_state()


# ---------------------------------------------------------------------------
# TestSignificantTransition
# ---------------------------------------------------------------------------

class TestSignificantTransition:
    def test_battle_to_nav(self):
        s = _state(last_action="NAVIGATE", last_node_fired="battle_bot")
        assert handoff_detector_node(s)["supervisor_pending"] is True

    def test_coms_to_nav(self):
        s = _state(last_action="NAVIGATE", last_node_fired="coms_bot")
        assert handoff_detector_node(s)["supervisor_pending"] is True

    def test_nav_to_coms(self):
        s = _state(last_action="DIALOGUE", last_node_fired="nav_bot")
        assert handoff_detector_node(s)["supervisor_pending"] is True

    def test_nav_to_battle(self):
        s = _state(last_action="BATTLE", last_node_fired="nav_bot")
        assert handoff_detector_node(s)["supervisor_pending"] is True

    def test_battle_to_coms(self):
        s = _state(last_action="DIALOGUE", last_node_fired="battle_bot")
        assert handoff_detector_node(s)["supervisor_pending"] is True

    def test_map_stitcher_to_nav(self):
        s = _state(last_action="NAVIGATE", last_node_fired="map_stitcher_relay")
        assert handoff_detector_node(s)["supervisor_pending"] is True


# ---------------------------------------------------------------------------
# TestInsignificantTransition
# ---------------------------------------------------------------------------

class TestInsignificantTransition:
    def test_nav_to_nav_with_stack(self):
        """Repeated nav_bot entries while moving do NOT wake the Supervisor."""
        s = _state(
            last_action="NAVIGATE",
            last_node_fired="nav_bot",
            state_data=_nav_state(x=10, y=20),
        )
        result = handoff_detector_node(s)
        assert result["supervisor_pending"] is False

    def test_battle_to_battle(self):
        s = _state(last_action="BATTLE", last_node_fired="battle_bot")
        assert handoff_detector_node(s)["supervisor_pending"] is False

    def test_coms_to_coms(self):
        s = _state(last_action="DIALOGUE", last_node_fired="coms_bot")
        assert handoff_detector_node(s)["supervisor_pending"] is False


# ---------------------------------------------------------------------------
# TestFirstStep
# ---------------------------------------------------------------------------

class TestFirstStep:
    def test_no_previous_node(self):
        """last_node_fired absent/empty → first step → supervisor_pending=True."""
        s = _state(last_action="NAVIGATE", last_node_fired="")
        assert handoff_detector_node(s)["supervisor_pending"] is True

    def test_none_previous_node(self):
        s = {
            "last_action":     "NAVIGATE",
            "last_node_fired": None,
            "goal_stack":      [{"goal_id": "g"}],
            "state_data":      {},
        }
        assert handoff_detector_node(s)["supervisor_pending"] is True

    def test_missing_last_node_fired_key(self):
        s = {
            "last_action": "NAVIGATE",
            "goal_stack":  [{"goal_id": "g"}],
            "state_data":  {},
        }
        assert handoff_detector_node(s)["supervisor_pending"] is True


# ---------------------------------------------------------------------------
# TestEmptyStack
# ---------------------------------------------------------------------------
# The "empty stack" trigger uses a *transition* detector: it only fires when the
# stack goes from non-empty → empty, NOT every step the stack happens to be
# empty.  This prevents spurious pending=True throughout Phase 1–3 before the
# Supervisor has ever populated the stack.

class TestEmptyStack:
    def test_stack_transition_to_empty_triggers(self):
        """Stack going from non-empty → empty fires supervisor_pending=True."""
        # Step 1: non-empty stack — records _prev_goal_stack_was_populated = True
        s1 = _state(
            last_action="NAVIGATE", last_node_fired="nav_bot",
            goal_stack=[{"goal_id": "g"}],
            state_data=_nav_state(x=1),
        )
        handoff_detector_node(s1)
        # Step 2: stack now empty — transition fires
        s2 = _state(
            last_action="NAVIGATE", last_node_fired="nav_bot",
            goal_stack=[],
            state_data=_nav_state(x=2),  # different pos → no stall
        )
        result = handoff_detector_node(s2)
        assert result["supervisor_pending"] is True

    def test_always_empty_stack_does_not_fire(self):
        """Stack empty from the start (never populated) → no spurious trigger."""
        s = _state(
            last_action="NAVIGATE", last_node_fired="nav_bot",
            goal_stack=[],
            state_data=_nav_state(),
        )
        result = handoff_detector_node(s)
        assert result["supervisor_pending"] is False

    def test_none_stack_treated_as_empty_no_trigger(self):
        """None goal_stack with no prior population → no trigger."""
        s = {
            "last_action":     "NAVIGATE",
            "last_node_fired": "nav_bot",
            "goal_stack":      None,
            "state_data":      _nav_state(),
        }
        result = handoff_detector_node(s)
        assert result["supervisor_pending"] is False

    def test_missing_goal_stack_no_trigger(self):
        """Missing goal_stack key with no prior population → no trigger."""
        s = {
            "last_action":     "NAVIGATE",
            "last_node_fired": "nav_bot",
            "state_data":      _nav_state(),
        }
        result = handoff_detector_node(s)
        assert result["supervisor_pending"] is False


# ---------------------------------------------------------------------------
# TestLastNodeFiredUpdated
# ---------------------------------------------------------------------------

class TestLastNodeFiredUpdated:
    def test_navigate_sets_nav_bot(self):
        s = _state(last_action="NAVIGATE", last_node_fired="nav_bot")
        assert handoff_detector_node(s)["last_node_fired"] == "nav_bot"

    def test_battle_sets_battle_bot(self):
        s = _state(last_action="BATTLE", last_node_fired="battle_bot")
        assert handoff_detector_node(s)["last_node_fired"] == "battle_bot"

    def test_dialogue_sets_coms_bot(self):
        s = _state(last_action="DIALOGUE", last_node_fired="nav_bot")
        assert handoff_detector_node(s)["last_node_fired"] == "coms_bot"

    def test_unknown_action_passes_through(self):
        """Unmapped last_action values are stored as-is."""
        s = _state(last_action="UNKNOWN_ACTION", last_node_fired="nav_bot")
        assert handoff_detector_node(s)["last_node_fired"] == "UNKNOWN_ACTION"

    def test_previous_state_preserved(self):
        """State keys other than last_node_fired and supervisor_pending are unchanged."""
        s = _state(last_action="NAVIGATE", last_node_fired="nav_bot")
        s["step_count"] = 42
        result = handoff_detector_node(s)
        assert result["step_count"] == 42


# ---------------------------------------------------------------------------
# TestNavStallDetection
# ---------------------------------------------------------------------------

class TestNavStallDetection:
    """Nav-stall tests exercise module-level globals; autouse fixture resets them."""

    _POS = {"state_data": _nav_state(x=5, y=5, location="ROUTE_101")}

    def _nav_s(self, x: int = 5, y: int = 5, loc: str = "ROUTE_101") -> dict:
        return _state(
            last_action="NAVIGATE",
            last_node_fired="nav_bot",
            state_data=_nav_state(x=x, y=y, location=loc),
        )

    def test_below_threshold_no_trigger(self):
        """14 steps at the same position → supervisor_pending=False throughout."""
        for _ in range(hd._NAV_STALL_THRESHOLD - 1):
            result = handoff_detector_node(self._nav_s())
            assert result["supervisor_pending"] is False

    def test_at_threshold_triggers(self):
        """The step that pushes the counter to exactly _NAV_STALL_THRESHOLD triggers."""
        # Call 1:  counter=0 (position initialised, no match yet)
        # Calls 2.._NAV_STALL_THRESHOLD: counter = 1 .. threshold-1
        # Call _NAV_STALL_THRESHOLD+1: counter = threshold → trigger
        for _ in range(hd._NAV_STALL_THRESHOLD):   # threshold calls to set up
            handoff_detector_node(self._nav_s())
        result = handoff_detector_node(self._nav_s())  # threshold+1-th call fires
        assert result["supervisor_pending"] is True

    def test_threshold_resets_counter(self):
        """The call immediately after the threshold fire has supervisor_pending=False."""
        # Run threshold+1 calls so the last one is the trigger call
        for _ in range(hd._NAV_STALL_THRESHOLD + 1):
            handoff_detector_node(self._nav_s())
        # Next call: counter was reset to 0 after trigger; increments to 1 now
        result = handoff_detector_node(self._nav_s())
        assert result["supervisor_pending"] is False

    def test_moving_position_never_triggers(self):
        """Position changes every step → stall counter never reaches threshold."""
        for i in range(hd._NAV_STALL_THRESHOLD + 5):
            result = handoff_detector_node(self._nav_s(x=i, y=0))
            assert result["supervisor_pending"] is False

    def test_leaving_nav_resets_counter(self):
        """Switching to battle_bot resets the stall counter; returning to nav_bot
        does not inherit stall history."""
        # Accumulate 14 stall steps
        for _ in range(hd._NAV_STALL_THRESHOLD - 1):
            handoff_detector_node(self._nav_s())

        # Switch to battle_bot — counter should reset
        battle_s = _state(last_action="BATTLE", last_node_fired="battle_bot")
        handoff_detector_node(battle_s)
        assert hd._consecutive_nav_stall_steps == 0

        # Return to nav_bot — first step here should NOT trigger stall
        result = handoff_detector_node(self._nav_s())
        # supervisor_pending is True here because battle_bot → nav_bot is a
        # significant transition, not a stall — that's fine
        assert hd._consecutive_nav_stall_steps <= 1  # at most 1 step counted

    def test_stall_counter_increments(self):
        """Verify the counter increments correctly.
        Call 1: position initialised, counter=0 (no previous position to match).
        Call 2+: same position matches → counter increments by 1 each time.
        """
        handoff_detector_node(self._nav_s())          # call 1: sets position, counter=0
        assert hd._consecutive_nav_stall_steps == 0
        for expected in range(1, 6):                  # calls 2-6: counter = 1,2,3,4,5
            handoff_detector_node(self._nav_s())
            assert hd._consecutive_nav_stall_steps == expected

    def test_missing_state_data_does_not_crash(self):
        """state_data missing or empty → nav_pos is (None, None, None) and
        the node runs without raising."""
        s = _state(last_action="NAVIGATE", last_node_fired="nav_bot", state_data={})
        result = handoff_detector_node(s)
        # No exception; supervisor_pending determined by other conditions
        assert "supervisor_pending" in result
