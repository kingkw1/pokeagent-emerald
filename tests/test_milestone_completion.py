"""
tests/test_milestone_completion.py — Phase 5.1 automated tests.

Verifies that:
  * ``completion_type="location"`` milestones still complete on ROM flag / coord match.
  * ``completion_type="dialogue"`` milestones do NOT complete on ROM flag alone.
  * ``completion_type="dialogue"`` milestones DO complete when dialogue_completed=True.
  * TransitionEvaluator returns the correct verdict.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent.objective_manager import (
    MILESTONE_PROGRESSION,
    ObjectiveManager,
    _MILESTONE_COMPLETION_TYPE,
)
from agent.graph.nodes.verification import make_verification_node
from agent.graph.transition_evaluator import TransitionEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides):
    """Return a minimal AgentState-like dict for testing."""
    base = {
        "milestone_index": 0,
        "state_data": {
            "game": {},
            "player": {"location": "", "position": {"x": 0, "y": 0}},
            "milestones": {},
            "party": [],
        },
        "perception": {},
        "context": "navigation",
        "dialogue_completed": False,
        "dialogue_transcript": [],
        "goal_coords": None,
        "goal_location": None,
        "npc_coords": None,
        "should_interact": False,
        "last_action": None,
        "last_buttons": [],
        "step_count": 0,
        "reward": None,
        "prev_state_snapshot": None,
        "telemetry": None,
        "frame": None,
    }
    base.update(overrides)
    return base


def _make_obj_manager():
    return ObjectiveManager()


def _milestone_index(milestone_id: str) -> int:
    for i, m in enumerate(MILESTONE_PROGRESSION):
        if m["milestone"] == milestone_id:
            return i
    raise KeyError(milestone_id)


# ---------------------------------------------------------------------------
# 5.1a: completion_type field present for all milestones
# ---------------------------------------------------------------------------

class TestCompletionTypeField:
    def test_all_milestones_have_completion_type(self):
        """Every entry in MILESTONE_PROGRESSION must have completion_type."""
        for m in MILESTONE_PROGRESSION:
            assert "completion_type" in m, (
                f"Milestone {m['milestone']} is missing completion_type"
            )

    def test_completion_type_values_are_valid(self):
        valid = {"location", "battle", "dialogue"}
        for m in MILESTONE_PROGRESSION:
            assert m["completion_type"] in valid, (
                f"Milestone {m['milestone']} has invalid completion_type: {m['completion_type']!r}"
            )

    def test_dialogue_milestones_have_keywords(self):
        for m in MILESTONE_PROGRESSION:
            if m["completion_type"] == "dialogue":
                assert "dialogue_keywords" in m and m["dialogue_keywords"], (
                    f"Dialogue milestone {m['milestone']} missing dialogue_keywords"
                )

    def test_milestone_completion_type_lookup_built(self):
        assert isinstance(_MILESTONE_COMPLETION_TYPE, dict)
        assert len(_MILESTONE_COMPLETION_TYPE) == len(MILESTONE_PROGRESSION)

    def test_dad_first_meeting_is_dialogue(self):
        assert _MILESTONE_COMPLETION_TYPE["DAD_FIRST_MEETING"] == "dialogue"

    def test_gym_explanation_is_dialogue(self):
        assert _MILESTONE_COMPLETION_TYPE["GYM_EXPLANATION"] == "dialogue"

    def test_location_milestones_correct_type(self):
        assert _MILESTONE_COMPLETION_TYPE["PETALBURG_CITY"] == "location"
        assert _MILESTONE_COMPLETION_TYPE["RUSTBORO_CITY"] == "location"

    def test_battle_milestones_correct_type(self):
        assert _MILESTONE_COMPLETION_TYPE["RIVAL_BATTLE_1"] == "battle"
        assert _MILESTONE_COMPLETION_TYPE["ROXANNE_DEFEATED"] == "battle"


# ---------------------------------------------------------------------------
# 5.1b: Location milestone — unchanged behaviour
# ---------------------------------------------------------------------------

class TestLocationMilestoneUnchanged:
    def test_location_milestone_completes_on_rom_flag(self):
        """completion_type='location' advances when completed_goals is set."""
        obj_manager = _make_obj_manager()
        # Pre-populate completed_goals as if ROM flag fired for PETALBURG_CITY
        obj_manager.completed_goals["PETALBURG_CITY"] = True

        idx = _milestone_index("PETALBURG_CITY")
        state = _make_state(
            milestone_index=idx,
            state_data={
                "game": {},
                "player": {"location": "PETALBURG CITY", "position": {"x": 0, "y": 0}},
                "milestones": {"PETALBURG_CITY": {"completed": True}},
                "party": [],
            },
        )

        node = make_verification_node(obj_manager)
        result = node(state)

        assert result["milestone_index"] == idx + 1

    def test_location_milestone_no_advance_without_flag(self):
        """Location milestone stays open when ROM flag not set."""
        obj_manager = _make_obj_manager()

        idx = _milestone_index("PETALBURG_CITY")
        state = _make_state(
            milestone_index=idx,
            state_data={
                "game": {},
                "player": {"location": "", "position": {"x": 0, "y": 0}},
                "milestones": {},
                "party": [],
            },
        )

        node = make_verification_node(obj_manager)
        result = node(state)

        assert result["milestone_index"] == idx


# ---------------------------------------------------------------------------
# 5.1c: Dialogue milestone — gated on dialogue_completed flag
# ---------------------------------------------------------------------------

class TestDialogueMilestoneGated:
    def test_dialogue_milestone_does_not_complete_on_rom_flag_alone(self):
        """DAD_FIRST_MEETING ROM flag fires → milestone must NOT advance."""
        obj_manager = _make_obj_manager()
        idx = _milestone_index("DAD_FIRST_MEETING")
        state = _make_state(
            milestone_index=idx,
            dialogue_completed=False,
            state_data={
                "game": {},
                "player": {
                    "location": "PETALBURG CITY GYM",
                    "position": {"x": 4, "y": 8},
                },
                # ROM flag set — but dialogue_completed is False
                "milestones": {"DAD_FIRST_MEETING": {"completed": True}},
                "party": [],
            },
        )

        node = make_verification_node(obj_manager)
        result = node(state)

        # Index must NOT advance
        assert result["milestone_index"] == idx

    def test_dialogue_milestone_does_not_advance_when_flag_absent(self):
        """No ROM flag, no dialogue_completed → stays open."""
        obj_manager = _make_obj_manager()
        idx = _milestone_index("DAD_FIRST_MEETING")
        state = _make_state(
            milestone_index=idx,
            dialogue_completed=False,
        )

        node = make_verification_node(obj_manager)
        result = node(state)

        assert result["milestone_index"] == idx


class TestDialogueMilestoneCompletes:
    def test_dialogue_milestone_advances_when_completed_flag_true(self):
        """dialogue_completed=True → DAD_FIRST_MEETING advances."""
        obj_manager = _make_obj_manager()
        idx = _milestone_index("DAD_FIRST_MEETING")
        state = _make_state(
            milestone_index=idx,
            dialogue_completed=True,
        )

        node = make_verification_node(obj_manager)
        result = node(state)

        assert result["milestone_index"] == idx + 1

    def test_dialogue_completed_reset_after_advance(self):
        """dialogue_completed is reset to False after the milestone advances."""
        obj_manager = _make_obj_manager()
        idx = _milestone_index("DAD_FIRST_MEETING")
        state = _make_state(
            milestone_index=idx,
            dialogue_completed=True,
        )

        node = make_verification_node(obj_manager)
        result = node(state)

        assert result.get("dialogue_completed") is False

    def test_gym_explanation_advances_on_dialogue_completed(self):
        """GYM_EXPLANATION also advances on dialogue_completed=True."""
        obj_manager = _make_obj_manager()
        idx = _milestone_index("GYM_EXPLANATION")
        state = _make_state(
            milestone_index=idx,
            dialogue_completed=True,
        )

        node = make_verification_node(obj_manager)
        result = node(state)

        assert result["milestone_index"] == idx + 1

    def test_mark_goal_complete_called_on_dialogue_advance(self):
        """mark_goal_complete is called with the correct milestone ID."""
        obj_manager = _make_obj_manager()
        obj_manager.mark_goal_complete = MagicMock()

        idx = _milestone_index("DAD_FIRST_MEETING")
        state = _make_state(
            milestone_index=idx,
            dialogue_completed=True,
        )

        node = make_verification_node(obj_manager)
        node(state)

        obj_manager.mark_goal_complete.assert_called_once()
        call_args = obj_manager.mark_goal_complete.call_args
        assert call_args[0][0] == "DAD_FIRST_MEETING"


# ---------------------------------------------------------------------------
# 5.2: TransitionEvaluator
# ---------------------------------------------------------------------------

class TestTransitionEvaluatorNoVlm:
    """Keyword-scan path (no VLM)."""

    def test_all_keywords_present_returns_yes(self):
        ev = TransitionEvaluator(vlm=None)
        transcript = [
            {"speaker": "Norman", "text": "I'm the Gym Leader Norman, nice to meet you!", "step": 1},
            {"speaker": "Norman", "text": "I can heal your Pokemon after our chat.", "step": 2},
        ]
        result = ev.evaluate(
            milestone_id="DAD_FIRST_MEETING",
            milestone_description="Meet Dad at Petalburg Gym",
            keywords=["Gym Leader", "Norman", "heal"],
            transcript=transcript,
        )
        assert result == "YES"

    def test_no_keywords_returns_no(self):
        ev = TransitionEvaluator(vlm=None)
        transcript = [{"speaker": "NPC", "text": "The weather is nice today.", "step": 1}]
        result = ev.evaluate(
            milestone_id="DAD_FIRST_MEETING",
            milestone_description="Meet Dad",
            keywords=["Gym Leader", "Norman", "heal"],
            transcript=transcript,
        )
        assert result == "NO"

    def test_empty_transcript_returns_no(self):
        ev = TransitionEvaluator(vlm=None)
        result = ev.evaluate(
            milestone_id="DAD_FIRST_MEETING",
            milestone_description="Meet Dad",
            keywords=["Norman"],
            transcript=[],
        )
        assert result == "NO"

    def test_partial_keywords_returns_partial(self):
        ev = TransitionEvaluator(vlm=None)
        # Only 1 of 4 keywords in text → threshold = max(1, 4//2) = 2 → PARTIAL
        # Speaker name does NOT count — only the .text field is scanned
        transcript = [{"speaker": "NPC", "text": "Norman told me to come here.", "step": 1}]
        result = ev.evaluate(
            milestone_id="DAD_FIRST_MEETING",
            milestone_description="Meet Dad",
            keywords=["Gym Leader", "Norman", "heal", "Pokemon"],
            transcript=transcript,
        )
        # 1 match (Norman) out of 4 keywords; threshold = max(1, 4//2) = 2 → PARTIAL
        assert result == "PARTIAL"

    def test_no_keywords_specified_nonempty_transcript_returns_yes(self):
        ev = TransitionEvaluator(vlm=None)
        transcript = [{"speaker": "NPC", "text": "Something happened.", "step": 1}]
        result = ev.evaluate(
            milestone_id="SOME_MILESTONE",
            milestone_description="Any dialogue",
            keywords=[],
            transcript=transcript,
        )
        assert result == "YES"


class TestTransitionEvaluatorYes:
    """LLM path — mocked VLM returns YES."""

    def test_vlm_yes_response_returns_yes(self):
        mock_vlm = MagicMock()
        mock_vlm.backend.get_text_query.return_value = "YES"
        ev = TransitionEvaluator(vlm=mock_vlm)
        transcript = [{"speaker": "Norman", "text": "I am the Gym Leader.", "step": 1}]
        result = ev.evaluate(
            milestone_id="DAD_FIRST_MEETING",
            milestone_description="Meet Dad",
            keywords=["Gym Leader"],
            transcript=transcript,
        )
        assert result == "YES"


class TestTransitionEvaluatorNo:
    """LLM path — mocked VLM returns NO."""

    def test_vlm_no_response_returns_no(self):
        mock_vlm = MagicMock()
        mock_vlm.backend.get_text_query.return_value = "NO"
        ev = TransitionEvaluator(vlm=mock_vlm)
        transcript = [{"speaker": "NPC", "text": "Nothing relevant.", "step": 1}]
        result = ev.evaluate(
            milestone_id="DAD_FIRST_MEETING",
            milestone_description="Meet Dad",
            keywords=["Gym Leader"],
            transcript=transcript,
        )
        assert result == "NO"

    def test_vlm_failure_falls_back_to_keyword_scan(self):
        mock_vlm = MagicMock()
        mock_vlm.backend.get_text_query.side_effect = RuntimeError("API error")
        ev = TransitionEvaluator(vlm=mock_vlm)
        # Keyword present → keyword scan returns YES despite VLM failure
        transcript = [
            {"speaker": "Norman", "text": "I'm the Gym Leader Norman.", "step": 1}
        ]
        result = ev.evaluate(
            milestone_id="DAD_FIRST_MEETING",
            milestone_description="Meet Dad",
            keywords=["Gym Leader"],
            transcript=transcript,
        )
        assert result in ("YES", "PARTIAL")


# ---------------------------------------------------------------------------
# 5.x: check_storyline_milestones skips dialogue milestones
# ---------------------------------------------------------------------------

class TestCheckStorylineMilestonesSkipsDialogue:
    def test_rom_flag_for_dad_first_meeting_does_not_mark_completed(self):
        """check_storyline_milestones must NOT auto-complete DAD_FIRST_MEETING."""
        obj_manager = _make_obj_manager()
        state_data = {
            "game": {"in_battle": False},
            "player": {"location": "PETALBURG CITY GYM", "position": {"x": 4, "y": 8}},
            "milestones": {"DAD_FIRST_MEETING": {"completed": True}},
            "party": [],
        }
        obj_manager.check_storyline_milestones(state_data)
        # completed_goals must NOT contain DAD_FIRST_MEETING
        assert not obj_manager.completed_goals.get("DAD_FIRST_MEETING", False)

    def test_rom_flag_for_location_milestone_does_mark_completed(self):
        """check_storyline_milestones SHOULD auto-complete location milestones."""
        obj_manager = _make_obj_manager()
        state_data = {
            "game": {"in_battle": False},
            "player": {"location": "PETALBURG CITY", "position": {"x": 0, "y": 0}},
            "milestones": {"PETALBURG_CITY": {"completed": True}},
            "party": [],
        }
        obj_manager.check_storyline_milestones(state_data)
        assert obj_manager.completed_goals.get("PETALBURG_CITY", False)


# ---------------------------------------------------------------------------
# Out-of-bounds guard
# ---------------------------------------------------------------------------

class TestOutOfBoundsIndex:
    def test_out_of_bounds_index_returns_state_unchanged(self):
        obj_manager = _make_obj_manager()
        state = _make_state(milestone_index=9999)
        node = make_verification_node(obj_manager)
        result = node(state)
        assert result["milestone_index"] == 9999
