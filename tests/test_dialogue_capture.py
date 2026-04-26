"""
tests/test_dialogue_capture.py — Phase 5.3 automated tests.

Verifies that coms_bot_node (via make_coms_bot_node factory):
  * Logs dialogue turns to ChromaDB when episodic_memory is provided.
  * Waits for script-idle (mode 0) before extracting text.
  * Assembles a session transcript across multiple turns.
  * Sets dialogue_completed flag management (session transcript populated).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

from agent.graph.nodes.coms_bot import (
    clear_session_transcript,
    get_session_transcript,
    make_coms_bot_node,
    _SKIP_SCRIPT_IDLE_LOCATIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides):
    base = {
        "milestone_index": 17,  # DAD_FIRST_MEETING index
        "state_data": {
            "game": {
                "in_dialog": True,
                "game_state": "dialog",
                "script_mode": 0,  # animation complete
                "map_id": 2049,
            },
            "player": {
                "location": "PETALBURG CITY GYM",
                "position": {"x": 4, "y": 8},
            },
            "milestones": {},
            "party": [],
        },
        "perception": {"visual_data": {}},
        "context": "dialogue",
        "dialogue_completed": False,
        "dialogue_transcript": [],
        "goal_coords": None,
        "goal_location": None,
        "npc_coords": None,
        "should_interact": False,
        "last_action": None,
        "last_buttons": [],
        "step_count": 42,
        "reward": None,
        "prev_state_snapshot": None,
        "telemetry": None,
        "frame": MagicMock(),  # fake frame image
    }
    base.update(overrides)
    return base


def _make_vlm_returning(text: str, speaker: str = "Norman", has_more: bool = True):
    """Return a mock VLM that yields a structured dialogue turn."""
    vlm = MagicMock()
    vlm.get_query.return_value = json.dumps(
        {"speaker": speaker, "text": text, "has_more": has_more}
    )
    return vlm


# ---------------------------------------------------------------------------
# Fixture: clear session transcript before each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_transcript():
    clear_session_transcript()
    yield
    clear_session_transcript()


# ---------------------------------------------------------------------------
# TestCaptureLogsToChromaDB
# ---------------------------------------------------------------------------

class TestCaptureLogsToChromaDB:
    def test_chromadb_write_called_with_dialogue_transcript_type(self):
        """coms_bot_node logs turn to episodic_memory with type='dialogue_transcript'."""
        vlm = _make_vlm_returning("Hello trainer!", speaker="Norman")
        mem = MagicMock()

        node = make_coms_bot_node(vlm=vlm, episodic_memory=mem)
        state = _make_state()
        node(state)

        mem.log_event.assert_called_once()
        call_kwargs = mem.log_event.call_args
        # First positional arg is the text
        logged_text = call_kwargs[0][0]
        assert "Hello trainer!" in logged_text

        # Metadata must include required fields
        metadata = call_kwargs[1].get("metadata") or call_kwargs[0][1]
        assert metadata.get("type") == "dialogue_transcript"
        assert "speaker" in metadata
        assert "step" in metadata
        assert "milestone" in metadata

    def test_chromadb_not_called_when_no_memory(self):
        """No ChromaDB call when episodic_memory is None."""
        vlm = _make_vlm_returning("Test dialogue")
        node = make_coms_bot_node(vlm=vlm, episodic_memory=None)
        state = _make_state()
        # Should not raise
        node(state)

    def test_document_contains_speaker_and_text(self):
        """Logged document contains speaker name."""
        vlm = _make_vlm_returning("I am the Gym Leader.", speaker="Norman")
        mem = MagicMock()

        node = make_coms_bot_node(vlm=vlm, episodic_memory=mem)
        state = _make_state()
        node(state)

        logged_text = mem.log_event.call_args[0][0]
        assert "Norman" in logged_text
        assert "Gym Leader" in logged_text


# ---------------------------------------------------------------------------
# TestCaptureWaitsForIdle
# ---------------------------------------------------------------------------

class TestCaptureWaitsForIdle:
    def test_no_capture_when_script_mode_1(self):
        """script_mode=1 (animation running) → VLM not called for capture."""
        vlm = MagicMock()
        node = make_coms_bot_node(vlm=vlm, episodic_memory=None)
        state = _make_state()
        state["state_data"]["game"]["script_mode"] = 1

        node(state)

        # VLM.get_query should NOT have been called (capture skipped)
        vlm.get_query.assert_not_called()

    def test_no_capture_when_script_mode_2(self):
        """script_mode=2 (native callback) → VLM not called for capture."""
        vlm = MagicMock()
        node = make_coms_bot_node(vlm=vlm, episodic_memory=None)
        state = _make_state()
        state["state_data"]["game"]["script_mode"] = 2

        node(state)

        vlm.get_query.assert_not_called()

    def test_capture_when_script_mode_0(self):
        """script_mode=0 (text fully rendered) → VLM IS called."""
        vlm = _make_vlm_returning("Text is ready.")
        node = make_coms_bot_node(vlm=vlm, episodic_memory=None)
        state = _make_state()
        state["state_data"]["game"]["script_mode"] = 0

        node(state)

        vlm.get_query.assert_called_once()

    def test_capture_skipped_in_intro_locations(self):
        """Capture skipped when location is in _SKIP_SCRIPT_IDLE_LOCATIONS."""
        vlm = MagicMock()
        node = make_coms_bot_node(vlm=vlm, episodic_memory=None)

        for loc in _SKIP_SCRIPT_IDLE_LOCATIONS:
            vlm.reset_mock()
            state = _make_state()
            state["state_data"]["player"]["location"] = loc
            state["state_data"]["game"]["script_mode"] = 0
            node(state)
            vlm.get_query.assert_not_called()

    def test_a_button_pressed_even_when_capture_skipped(self):
        """When script_mode=1, capture is skipped but A is still pressed."""
        vlm = MagicMock()
        with patch("agent.graph.nodes.coms_bot.get_opener_bot") as mock_opener, \
             patch("agent.graph.nodes.coms_bot.wait_for_script_idle"):
            opener = MagicMock()
            opener.should_handle.return_value = False
            mock_opener.return_value = opener

            node = make_coms_bot_node(vlm=vlm, episodic_memory=None)
            state = _make_state()
            state["state_data"]["game"]["script_mode"] = 1

            result = node(state)

        assert result["last_action"] == "DIALOGUE"
        assert "A" in result["last_buttons"]


# ---------------------------------------------------------------------------
# TestSessionTranscriptAssembled
# ---------------------------------------------------------------------------

class TestSessionTranscriptAssembled:
    def test_three_turns_accumulate_in_transcript(self):
        """Three node calls accumulate 3 entries in the session transcript."""
        vlm = MagicMock()
        texts = ["Hello there!", "I'm Norman.", "Goodbye!"]
        vlm.get_query.side_effect = [
            json.dumps({"speaker": "Norman", "text": t, "has_more": True})
            for t in texts
        ]

        node = make_coms_bot_node(vlm=vlm, episodic_memory=None)
        state = _make_state()
        state["state_data"]["game"]["script_mode"] = 0

        for i in range(3):
            s = dict(state)
            s["step_count"] = i
            node(s)

        transcript = get_session_transcript()
        assert len(transcript) == 3
        assert transcript[0]["text"] == "Hello there!"
        assert transcript[1]["text"] == "I'm Norman."
        assert transcript[2]["text"] == "Goodbye!"

    def test_transcript_entries_contain_required_fields(self):
        """Each transcript entry has speaker, text, and step fields."""
        vlm = _make_vlm_returning("Some dialogue", speaker="NPC")
        node = make_coms_bot_node(vlm=vlm, episodic_memory=None)
        state = _make_state()
        state["state_data"]["game"]["script_mode"] = 0

        node(state)

        transcript = get_session_transcript()
        assert len(transcript) == 1
        entry = transcript[0]
        assert "speaker" in entry
        assert "text" in entry
        assert "step" in entry

    def test_clear_transcript_empties_list(self):
        """clear_session_transcript resets the list to empty."""
        vlm = _make_vlm_returning("Some text")
        node = make_coms_bot_node(vlm=vlm, episodic_memory=None)
        state = _make_state()
        state["state_data"]["game"]["script_mode"] = 0
        node(state)

        assert len(get_session_transcript()) == 1

        clear_session_transcript()
        assert get_session_transcript() == []

    def test_empty_vlm_response_not_appended(self):
        """A VLM response with empty text is not added to the transcript."""
        vlm = MagicMock()
        vlm.get_query.return_value = json.dumps(
            {"speaker": "NPC", "text": "   ", "has_more": False}
        )
        node = make_coms_bot_node(vlm=vlm, episodic_memory=None)
        state = _make_state()
        state["state_data"]["game"]["script_mode"] = 0

        node(state)

        assert get_session_transcript() == []

    def test_no_vlm_produces_no_transcript(self):
        """Without a VLM, no transcript entries are produced."""
        node = make_coms_bot_node(vlm=None, episodic_memory=None)
        state = _make_state()
        node(state)
        assert get_session_transcript() == []


# ---------------------------------------------------------------------------
# TestDialogueSessionCompletedFlag
# ---------------------------------------------------------------------------

class TestDialogueSessionCompletedFlag:
    def test_dialogue_completed_not_set_by_coms_bot(self):
        """coms_bot_node itself does not set dialogue_completed=True.
        That is Agent.step()'s responsibility on the dialogue→nav transition."""
        vlm = _make_vlm_returning("Some text", has_more=False)
        node = make_coms_bot_node(vlm=vlm, episodic_memory=None)
        state = _make_state()
        state["state_data"]["game"]["script_mode"] = 0

        result = node(state)

        # The node should not set dialogue_completed
        assert not result.get("dialogue_completed")

    def test_node_returns_dialogue_action(self):
        """Node always returns last_action='DIALOGUE'."""
        node = make_coms_bot_node(vlm=None, episodic_memory=None)
        state = _make_state()
        result = node(state)
        assert result["last_action"] == "DIALOGUE"
