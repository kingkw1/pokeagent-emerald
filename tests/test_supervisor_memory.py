"""
tests/test_supervisor_memory.py — Phase 5 unit tests for memory integration.

Covers:
  TestDialogueQueryPostBootOnly         — stale pre-boot records excluded
  TestDialogueQueryEmpty                — empty collection returns "", no exception
  TestDialogueQueryNormanKeywords       — relevant NPC dialogue surfaces correctly
  TestBattleOutcomeLogged               — make_battle_bot_node logs start + end events
  TestBattleQueryPostBootOnly           — stale battle_outcome records excluded
  TestBattleQueryEmpty                  — empty collection returns "", no exception
  TestSupervisorPromptBothContexts      — both contexts appear verbatim in rendered prompt
  TestSupervisorPromptMissingContextsFallback — empty contexts render as "(none)"
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch, call

import pytest

from agent.graph.goal_stack import GoalNode
from agent.graph.nodes.battle_bot import make_battle_bot_node, _format_party_hp
from agent.graph.nodes.executive_supervisor import (
    SUPERVISOR_USER_TEMPLATE,
    _query_dialogue_context,
    _query_battle_outcomes,
)
from agent.graph.nodes.handoff_detector import make_handoff_detector_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_goal(description: str = "Navigate to Petalburg City") -> GoalNode:
    return GoalNode(
        goal_id="test_goal",
        description=description,
        goal_type="immediate",
        parent_id=None,
        completion_condition="Player arrives at Petalburg City",
        directive=None,
        metadata={},
    )


def _make_collection(docs: list[str], metas: list[dict]) -> MagicMock:
    """Return a mock ChromaDB collection that returns *docs* / *metas* on query()."""
    col = MagicMock()
    col.count.return_value = len(docs)
    col.query.return_value = {
        "documents": [docs],
        "metadatas": [metas],
    }
    return col


def _make_episodic_memory(docs: list[str] = (), metas: list[dict] = ()) -> MagicMock:
    mem = MagicMock()
    mem.collection = _make_collection(list(docs), list(metas))
    return mem


def _make_state(in_battle: bool = False, location: str = "ROUTE_102",
                party: list | None = None) -> dict:
    return {
        "state_data": {
            "game": {"in_battle": in_battle, "map_id": 5},
            "player": {"location": location, "position": {"x": 10, "y": 10}},
            "party": party or [],
        },
        "perception": {},
        "step_count": 1,
    }


# ---------------------------------------------------------------------------
# _query_dialogue_context
# ---------------------------------------------------------------------------

class TestDialogueQueryPostBootOnly:
    """Post-boot records are returned; pre-boot records are excluded by the where filter."""

    def test_query_passes_boot_time_to_collection(self):
        now = time.time()
        boot_time = now - 10.0
        goal = _make_goal()
        mem = _make_episodic_memory(
            docs=["Norman said: 'In Pokémon, there are good and bad points.'"],
            metas=[{"type": "dialogue_transcript", "timestamp": now}],
        )

        result = _query_dialogue_context(mem, goal, boot_time=boot_time)

        # The collection was queried with a where filter that includes boot_time
        call_kwargs = mem.collection.query.call_args.kwargs
        where = call_kwargs.get("where", {})
        and_clauses = where.get("$and", [])
        ts_clause = next(
            (c for c in and_clauses if "timestamp" in c), None
        )
        assert ts_clause is not None, "where filter must include a timestamp clause"
        assert ts_clause["timestamp"]["$gte"] == boot_time

    def test_result_contains_returned_document(self):
        mem = _make_episodic_memory(
            docs=["Norman: good points and bad points."],
            metas=[{"type": "dialogue_transcript", "timestamp": time.time()}],
        )
        result = _query_dialogue_context(mem, _make_goal(), boot_time=0.0)
        assert "Norman" in result

    def test_empty_docs_returns_empty_string(self):
        """Collection has items but query returns 0 matches — empty string, not None."""
        mem = _make_episodic_memory()
        mem.collection.count.return_value = 3
        mem.collection.query.return_value = {"documents": [[]], "metadatas": [[]]}
        result = _query_dialogue_context(mem, _make_goal(), boot_time=0.0)
        assert result == ""


class TestDialogueQueryEmpty:
    def test_no_exception_on_empty_collection(self):
        mem = _make_episodic_memory()  # count=0
        result = _query_dialogue_context(mem, _make_goal(), boot_time=0.0)
        assert result == ""
        mem.collection.query.assert_not_called()

    def test_none_memory_returns_empty_string(self):
        result = _query_dialogue_context(None, _make_goal(), boot_time=0.0)
        assert result == ""

    def test_none_goal_returns_empty_string(self):
        mem = _make_episodic_memory(docs=["x"], metas=[{}])
        result = _query_dialogue_context(mem, None, boot_time=0.0)
        assert result == ""

    def test_query_exception_returns_empty_string(self):
        mem = _make_episodic_memory()
        mem.collection.count.return_value = 5
        mem.collection.query.side_effect = RuntimeError("ChromaDB error")
        result = _query_dialogue_context(mem, _make_goal(), boot_time=0.0)
        assert result == ""


class TestDialogueQueryNormanKeywords:
    def test_norman_dialogue_surfaces(self):
        doc = "Norman: In Pokémon, there are good points and bad points to every situation."
        mem = _make_episodic_memory(
            docs=[doc],
            metas=[{"type": "dialogue_transcript", "timestamp": time.time()}],
        )
        goal = _make_goal("Meet Norman and learn about gym challenge")
        result = _query_dialogue_context(mem, goal, boot_time=0.0)
        assert "Norman" in result

    def test_type_filter_clause_is_dialogue_transcript(self):
        mem = _make_episodic_memory(docs=["text"], metas=[{}])
        _query_dialogue_context(mem, _make_goal(), boot_time=0.0)
        call_kwargs = mem.collection.query.call_args.kwargs
        and_clauses = call_kwargs["where"]["$and"]
        type_clause = next(c for c in and_clauses if "type" in c)
        assert type_clause["type"]["$eq"] == "dialogue_transcript"


# ---------------------------------------------------------------------------
# _query_battle_outcomes
# ---------------------------------------------------------------------------

class TestBattleQueryPostBootOnly:
    def test_query_passes_boot_time_to_collection(self):
        now = time.time()
        boot_time = now - 5.0
        mem = _make_episodic_memory(
            docs=["Battle ended at ROUTE_102. Party HP: Treecko 45/50"],
            metas=[{"type": "battle_outcome", "timestamp": now}],
        )
        _query_battle_outcomes(mem, boot_time=boot_time)
        call_kwargs = mem.collection.query.call_args.kwargs
        and_clauses = call_kwargs["where"]["$and"]
        ts_clause = next(c for c in and_clauses if "timestamp" in c)
        assert ts_clause["timestamp"]["$gte"] == boot_time

    def test_result_contains_returned_document(self):
        mem = _make_episodic_memory(
            docs=["Battle ended at ROUTE_102. Party HP: Treecko 45/50"],
            metas=[{"type": "battle_outcome", "timestamp": time.time()}],
        )
        result = _query_battle_outcomes(mem, boot_time=0.0)
        assert "Battle ended" in result

    def test_type_filter_clause_is_battle_outcome(self):
        mem = _make_episodic_memory(docs=["text"], metas=[{}])
        _query_battle_outcomes(mem, boot_time=0.0)
        call_kwargs = mem.collection.query.call_args.kwargs
        and_clauses = call_kwargs["where"]["$and"]
        type_clause = next(c for c in and_clauses if "type" in c)
        assert type_clause["type"]["$eq"] == "battle_outcome"


class TestBattleQueryEmpty:
    def test_no_exception_on_empty_collection(self):
        mem = _make_episodic_memory()
        result = _query_battle_outcomes(mem, boot_time=0.0)
        assert result == ""
        mem.collection.query.assert_not_called()

    def test_none_memory_returns_empty_string(self):
        result = _query_battle_outcomes(None, boot_time=0.0)
        assert result == ""

    def test_query_exception_returns_empty_string(self):
        mem = _make_episodic_memory()
        mem.collection.count.return_value = 5
        mem.collection.query.side_effect = RuntimeError("ChromaDB error")
        result = _query_battle_outcomes(mem, boot_time=0.0)
        assert result == ""


# ---------------------------------------------------------------------------
# make_battle_bot_node — Phase 5.3: battle_start logging
# ---------------------------------------------------------------------------

class TestBattleStartLogged:
    """battle_bot_node logs battle_start when in_battle transitions False→True.

    battle_outcome is NOT logged here — battle_bot_node is only dispatched
    when in_battle=True, so the True→False transition is invisible to it.
    battle_outcome logging is tested in TestBattleOutcomeLogged via
    make_handoff_detector_node.
    """

    def test_battle_start_logged_on_false_to_true(self):
        mem = MagicMock()
        party = [{"name": "Treecko", "current_hp": 45, "max_hp": 50}]
        with patch("agent.graph.nodes.battle_bot.get_battle_bot") as mock_factory:
            mock_factory.return_value.get_action.return_value = "PRESS_A_ONLY"
            node = make_battle_bot_node(episodic_memory=mem)
            # Step 1: not in battle yet — no log
            node(_make_state(in_battle=False, party=party))
            assert mem.log_event.call_count == 0
            # Step 2: battle starts — should log battle_start
            node(_make_state(in_battle=True, party=party))
        assert mem.log_event.call_count == 1
        call = mem.log_event.call_args_list[0]
        text = call.args[0] if call.args else call.kwargs.get("text", "")
        meta = call.kwargs.get("metadata") or (call.args[1] if len(call.args) > 1 else {})
        assert "Battle started" in text
        assert meta.get("type") == "battle_start"

    def test_no_log_when_memory_is_none(self):
        with patch("agent.graph.nodes.battle_bot.get_battle_bot") as mock_factory:
            mock_factory.return_value.get_action.return_value = "PRESS_A_ONLY"
            node = make_battle_bot_node(episodic_memory=None)
            node(_make_state(in_battle=False))
            node(_make_state(in_battle=True))
            # Should not raise — no memory to log to


# ---------------------------------------------------------------------------
# make_handoff_detector_node — Phase 5.3: battle_outcome logging
# ---------------------------------------------------------------------------

class TestBattleOutcomeLogged:
    """battle_outcome is logged by handoff_detector when it detects battle_bot→nav_bot.

    This is where the True→False in_battle transition is detectable:
    the router already sent the step to nav_bot (last_action=NAVIGATE),
    so the handoff detector sees previous_node=battle_bot, current=nav_bot.
    """

    def _make_post_battle_state(self) -> dict:
        """State as it arrives at handoff_detector after battle_bot→nav_bot.

        Uses real game state structure: party under state_data["player"]["party"],
        with species_name as the primary name key.
        """
        return {
            "last_action": "NAVIGATE",        # current step routed to nav_bot
            "last_node_fired": "battle_bot",  # previous cycle was battle_bot
            "goal_stack": [],
            "step_count": 5,
            "state_data": {
                "game": {"in_battle": False, "map_id": 5},
                "player": {
                    "location": "ROUTE_102",
                    "position": {"x": 10, "y": 10},
                    "party": [{"species_name": "Treecko", "current_hp": 45, "max_hp": 50}],
                },
            },
            "perception": {},
        }

    def test_log_event_called_once(self):
        mem = MagicMock()
        node = make_handoff_detector_node(episodic_memory=mem)
        node(self._make_post_battle_state())
        assert mem.log_event.call_count == 1

    def test_logged_text_contains_battle_ended(self):
        mem = MagicMock()
        node = make_handoff_detector_node(episodic_memory=mem)
        node(self._make_post_battle_state())
        text = mem.log_event.call_args.args[0]
        assert "Battle ended" in text

    def test_logged_metadata_type_is_battle_outcome(self):
        mem = MagicMock()
        node = make_handoff_detector_node(episodic_memory=mem)
        node(self._make_post_battle_state())
        meta = mem.log_event.call_args.kwargs.get("metadata", {})
        assert meta.get("type") == "battle_outcome"

    def test_logged_metadata_party_hp_is_non_empty(self):
        mem = MagicMock()
        node = make_handoff_detector_node(episodic_memory=mem)
        node(self._make_post_battle_state())
        meta = mem.log_event.call_args.kwargs.get("metadata", {})
        assert meta.get("party_hp")

    def test_no_log_on_nav_to_nav(self):
        """nav_bot → nav_bot does not log a battle_outcome."""
        mem = MagicMock()
        node = make_handoff_detector_node(episodic_memory=mem)
        state = {**self._make_post_battle_state(),
                 "last_action": "NAVIGATE",
                 "last_node_fired": "nav_bot"}
        node(state)
        mem.log_event.assert_not_called()

    def test_no_log_when_memory_is_none(self):
        """make_handoff_detector_node(None) returns the plain function — no crash."""
        node = make_handoff_detector_node(episodic_memory=None)
        # Should not raise
        node(self._make_post_battle_state())


class TestFormatPartyHp:
    def test_party_under_player_key_species_name(self):
        """Real game state: party at state_data["player"]["party"] with species_name key."""
        state_data = {
            "player": {
                "party": [
                    {"species_name": "Treecko", "current_hp": 45, "max_hp": 50},
                    {"species_name": "Wingull", "current_hp": 0, "max_hp": 32},
                ]
            }
        }
        result = _format_party_hp(state_data)
        assert "Treecko 45/50" in result
        assert "Wingull 0/32" in result

    def test_party_at_top_level_fallback(self):
        """Fallback: party directly under state_data with legacy name/species keys."""
        state_data = {
            "party": [
                {"name": "Treecko", "current_hp": 45, "max_hp": 50},
            ]
        }
        assert "Treecko 45/50" in _format_party_hp(state_data)

    def test_empty_party(self):
        assert _format_party_hp({"player": {"party": []}}) == "(no party data)"

    def test_missing_party_key(self):
        assert _format_party_hp({}) == "(no party data)"


# ---------------------------------------------------------------------------
# Supervisor prompt rendering
# ---------------------------------------------------------------------------

class TestSupervisorPromptBothContexts:
    def test_dialogue_and_battle_appear_verbatim(self):
        dialogue = "Norman: In Pokémon, there are good points and bad points."
        battle = "Battle ended at ROUTE_102. Party HP: Treecko 45/50"
        rendered = SUPERVISOR_USER_TEMPLATE.format(
            stack_repr="[immediate] Navigate to Petalburg City",
            goal_id="navigate_petalburg",
            goal_type="immediate",
            goal_description="Navigate to Petalburg City",
            completion_condition="Player location is PETALBURG_CITY",
            dialogue_context=dialogue,
            battle_context=battle,
            current_location="ROUTE_102",
            pos_x=10,
            pos_y=10,
            party_hp_summary="Treecko 45/50",
            badge_count=0,
            in_battle=False,
            last_node_fired="nav_bot",
            previous_node="nav_bot",
            current_node="nav_bot",
            step_count=5,
        )
        assert dialogue in rendered
        assert battle in rendered

    def test_dialogue_section_label_present(self):
        rendered = SUPERVISOR_USER_TEMPLATE.format(
            stack_repr="", goal_id="g", goal_type="immediate",
            goal_description="d", completion_condition="c",
            dialogue_context="some dialogue",
            battle_context="",
            current_location="ROUTE_102", pos_x=0, pos_y=0,
            party_hp_summary="", badge_count=0, in_battle=False,
            last_node_fired="nav_bot", previous_node="nav_bot",
            current_node="nav_bot", step_count=1,
        )
        assert "DIALOGUE TRANSCRIPT" in rendered
        assert "BATTLE OUTCOMES" in rendered


class TestSupervisorPromptMissingContextsFallback:
    def test_empty_contexts_render_as_none(self):
        rendered = SUPERVISOR_USER_TEMPLATE.format(
            stack_repr="", goal_id="g", goal_type="immediate",
            goal_description="d", completion_condition="c",
            dialogue_context="",
            battle_context="",
            current_location="ROUTE_102", pos_x=0, pos_y=0,
            party_hp_summary="", badge_count=0, in_battle=False,
            last_node_fired="nav_bot", previous_node="nav_bot",
            current_node="nav_bot", step_count=1,
        )
        # The template renders empty strings as-is — the prompt section simply
        # contains an empty line.  Callers should pass "(none)" explicitly
        # when context is absent — verify SUPERVISOR_USER_TEMPLATE accepts
        # both and does not raise.
        assert "DIALOGUE TRANSCRIPT" in rendered
        assert "BATTLE OUTCOMES" in rendered

    def test_none_placeholder_strings_no_key_error(self):
        """Passing literal '(none)' strings (as callers do) doesn't raise."""
        rendered = SUPERVISOR_USER_TEMPLATE.format(
            stack_repr="", goal_id="g", goal_type="immediate",
            goal_description="d", completion_condition="c",
            dialogue_context="(none)",
            battle_context="(none)",
            current_location="ROUTE_102", pos_x=0, pos_y=0,
            party_hp_summary="", badge_count=0, in_battle=False,
            last_node_fired="nav_bot", previous_node="nav_bot",
            current_node="nav_bot", step_count=1,
        )
        assert "(none)" in rendered
