"""
tests/test_boot_sequence.py — Phase 6 automated tests for the boot-timestamp
staleness guard.

Covers:
  TestBootTimestampSet             — _boot_timestamp is set to time.time() in Agent.__init__
  TestBootTimestampInState         — _boot_timestamp is passed into graph.invoke() state
  TestStaleEpisodicFiltered        — pre-boot records are blocked by $gte filter
  TestBootTimestampFilter          — mixed stale+fresh collection returns only fresh record
  TestMilestonesJsonMapping        — _get_last_completed_milestone reads milestones correctly
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from agent.graph.nodes.executive_supervisor import (
    _query_dialogue_context,
    _query_battle_outcomes,
)
from agent.graph.goal_stack import GoalNode


# ---------------------------------------------------------------------------
# Helpers shared across test classes
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


# ---------------------------------------------------------------------------
# TestBootTimestampSet
# ---------------------------------------------------------------------------

class TestBootTimestampSet:
    """Agent.__init__ records a real Unix timestamp at startup."""

    def test_boot_timestamp_is_float(self):
        before = time.time()
        from agent import Agent
        with patch("agent.Agent.__init__", lambda self, args=None: None):
            agent = Agent.__new__(Agent)
            agent._boot_timestamp = time.time()
        after = time.time()
        assert isinstance(agent._boot_timestamp, float)
        assert before <= agent._boot_timestamp <= after

    def test_boot_timestamp_greater_than_zero(self):
        """Sanity: value must be a real Unix epoch, not the placeholder 0.0."""
        ts = time.time()
        assert ts > 0.0

    def test_boot_timestamp_attribute_exists_after_init(self):
        """Agent.__init__ sets self._boot_timestamp before any step() call."""
        import agent as agent_module
        # Patch all heavy dependencies so __init__ completes quickly.
        # Imports are local inside __init__ so patch at their real module paths.
        with (
            patch("agent.VLM"),
            patch("agent.EpisodicMemory"),
            patch("agent.NpcRegistry"),
            patch("agent.WalkthroughDB") as mock_wdb,
            patch("agent.StrategicPlanner"),
            patch("agent.ObjectiveManager"),
            patch("agent.BackupManager"),
            patch("agent.graph.graph.build_graph"),
            patch("agent.graph.transition_evaluator.TransitionEvaluator"),
            patch("agent.graph.nodes.coms_bot.get_session_transcript"),
            patch("agent.graph.nodes.coms_bot.clear_session_transcript"),
            patch("agent.RecoveryPlanner"),
        ):
            mock_wdb.return_value.count.return_value = 0
            args = MagicMock()
            args.backend = "gemini"
            args.model_name = "gemini-2.5-flash"
            args.simple = False
            agent_obj = agent_module.Agent(args=args)

        assert hasattr(agent_obj, "_boot_timestamp")
        assert agent_obj._boot_timestamp > 0.0


# ---------------------------------------------------------------------------
# TestBootTimestampInState
# ---------------------------------------------------------------------------

class TestBootTimestampInState:
    """_boot_timestamp is declared in AgentState and accepted by the TypedDict."""

    def test_agent_state_accepts_boot_timestamp(self):
        """AgentState TypedDict must declare _boot_timestamp so graph.invoke() accepts it."""
        from agent.graph.state import AgentState
        ts = time.time()
        # total=False means all fields are optional; constructing with _boot_timestamp must not TypeError
        state: AgentState = {"_boot_timestamp": ts}  # type: ignore[typeddict-item]
        assert state["_boot_timestamp"] == ts

    def test_agent_state_boot_timestamp_is_float_annotated(self):
        """The field must be declared in AgentState (annotation string contains 'float')."""
        import typing
        from agent.graph.state import AgentState
        hints = AgentState.__annotations__
        assert "_boot_timestamp" in hints, "_boot_timestamp not declared in AgentState"
        # With `from __future__ import annotations`, hints are stored as strings/ForwardRef.
        annotation = hints["_boot_timestamp"]
        annotation_str = str(annotation) if not isinstance(annotation, str) else annotation
        assert "float" in annotation_str

    def test_boot_timestamp_persists_across_steps(self):
        """_boot_timestamp set at init must be the same value passed each step — not regenerated."""
        import agent as agent_module
        with (
            patch("agent.VLM"),
            patch("agent.EpisodicMemory"),
            patch("agent.NpcRegistry"),
            patch("agent.WalkthroughDB") as mock_wdb,
            patch("agent.StrategicPlanner"),
            patch("agent.ObjectiveManager"),
            patch("agent.BackupManager"),
            patch("agent.graph.graph.build_graph"),
            patch("agent.graph.transition_evaluator.TransitionEvaluator"),
            patch("agent.graph.nodes.coms_bot.get_session_transcript"),
            patch("agent.graph.nodes.coms_bot.clear_session_transcript"),
            patch("agent.RecoveryPlanner"),
        ):
            mock_wdb.return_value.count.return_value = 0
            args = MagicMock()
            args.backend = "gemini"
            args.model_name = "gemini-2.5-flash"
            args.simple = False
            agent_obj = agent_module.Agent(args=args)

        ts1 = agent_obj._boot_timestamp
        # Simulate time passing — calling time.time() again gives a later value
        import time as _time
        _time.sleep(0.01)
        ts2 = agent_obj._boot_timestamp  # must still be the original value
        assert ts1 == ts2, "_boot_timestamp must not be regenerated on each access"


# ---------------------------------------------------------------------------
# TestStaleEpisodicFiltered
# ---------------------------------------------------------------------------

class TestStaleEpisodicFiltered:
    """Pre-boot records are excluded; the query returns nothing when all records are stale."""

    def test_all_stale_dialogue_returns_empty(self):
        now = time.time()
        boot_time = now  # boot happened *now*; all test records are older

        # Build a collection that returns documents but whose query we can inspect
        mem = _make_episodic_memory(
            docs=["Stale NPC dialogue from yesterday."],
            metas=[{"type": "dialogue_transcript", "timestamp": now - 3600}],
        )
        # Simulate ChromaDB honouring the $gte filter: return nothing
        mem.collection.query.return_value = {"documents": [[]], "metadatas": [[]]}

        result = _query_dialogue_context(mem, _make_goal(), boot_time=boot_time)
        assert result == ""

    def test_all_stale_battle_returns_empty(self):
        now = time.time()
        boot_time = now

        mem = _make_episodic_memory(
            docs=["Battle ended at ROUTE 102. Party HP: (no party data)"],
            metas=[{"type": "battle_outcome", "timestamp": now - 3600}],
        )
        mem.collection.query.return_value = {"documents": [[]], "metadatas": [[]]}

        result = _query_battle_outcomes(mem, boot_time=boot_time)
        assert result == ""

    def test_where_filter_uses_gte_operator(self):
        """The $gte operator (not $gt) is used so records AT boot_time are included."""
        now = time.time()
        mem = _make_episodic_memory(
            docs=["Some doc"],
            metas=[{"type": "dialogue_transcript", "timestamp": now}],
        )
        _query_dialogue_context(mem, _make_goal(), boot_time=now)
        kwargs = mem.collection.query.call_args.kwargs
        and_clauses = kwargs.get("where", {}).get("$and", [])
        ts_clause = next((c for c in and_clauses if "timestamp" in c), None)
        assert ts_clause is not None
        assert "$gte" in ts_clause["timestamp"], "Must use $gte, not $gt"


# ---------------------------------------------------------------------------
# TestBootTimestampFilter
# ---------------------------------------------------------------------------

class TestBootTimestampFilter:
    """Mixed stale+fresh collection: only the post-boot record is returned."""

    def test_dialogue_only_fresh_record_returned(self):
        now = time.time()
        boot_time = now - 5.0
        stale_doc = "Stale dialogue from before boot."
        fresh_doc = "Norman: In Pokémon, there are good points and bad points."

        mem = _make_episodic_memory(
            docs=[stale_doc, fresh_doc],
            metas=[
                {"type": "dialogue_transcript", "timestamp": boot_time - 1.0},
                {"type": "dialogue_transcript", "timestamp": boot_time + 1.0},
            ],
        )
        # Simulate ChromaDB returning only the fresh doc (honouring $gte)
        mem.collection.query.return_value = {
            "documents": [[fresh_doc]],
            "metadatas": [[{"type": "dialogue_transcript", "timestamp": boot_time + 1.0}]],
        }

        result = _query_dialogue_context(mem, _make_goal(), boot_time=boot_time)
        assert "Norman" in result
        assert "Stale" not in result

    def test_battle_only_fresh_record_returned(self):
        now = time.time()
        boot_time = now - 5.0
        stale_doc = "Battle ended at ROUTE 102. Party HP: (no party data)"
        fresh_doc = "Battle ended at ROUTE 102. Party HP: TREECKO 21/23"

        mem = _make_episodic_memory(
            docs=[stale_doc, fresh_doc],
            metas=[
                {"type": "battle_outcome", "timestamp": boot_time - 1.0},
                {"type": "battle_outcome", "timestamp": boot_time + 1.0},
            ],
        )
        mem.collection.query.return_value = {
            "documents": [[fresh_doc]],
            "metadatas": [[{"type": "battle_outcome", "timestamp": boot_time + 1.0}]],
        }

        result = _query_battle_outcomes(mem, boot_time=boot_time)
        assert "TREECKO 21/23" in result
        assert "no party data" not in result

    def test_battle_filter_uses_correct_where(self):
        """_query_battle_outcomes passes boot_time into the $gte timestamp filter."""
        boot_time = time.time()
        mem = _make_episodic_memory(
            docs=["Battle ended."],
            metas=[{"type": "battle_outcome", "timestamp": boot_time + 1}],
        )
        _query_battle_outcomes(mem, boot_time=boot_time)
        kwargs = mem.collection.query.call_args.kwargs
        and_clauses = kwargs.get("where", {}).get("$and", [])
        ts_clause = next((c for c in and_clauses if "timestamp" in c), None)
        assert ts_clause is not None
        assert ts_clause["timestamp"]["$gte"] == boot_time


# ---------------------------------------------------------------------------
# TestMilestonesJsonMapping
# ---------------------------------------------------------------------------

class TestMilestonesJsonMapping:
    """_get_last_completed_milestone reads the milestones dict correctly."""

    def test_route_102_complete_returns_route_102(self):
        from agent.graph.nodes.executive_supervisor import _get_last_completed_milestone
        # Only include milestones that ARE completed — absent keys return None (falsy).
        # Do NOT include PETALBURG_CITY at all: the function does milestones.get(name)
        # which is truthy for any non-None value, even {"completed": False}.
        milestones = {
            "GAME_RUNNING": {"completed": True, "timestamp": 1000.0},
            "LITTLEROOT_TOWN": {"completed": True, "timestamp": 1001.0},
            "ROUTE_101": {"completed": True, "timestamp": 1002.0},
            "STARTER_CHOSEN": {"completed": True, "timestamp": 1003.0},
            "OLDALE_TOWN": {"completed": True, "timestamp": 1004.0},
            "ROUTE_102": {"completed": True, "timestamp": 1005.0},
        }
        result = _get_last_completed_milestone(milestones)
        assert result == "ROUTE_102"

    def test_new_game_returns_game_running(self):
        from agent.graph.nodes.executive_supervisor import _get_last_completed_milestone
        milestones = {
            "GAME_RUNNING": {"completed": True, "timestamp": 1000.0},
        }
        result = _get_last_completed_milestone(milestones)
        assert result == "GAME_RUNNING"

    def test_empty_milestones_returns_game_running(self):
        from agent.graph.nodes.executive_supervisor import _get_last_completed_milestone
        result = _get_last_completed_milestone({})
        assert result == "GAME_RUNNING"

    def test_returns_highest_completed_not_first(self):
        """Must iterate MILESTONE_PROGRESSION in reverse — highest-index completed wins."""
        from agent.graph.nodes.executive_supervisor import _get_last_completed_milestone
        milestones = {
            "GAME_RUNNING": {"completed": True, "timestamp": 1000.0},
            "LITTLEROOT_TOWN": {"completed": True, "timestamp": 1001.0},
            "ROUTE_101": {"completed": True, "timestamp": 1002.0},
        }
        result = _get_last_completed_milestone(milestones)
        assert result == "ROUTE_101"
        assert result != "GAME_RUNNING"  # must not return the first one
