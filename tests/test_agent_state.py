"""
Tests for Phase 1 — AgentState schema, RewardVector, and TelemetrySnapshot.

Covers:
  TestAgentStateSchema         — TypedDict instantiation and key contract
  TestRewardVectorDelta        — compute_delta() correctness
  TestRewardVectorTotal        — weighted total property
  TestRewardVectorSerialization — to_dict() / JSON round-trip
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from agent.graph.state import AgentState, RewardVector, TelemetrySnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> AgentState:
    """Return a minimal valid AgentState dict."""
    base: AgentState = {
        "frame": None,
        "state_data": {},
        "perception": {},
        "goal_coords": None,
        "goal_location": None,
        "npc_coords": None,
        "should_interact": False,
        "milestone_index": 0,
        "context": "navigation",
        "reward": None,
        "prev_state_snapshot": None,
        "last_action": None,
        "last_buttons": [],
        "step_count": 0,
        "telemetry": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TestAgentStateSchema
# ---------------------------------------------------------------------------


class TestAgentStateSchema:
    def test_instantiate_all_required_fields(self):
        """Constructing a dict with all AgentState keys raises no error."""
        state = _make_state()
        assert isinstance(state, dict)

    def test_keys_match_expected_set(self):
        """All expected top-level keys are present."""
        state = _make_state()
        expected = {
            "frame", "state_data", "perception",
            "goal_coords", "goal_location", "npc_coords", "should_interact",
            "milestone_index", "context",
            "reward", "prev_state_snapshot",
            "last_action", "last_buttons", "step_count",
            "telemetry",
        }
        assert expected.issubset(set(state.keys()))

    def test_optional_reward_accepts_none(self):
        state = _make_state(reward=None)
        assert state["reward"] is None

    def test_optional_telemetry_accepts_none(self):
        state = _make_state(telemetry=None)
        assert state["telemetry"] is None

    def test_optional_reward_accepts_reward_vector(self):
        rv = RewardVector(milestone_delta=1)
        state = _make_state(reward=rv)
        assert state["reward"].milestone_delta == 1

    def test_optional_telemetry_accepts_snapshot(self):
        snap = TelemetrySnapshot(vlm_calls=2)
        state = _make_state(telemetry=snap)
        assert state["telemetry"].vlm_calls == 2


# ---------------------------------------------------------------------------
# TestRewardVectorDelta
# ---------------------------------------------------------------------------


def _make_state_data(
    milestone_index: int = 0,
    money: int = 0,
    position: tuple = (0, 0),
    goal_coords: tuple | None = None,
    party_levels: list | None = None,
) -> dict:
    return {
        "milestone_index": milestone_index,
        "player": {
            "money": money,
            "position": {"x": position[0], "y": position[1]},
            "party": [{"level": lvl} for lvl in (party_levels or [])],
        },
        "goal_coords": goal_coords,
    }


class TestRewardVectorDelta:
    def test_milestone_delta(self):
        prev = _make_state_data(milestone_index=2)
        curr = _make_state_data(milestone_index=3)
        rv = RewardVector.compute_delta(prev, curr)
        assert rv.milestone_delta == 1

    def test_milestone_delta_no_change(self):
        prev = _make_state_data(milestone_index=5)
        curr = _make_state_data(milestone_index=5)
        rv = RewardVector.compute_delta(prev, curr)
        assert rv.milestone_delta == 0

    def test_manhattan_delta_closer(self):
        """Agent moves from distance 10 to distance 8 → delta = +2."""
        # goal at (10, 0); prev at (0, 0); curr at (2, 0)
        prev = _make_state_data(position=(0, 0), goal_coords=(10, 0))
        curr = _make_state_data(position=(2, 0), goal_coords=(10, 0))
        rv = RewardVector.compute_delta(prev, curr)
        assert rv.manhattan_delta == pytest.approx(2.0)

    def test_manhattan_delta_farther(self):
        """Agent moves away → delta is negative."""
        prev = _make_state_data(position=(5, 0), goal_coords=(10, 0))
        curr = _make_state_data(position=(3, 0), goal_coords=(10, 0))
        rv = RewardVector.compute_delta(prev, curr)
        assert rv.manhattan_delta == pytest.approx(-2.0)

    def test_party_level_sum_delta(self):
        """Treecko levels 7 → 8 → delta = 1."""
        prev = _make_state_data(party_levels=[7])
        curr = _make_state_data(party_levels=[8])
        rv = RewardVector.compute_delta(prev, curr)
        assert rv.party_level_sum_delta == 1

    def test_pokédollar_delta(self):
        """Money 500 → 600 → delta = 100."""
        prev = _make_state_data(money=500)
        curr = _make_state_data(money=600)
        rv = RewardVector.compute_delta(prev, curr)
        assert rv.pokédollar_delta == 100

    def test_empty_states_produce_zero_vector(self):
        rv = RewardVector.compute_delta({}, {})
        assert rv.milestone_delta == 0
        assert rv.manhattan_delta == pytest.approx(0.0)
        assert rv.party_level_sum_delta == 0
        assert rv.pokédollar_delta == 0


# ---------------------------------------------------------------------------
# TestRewardVectorTotal
# ---------------------------------------------------------------------------


class TestRewardVectorTotal:
    def test_milestone_only(self):
        """milestone=1, others=0 → total = 10.0 * 1 = 10.0."""
        rv = RewardVector(milestone_delta=1)
        assert rv.total == pytest.approx(10.0)

    def test_all_zero(self):
        rv = RewardVector()
        assert rv.total == pytest.approx(0.0)

    def test_known_composite(self):
        """milestone=1, manhattan=2.0, level=1, money=100.
        total = 10*1 + 0.01*2 + 1*1 + 0.001*100 = 11.12
        """
        rv = RewardVector(
            milestone_delta=1,
            manhattan_delta=2.0,
            party_level_sum_delta=1,
            pokédollar_delta=100,
        )
        assert rv.total == pytest.approx(11.12)


# ---------------------------------------------------------------------------
# TestRewardVectorSerialization
# ---------------------------------------------------------------------------


class TestRewardVectorSerialization:
    def test_to_dict_excludes_weights(self):
        rv = RewardVector(milestone_delta=1, pokédollar_delta=50)
        d = rv.to_dict()
        assert "WEIGHTS" not in d
        assert d["milestone_delta"] == 1
        assert d["pokédollar_delta"] == 50

    def test_json_round_trip(self):
        rv = RewardVector(
            milestone_delta=2,
            manhattan_delta=3.5,
            party_level_sum_delta=1,
            pokédollar_delta=200,
        )
        serialized = json.dumps(rv.to_dict())
        deserialized = json.loads(serialized)
        assert deserialized["milestone_delta"] == 2
        assert deserialized["manhattan_delta"] == pytest.approx(3.5)

    def test_asdict_is_json_serializable(self):
        """dataclasses.asdict() output can always be JSON-serialised."""
        rv = RewardVector(milestone_delta=1)
        d = asdict(rv)
        # Remove WEIGHTS tuple (not JSON-serialisable) per to_dict() contract
        d.pop("WEIGHTS")
        json.dumps(d)  # Should not raise
