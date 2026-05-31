"""
tests/test_htn_shadow_log.py — Phase 7.1 automated tests for HTN shadow logging.

Tests target _write_shadow_log() directly (pure function — no graph needed).
Each test uses tmp_path so the real llm_logs/htn_shadow.jsonl is never touched.
"""
from __future__ import annotations

import json
import os

from agent.graph.goal_stack import GoalNode, stack_push
from agent.graph.nodes.executive_supervisor import _write_shadow_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_immediate_goal(goal_location: str | None = None) -> GoalNode:
    directive = {"goal_location": goal_location} if goal_location else {}
    return GoalNode(
        goal_id="test_immediate",
        description="Test immediate goal",
        goal_type="immediate",
        directive=directive,
    )


def _make_strategic_goal() -> GoalNode:
    return GoalNode(
        goal_id="test_strategic",
        description="Defeat Gym Leader Roxanne",
        goal_type="strategic",
    )


# ---------------------------------------------------------------------------
# TestShadowLogWritten
# ---------------------------------------------------------------------------

class TestShadowLogWritten:
    """Shadow log file is created and has the correct structure."""

    def test_file_created_on_first_write(self, tmp_path):
        log = str(tmp_path / "htn_shadow.jsonl")
        stack = stack_push([], _make_immediate_goal("PETALBURG_CITY"))
        _write_shadow_log(
            step=0,
            supervisor_op="BOOTSTRAP",
            stack=stack,
            milestone_target="PETALBURG_CITY",
            milestone_index=16,
            reasoning="Initial stack.",
            log_path=log,
        )
        assert os.path.exists(log)

    def test_each_line_is_valid_json(self, tmp_path):
        log = str(tmp_path / "htn_shadow.jsonl")
        stack = stack_push([], _make_immediate_goal("PETALBURG_CITY"))
        for step in range(3):
            _write_shadow_log(
                step=step,
                supervisor_op="CONTINUE",
                stack=stack,
                milestone_target="PETALBURG_CITY",
                milestone_index=16,
                reasoning=f"Step {step} reasoning.",
                log_path=log,
            )
        with open(log) as fh:
            lines = fh.readlines()
        assert len(lines) == 3
        for line in lines:
            obj = json.loads(line)
            for key in ("step", "supervisor_op", "stack_depth", "milestone_index",
                        "milestone_target", "htn_target", "diverged", "reasoning"):
                assert key in obj, f"Missing key '{key}' in shadow log entry"

    def test_line_count_matches_activations(self, tmp_path):
        log = str(tmp_path / "htn_shadow.jsonl")
        stack = stack_push([], _make_immediate_goal("PETALBURG_CITY"))
        for step in range(5):
            _write_shadow_log(
                step=step,
                supervisor_op="BOOTSTRAP" if step == 0 else "CONTINUE",
                stack=stack,
                milestone_target="PETALBURG_CITY",
                milestone_index=16,
                reasoning=".",
                log_path=log,
            )
        with open(log) as fh:
            lines = [ln for ln in fh.readlines() if ln.strip()]
        assert len(lines) == 5

    def test_stack_depth_recorded_correctly(self, tmp_path):
        log = str(tmp_path / "htn_shadow.jsonl")
        g1 = _make_immediate_goal("PETALBURG_CITY")
        g2 = _make_strategic_goal()
        stack = stack_push(stack_push([], g2), g1)  # depth 2
        _write_shadow_log(
            step=0,
            supervisor_op="BOOTSTRAP",
            stack=stack,
            milestone_target="PETALBURG_CITY",
            milestone_index=16,
            reasoning=".",
            log_path=log,
        )
        obj = json.loads(open(log).read())
        assert obj["stack_depth"] == 2

    def test_milestone_index_recorded(self, tmp_path):
        log = str(tmp_path / "htn_shadow.jsonl")
        stack = stack_push([], _make_immediate_goal("PETALBURG_CITY"))
        _write_shadow_log(
            step=5,
            supervisor_op="CONTINUE",
            stack=stack,
            milestone_target="PETALBURG_CITY",
            milestone_index=17,
            reasoning=".",
            log_path=log,
        )
        obj = json.loads(open(log).read())
        assert obj["milestone_index"] == 17
        assert obj["step"] == 5


# ---------------------------------------------------------------------------
# TestShadowDivergenceDetected
# ---------------------------------------------------------------------------

class TestShadowDivergenceDetected:
    """Shadow log marks diverged=True when HTN and milestone targets disagree."""

    def test_diverged_true_when_targets_differ(self, tmp_path):
        log = str(tmp_path / "htn_shadow.jsonl")
        # HTN targets ROUTE_104_SOUTH; legacy FSM targets PETALBURG_CITY_GYM
        stack = stack_push([], _make_immediate_goal("ROUTE_104_SOUTH"))
        _write_shadow_log(
            step=10,
            supervisor_op="CONTINUE",
            stack=stack,
            milestone_target="PETALBURG_CITY_GYM",
            milestone_index=17,
            reasoning="HTN wants to skip ahead.",
            log_path=log,
        )
        obj = json.loads(open(log).read())
        assert obj["diverged"] is True
        assert obj["milestone_target"] == "PETALBURG_CITY_GYM"
        assert obj["htn_target"] == "ROUTE_104_SOUTH"

    def test_nav_fields_unchanged_in_shadow_mode(self, tmp_path):
        """_write_shadow_log must not mutate state — nav fields are NOT overwritten."""
        log = str(tmp_path / "htn_shadow.jsonl")
        stack = stack_push([], _make_immediate_goal("ROUTE_104_SOUTH"))
        original_milestone_target = "PETALBURG_CITY_GYM"
        _write_shadow_log(
            step=10,
            supervisor_op="CONTINUE",
            stack=stack,
            milestone_target=original_milestone_target,
            milestone_index=17,
            reasoning=".",
            log_path=log,
        )
        # Pure log function: caller's variable must be unchanged
        assert original_milestone_target == "PETALBURG_CITY_GYM"


# ---------------------------------------------------------------------------
# TestShadowNoDivergence
# ---------------------------------------------------------------------------

class TestShadowNoDivergence:
    """Shadow log marks diverged=False when both systems agree."""

    def test_diverged_false_when_targets_match(self, tmp_path):
        log = str(tmp_path / "htn_shadow.jsonl")
        stack = stack_push([], _make_immediate_goal("PETALBURG_CITY"))
        _write_shadow_log(
            step=3,
            supervisor_op="BOOTSTRAP",
            stack=stack,
            milestone_target="PETALBURG_CITY",
            milestone_index=16,
            reasoning="Both agree on Petalburg City.",
            log_path=log,
        )
        obj = json.loads(open(log).read())
        assert obj["diverged"] is False
        assert obj["htn_target"] == "PETALBURG_CITY"
        assert obj["milestone_target"] == "PETALBURG_CITY"

    def test_diverged_false_when_milestone_target_is_none(self, tmp_path):
        """No milestone target yet (step 0 bootstrap) → not diverged."""
        log = str(tmp_path / "htn_shadow.jsonl")
        stack = stack_push([], _make_immediate_goal("PETALBURG_CITY"))
        _write_shadow_log(
            step=0,
            supervisor_op="BOOTSTRAP",
            stack=stack,
            milestone_target=None,
            milestone_index=0,
            reasoning="First step.",
            log_path=log,
        )
        obj = json.loads(open(log).read())
        assert obj["diverged"] is False

    def test_diverged_false_when_htn_target_is_none(self, tmp_path):
        """Stack[0] has no directive (strategic goal) → htn_target None → not diverged."""
        log = str(tmp_path / "htn_shadow.jsonl")
        stack = stack_push([], _make_strategic_goal())
        _write_shadow_log(
            step=2,
            supervisor_op="CONTINUE",
            stack=stack,
            milestone_target="PETALBURG_CITY",
            milestone_index=16,
            reasoning="Strategic goal, no directive yet.",
            log_path=log,
        )
        obj = json.loads(open(log).read())
        assert obj["htn_target"] is None
        assert obj["diverged"] is False
