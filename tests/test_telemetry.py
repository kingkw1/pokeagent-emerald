"""
Tests for Phase 1 — TelemetryLogger.

Covers:
  TestTelemetrySnapshotDefaults — default field values
  TestBeginEndStep              — begin_step / end_step lifecycle
  TestRecordVlmCall             — accumulation and node_fired tracking
  TestSummaryVlmRate            — vlm_call_rate and savings string
  TestSummaryCostEstimate       — estimated_cost_usd calculation
  TestLogRoundtrip              — JSONL log integrity
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agent.graph.state import TelemetrySnapshot
from agent.graph.telemetry import TelemetryLogger, _INPUT_COST_PER_1K, _OUTPUT_COST_PER_1K


# ---------------------------------------------------------------------------
# TestTelemetrySnapshotDefaults
# ---------------------------------------------------------------------------


class TestTelemetrySnapshotDefaults:
    def test_vlm_calls_zero(self):
        snap = TelemetrySnapshot()
        assert snap.vlm_calls == 0

    def test_latency_zero(self):
        snap = TelemetrySnapshot()
        assert snap.step_latency_ms == pytest.approx(0.0)

    def test_node_fired_empty(self):
        snap = TelemetrySnapshot()
        assert snap.node_fired == ""

    def test_tokens_zero(self):
        snap = TelemetrySnapshot()
        assert snap.input_tokens == 0
        assert snap.output_tokens == 0


# ---------------------------------------------------------------------------
# TestBeginEndStep
# ---------------------------------------------------------------------------


class TestBeginEndStep:
    def test_begin_resets_snapshot(self, tmp_path):
        tl = TelemetryLogger(log_dir=str(tmp_path))
        tl.begin_step()
        tl.record_vlm_call(100, 50, "some_node")
        # begin again — should reset
        tl.begin_step()
        assert tl._current.vlm_calls == 0
        assert tl._current.node_fired == ""

    def test_end_step_returns_snapshot_with_nonzero_latency(self, tmp_path):
        tl = TelemetryLogger(log_dir=str(tmp_path))
        tl.begin_step()
        time.sleep(0.001)
        snap = tl.end_step(step=1, milestone_index=0, last_action="NAVIGATE")
        assert snap.step_latency_ms > 0.0

    def test_end_step_writes_one_jsonl_line(self, tmp_path):
        tl = TelemetryLogger(log_dir=str(tmp_path))
        tl.begin_step()
        tl.end_step(step=1, milestone_index=2, last_action="BATTLE")
        lines = tl.log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

    def test_jsonl_line_contains_required_fields(self, tmp_path):
        tl = TelemetryLogger(log_dir=str(tmp_path))
        tl.begin_step()
        tl.end_step(step=5, milestone_index=3, last_action="DIALOGUE")
        entry = json.loads(tl.log_path.read_text(encoding="utf-8").strip())
        assert entry["step"] == 5
        assert entry["milestone_index"] == 3
        assert entry["last_action"] == "DIALOGUE"
        assert "vlm_calls" in entry
        assert "step_latency_ms" in entry
        assert "node_fired" in entry


# ---------------------------------------------------------------------------
# TestRecordVlmCall
# ---------------------------------------------------------------------------


class TestRecordVlmCall:
    def test_single_call_increments_vlm_calls(self, tmp_path):
        tl = TelemetryLogger(log_dir=str(tmp_path))
        tl.begin_step()
        tl.record_vlm_call(100, 50, "map_stitcher_relay")
        assert tl._current.vlm_calls == 1

    def test_single_call_records_tokens(self, tmp_path):
        tl = TelemetryLogger(log_dir=str(tmp_path))
        tl.begin_step()
        tl.record_vlm_call(100, 50, "map_stitcher_relay")
        assert tl._current.input_tokens == 100
        assert tl._current.output_tokens == 50

    def test_multiple_calls_accumulate(self, tmp_path):
        tl = TelemetryLogger(log_dir=str(tmp_path))
        tl.begin_step()
        tl.record_vlm_call(100, 50, "node_a")
        tl.record_vlm_call(200, 80, "node_b")
        assert tl._current.vlm_calls == 2
        assert tl._current.input_tokens == 300
        assert tl._current.output_tokens == 130

    def test_node_fired_reflects_last_call(self, tmp_path):
        tl = TelemetryLogger(log_dir=str(tmp_path))
        tl.begin_step()
        tl.record_vlm_call(10, 5, "node_a")
        tl.record_vlm_call(10, 5, "node_b")
        assert tl._current.node_fired == "node_b"

    def test_totals_accumulate_across_steps(self, tmp_path):
        tl = TelemetryLogger(log_dir=str(tmp_path))
        for i in range(3):
            tl.begin_step()
            tl.record_vlm_call(100, 50, "nav_bot")
            tl.end_step(step=i, milestone_index=0, last_action="NAVIGATE")
        assert tl._total_vlm_calls == 3
        assert tl._total_input_tok == 300
        assert tl._total_output_tok == 150


# ---------------------------------------------------------------------------
# TestSummaryVlmRate
# ---------------------------------------------------------------------------


def _run_n_steps(tl: TelemetryLogger, n: int, vlm_steps: int) -> None:
    """Run *n* steps, making one VLM call on the first *vlm_steps* steps."""
    for i in range(n):
        tl.begin_step()
        if i < vlm_steps:
            tl.record_vlm_call(100, 50, "test_node")
        tl.end_step(step=i, milestone_index=0, last_action="TEST")


class TestSummaryVlmRate:
    def test_one_in_ten(self, tmp_path):
        tl = TelemetryLogger(log_dir=str(tmp_path))
        _run_n_steps(tl, n=10, vlm_steps=1)
        s = tl.summary()
        assert s["vlm_call_rate"] == pytest.approx(0.1, abs=1e-4)
        assert s["vlm_savings_vs_naive"] == "90.0%"

    def test_zero_vlm_calls(self, tmp_path):
        tl = TelemetryLogger(log_dir=str(tmp_path))
        _run_n_steps(tl, n=10, vlm_steps=0)
        s = tl.summary()
        assert s["vlm_call_rate"] == pytest.approx(0.0)
        assert s["vlm_savings_vs_naive"] == "100.0%"

    def test_all_vlm_calls(self, tmp_path):
        tl = TelemetryLogger(log_dir=str(tmp_path))
        _run_n_steps(tl, n=10, vlm_steps=10)
        s = tl.summary()
        assert s["vlm_call_rate"] == pytest.approx(1.0, abs=1e-4)
        assert s["vlm_savings_vs_naive"] == "0.0%"


# ---------------------------------------------------------------------------
# TestSummaryCostEstimate
# ---------------------------------------------------------------------------


class TestSummaryCostEstimate:
    def test_nonzero_tokens_produce_positive_cost(self, tmp_path):
        tl = TelemetryLogger(log_dir=str(tmp_path))
        tl.begin_step()
        tl.record_vlm_call(1000, 200, "test_node")
        tl.end_step(step=0, milestone_index=0, last_action="TEST")
        s = tl.summary()
        assert s["estimated_cost_usd"] > 0.0

    def test_correct_cost_calculation(self, tmp_path):
        tl = TelemetryLogger(log_dir=str(tmp_path))
        tl.begin_step()
        tl.record_vlm_call(1000, 200, "test_node")
        tl.end_step(step=0, milestone_index=0, last_action="TEST")
        expected = (1000 / 1000 * _INPUT_COST_PER_1K) + (200 / 1000 * _OUTPUT_COST_PER_1K)
        s = tl.summary()
        assert s["estimated_cost_usd"] == pytest.approx(expected, rel=1e-4)

    def test_zero_tokens_zero_cost(self, tmp_path):
        tl = TelemetryLogger(log_dir=str(tmp_path))
        tl.begin_step()
        tl.end_step(step=0, milestone_index=0, last_action="PASS")
        s = tl.summary()
        assert s["estimated_cost_usd"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestLogRoundtrip
# ---------------------------------------------------------------------------


class TestLogRoundtrip:
    def test_all_entries_parseable(self, tmp_path):
        tl = TelemetryLogger(log_dir=str(tmp_path))
        for i in range(5):
            tl.begin_step()
            tl.end_step(step=i, milestone_index=0, last_action="NAVIGATE")
        lines = tl.log_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            entry = json.loads(line)
            assert isinstance(entry, dict)

    def test_numeric_fields_are_correct_type(self, tmp_path):
        tl = TelemetryLogger(log_dir=str(tmp_path))
        tl.begin_step()
        tl.record_vlm_call(100, 50, "test_node")
        tl.end_step(step=0, milestone_index=1, last_action="BATTLE")
        entry = json.loads(tl.log_path.read_text(encoding="utf-8").strip())
        assert isinstance(entry["vlm_calls"], int)
        assert isinstance(entry["step_latency_ms"], float)
        assert isinstance(entry["input_tokens"], int)
        assert isinstance(entry["output_tokens"], int)

    def test_five_steps_produce_five_lines(self, tmp_path):
        tl = TelemetryLogger(log_dir=str(tmp_path))
        for i in range(5):
            tl.begin_step()
            tl.end_step(step=i, milestone_index=0, last_action="NAVIGATE")
        lines = tl.log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5
