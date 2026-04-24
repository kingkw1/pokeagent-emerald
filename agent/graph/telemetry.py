"""
agent/graph/telemetry — TelemetryLogger for VLM call / token / latency tracking.

Tracks per-step API usage and writes a JSONL log to run_logs/.
Use summary() to get aggregate statistics including estimated cost and
VLM savings rate vs. a naive full-VLM approach.

Usage::

    telemetry = TelemetryLogger()

    # --- in Agent.step() before graph.invoke() ---
    telemetry.begin_step()

    # --- inside any node that calls the VLM ---
    telemetry.record_vlm_call(
        input_tokens=resp.usage_metadata.prompt_token_count,
        output_tokens=resp.usage_metadata.candidates_token_count,
        node="map_stitcher_relay",
    )

    # --- in Agent.step() after graph.invoke() ---
    snapshot = telemetry.end_step(
        step=step_count,
        milestone_index=state["milestone_index"],
        last_action=state.get("last_action", ""),
    )
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from agent.graph.state import TelemetrySnapshot

# ---------------------------------------------------------------------------
# Gemini Flash pricing — update if the model pricing changes
# ---------------------------------------------------------------------------
_INPUT_COST_PER_1K = 0.000075   # USD per 1K input tokens
_OUTPUT_COST_PER_1K = 0.000300  # USD per 1K output tokens


class TelemetryLogger:
    """Tracks VLM API calls, token consumption, and step latency.

    Writes one JSONL entry per step to ``run_logs/telemetry_<timestamp>.jsonl``.
    """

    def __init__(self, log_dir: str = "run_logs") -> None:
        self._log_path = (
            Path(log_dir)
            / f"telemetry_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        )
        self._log_path.parent.mkdir(exist_ok=True)

        self._total_vlm_calls: int = 0
        self._total_input_tok: int = 0
        self._total_output_tok: int = 0
        self._total_steps: int = 0

        self._step_start: float | None = None
        self._current: TelemetrySnapshot = TelemetrySnapshot()

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------

    def begin_step(self) -> None:
        """Reset the current-step snapshot and start the latency timer.

        Call this from ``Agent.step()`` **before** ``graph.invoke()``.
        """
        self._step_start = time.perf_counter()
        self._current = TelemetrySnapshot()

    def end_step(
        self,
        step: int,
        milestone_index: int,
        last_action: str,
    ) -> TelemetrySnapshot:
        """Finalise the current step, write a JSONL entry, and return the snapshot.

        Call this from ``Agent.step()`` **after** ``graph.invoke()``.

        Args:
            step: Cumulative step number.
            milestone_index: Current milestone pointer.
            last_action: Human-readable label for the action taken.

        Returns:
            The completed :class:`TelemetrySnapshot` for this step.
        """
        elapsed_ms = (
            (time.perf_counter() - self._step_start) * 1000
            if self._step_start is not None
            else 0.0
        )
        self._current.step_latency_ms = elapsed_ms
        self._total_steps += 1

        entry = {
            "step": step,
            "milestone_index": milestone_index,
            "last_action": last_action,
            **asdict(self._current),
        }
        with open(self._log_path, "a", encoding="utf-8") as fh:
            json.dump(entry, fh)
            fh.write("\n")

        return self._current

    # ------------------------------------------------------------------
    # VLM call recording
    # ------------------------------------------------------------------

    def record_vlm_call(
        self,
        input_tokens: int,
        output_tokens: int,
        node: str,
    ) -> None:
        """Record a single VLM API call made during the current step.

        Call this from any node that invokes the Gemini VLM.

        Args:
            input_tokens: Prompt token count from ``response.usage_metadata``.
            output_tokens: Completion token count from ``response.usage_metadata``.
            node: Name of the node making the call (e.g. ``"map_stitcher_relay"``).
        """
        self._current.vlm_calls += 1
        self._current.input_tokens += input_tokens
        self._current.output_tokens += output_tokens
        self._current.node_fired = node

        self._total_vlm_calls += 1
        self._total_input_tok += input_tokens
        self._total_output_tok += output_tokens

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return aggregate statistics for the current run.

        Returns:
            Dict with keys: ``total_steps``, ``total_vlm_calls``,
            ``vlm_call_rate``, ``vlm_savings_vs_naive``,
            ``total_input_tokens``, ``total_output_tokens``,
            ``estimated_cost_usd``.
        """
        total_steps = max(self._total_steps, 1)
        vlm_rate = self._total_vlm_calls / total_steps
        cost = (
            self._total_input_tok / 1000 * _INPUT_COST_PER_1K
            + self._total_output_tok / 1000 * _OUTPUT_COST_PER_1K
        )
        return {
            "total_steps": self._total_steps,
            "total_vlm_calls": self._total_vlm_calls,
            "vlm_call_rate": round(vlm_rate, 4),
            "vlm_savings_vs_naive": f"{(1 - vlm_rate) * 100:.1f}%",
            "total_input_tokens": self._total_input_tok,
            "total_output_tokens": self._total_output_tok,
            "estimated_cost_usd": round(cost, 6),
        }

    # ------------------------------------------------------------------
    # Properties (read-only introspection)
    # ------------------------------------------------------------------

    @property
    def log_path(self) -> Path:
        """Path to the JSONL telemetry log file."""
        return self._log_path
