"""
RunDataManager — structured run-specific data collection.

Ports Seth Karten's checkpoint pattern to this codebase.
Enables Karpathy-loop restarts by writing milestone-boundary
metadata to disk so any run can be resumed from a prior checkpoint.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RunDataManager:
    """Manages structured data collection for a single agent run."""

    def __init__(
        self,
        run_id: Optional[str] = None,
        base_dir: str = "run_data",
    ):
        """
        Args:
            run_id: Unique identifier for this run; auto-generated if None.
            base_dir: Root directory under which per-run subdirectories live.
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        if run_id is None:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.run_id: str = run_id
        self.run_dir: Path = self.base_dir / run_id
        self.run_dir.mkdir(exist_ok=True)

        self._milestones_log: Path = self.run_dir / "milestones.json"
        self._metadata_file: Path = self.run_dir / "metadata.json"
        self._milestones: List[Dict[str, Any]] = []

        self._write_metadata()
        logger.info(f"RunDataManager initialized: {self.run_dir}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_metadata(self) -> None:
        """Write initial run metadata to metadata.json."""
        metadata: Dict[str, Any] = {
            "run_id": self.run_id,
            "start_time": datetime.now().isoformat(),
        }
        with open(self._metadata_file, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_milestone_completion(
        self,
        milestone: str,
        step: int,
        state_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a milestone completion entry to milestones.json.

        Args:
            milestone: Milestone identifier string (e.g. "LITTLEROOT_TOWN").
            step: Current agent step counter at the time of completion.
            state_data: Optional game state snapshot for context.
        """
        entry: Dict[str, Any] = {
            "milestone": milestone,
            "step": step,
            "timestamp": datetime.now().isoformat(),
        }
        if state_data:
            entry["state_data"] = state_data

        self._milestones.append(entry)

        with open(self._milestones_log, "w", encoding="utf-8") as fh:
            json.dump(self._milestones, fh, indent=2)

        logger.info(f"[MILESTONE] {milestone} logged at step {step}")

    def get_run_summary(self) -> Dict[str, Any]:
        """Return a dict summarising the run so far.

        Returns:
            Dict with run_id, run_dir path string, milestone count, and
            the full ordered list of milestone entries.
        """
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "milestone_count": len(self._milestones),
            "milestones": list(self._milestones),
        }

    def __str__(self) -> str:
        return f"RunDataManager(run_id={self.run_id}, dir={self.run_dir})"
