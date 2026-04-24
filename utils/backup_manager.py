"""
BackupManager — timestamped .state file checkpoints.

Ports Seth Karten's backup pattern to a class-based design.
Writes .state file copies into cache_dir whenever a milestone fires,
enabling Karpathy-loop restarts from any milestone boundary without
replaying the opening sequence.
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BackupManager:
    """Creates timestamped .state checkpoint files on milestone completion."""

    def __init__(self, cache_dir: str = ".pokeagent_cache"):
        """
        Args:
            cache_dir: Directory where backup .state files are written.
                       Created automatically if it does not exist.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"BackupManager ready: {self.cache_dir}")

    def create_cache_backup(
        self,
        state_file: str,
        milestone_name: str,
    ) -> Optional[str]:
        """Copy *state_file* into cache_dir with milestone + timestamp in the name.

        The destination filename is ``<timestamp>_<milestone_name><ext>`` so that:
        - Multiple calls for the same milestone produce distinct files (idempotent).
        - Files sort chronologically by name.

        Args:
            state_file: Path to the source .state file to back up.
            milestone_name: Milestone identifier included in the backup filename.

        Returns:
            Absolute path to the newly created backup file, or None on failure.
        """
        src = Path(state_file)
        if not src.exists():
            logger.warning(f"[BACKUP] Source state file not found: {state_file}")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(
            c if c.isalnum() or c in ("_", "-") else "_" for c in milestone_name
        )
        dest_filename = f"{timestamp}_{safe_name}{src.suffix}"
        dest = self.cache_dir / dest_filename

        try:
            shutil.copy2(src, dest)
            logger.info(f"[BACKUP] {milestone_name} → {dest}")
            print(f"💾 [BACKUP] {milestone_name} → {dest}")
            return str(dest)
        except OSError as exc:
            logger.error(f"[BACKUP] Failed to copy {state_file} → {dest}: {exc}")
            return None
