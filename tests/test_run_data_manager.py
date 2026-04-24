"""
Tests for RunDataManager and BackupManager (Phase 0).

Covers:
  TestRunDataManagerInit        — directory + metadata file creation
  TestMilestoneLogging          — log_milestone_completion() persistence
  TestBackupCreation            — create_cache_backup() file copy + naming
  TestBackupManagerIdempotent   — two calls → two distinct timestamped files
"""

import json
import time
from pathlib import Path

import pytest

from utils.run_data_manager import RunDataManager
from utils.backup_manager import BackupManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state_file(tmp_path: Path, name: str = "test.state") -> Path:
    """Create a small dummy .state file in tmp_path."""
    f = tmp_path / name
    f.write_bytes(b"dummy_state_bytes")
    return f


# ---------------------------------------------------------------------------
# TestRunDataManagerInit
# ---------------------------------------------------------------------------


class TestRunDataManagerInit:
    """RunDataManager creates its run directory and metadata.json on init."""

    def test_run_dir_created(self, tmp_path):
        rdm = RunDataManager(run_id="test_run_001", base_dir=str(tmp_path))
        assert (tmp_path / "test_run_001").is_dir()

    def test_metadata_json_exists(self, tmp_path):
        rdm = RunDataManager(run_id="test_run_002", base_dir=str(tmp_path))
        metadata_file = tmp_path / "test_run_002" / "metadata.json"
        assert metadata_file.exists()

    def test_metadata_json_contains_run_id(self, tmp_path):
        rdm = RunDataManager(run_id="test_run_003", base_dir=str(tmp_path))
        metadata_file = tmp_path / "test_run_003" / "metadata.json"
        data = json.loads(metadata_file.read_text())
        assert data["run_id"] == "test_run_003"

    def test_metadata_json_has_start_time(self, tmp_path):
        rdm = RunDataManager(run_id="test_run_004", base_dir=str(tmp_path))
        metadata_file = tmp_path / "test_run_004" / "metadata.json"
        data = json.loads(metadata_file.read_text())
        assert "start_time" in data

    def test_auto_generated_run_id(self, tmp_path):
        rdm = RunDataManager(base_dir=str(tmp_path))
        assert rdm.run_id.startswith("run_")
        assert rdm.run_dir.is_dir()


# ---------------------------------------------------------------------------
# TestMilestoneLogging
# ---------------------------------------------------------------------------


class TestMilestoneLogging:
    """log_milestone_completion() appends entries to milestones.json."""

    def test_milestones_file_created_after_first_log(self, tmp_path):
        rdm = RunDataManager(run_id="log_test_01", base_dir=str(tmp_path))
        rdm.log_milestone_completion("LITTLEROOT_TOWN", step=5)
        milestones_file = tmp_path / "log_test_01" / "milestones.json"
        assert milestones_file.exists()

    def test_entry_contains_milestone_name(self, tmp_path):
        rdm = RunDataManager(run_id="log_test_02", base_dir=str(tmp_path))
        rdm.log_milestone_completion("ROUTE_101", step=10)
        milestones_file = tmp_path / "log_test_02" / "milestones.json"
        entries = json.loads(milestones_file.read_text())
        assert entries[0]["milestone"] == "ROUTE_101"

    def test_entry_contains_step_count(self, tmp_path):
        rdm = RunDataManager(run_id="log_test_03", base_dir=str(tmp_path))
        rdm.log_milestone_completion("OLDALE_TOWN", step=42)
        milestones_file = tmp_path / "log_test_03" / "milestones.json"
        entries = json.loads(milestones_file.read_text())
        assert entries[0]["step"] == 42

    def test_entry_contains_timestamp(self, tmp_path):
        rdm = RunDataManager(run_id="log_test_04", base_dir=str(tmp_path))
        rdm.log_milestone_completion("STARTER_CHOSEN", step=7)
        milestones_file = tmp_path / "log_test_04" / "milestones.json"
        entries = json.loads(milestones_file.read_text())
        assert "timestamp" in entries[0]

    def test_multiple_milestones_all_persisted(self, tmp_path):
        rdm = RunDataManager(run_id="log_test_05", base_dir=str(tmp_path))
        rdm.log_milestone_completion("LITTLEROOT_TOWN", step=1)
        rdm.log_milestone_completion("ROUTE_101", step=20)
        rdm.log_milestone_completion("STARTER_CHOSEN", step=35)
        milestones_file = tmp_path / "log_test_05" / "milestones.json"
        entries = json.loads(milestones_file.read_text())
        assert len(entries) == 3
        assert [e["milestone"] for e in entries] == [
            "LITTLEROOT_TOWN",
            "ROUTE_101",
            "STARTER_CHOSEN",
        ]

    def test_get_run_summary_reflects_logged_milestones(self, tmp_path):
        rdm = RunDataManager(run_id="log_test_06", base_dir=str(tmp_path))
        rdm.log_milestone_completion("GAME_RUNNING", step=0)
        rdm.log_milestone_completion("PLAYER_NAME_SET", step=3)
        summary = rdm.get_run_summary()
        assert summary["milestone_count"] == 2
        assert summary["run_id"] == "log_test_06"

    def test_state_data_included_when_provided(self, tmp_path):
        rdm = RunDataManager(run_id="log_test_07", base_dir=str(tmp_path))
        rdm.log_milestone_completion(
            "PETALBURG_CITY",
            step=100,
            state_data={"x": 5, "y": 10, "location": "PETALBURG_CITY"},
        )
        milestones_file = tmp_path / "log_test_07" / "milestones.json"
        entries = json.loads(milestones_file.read_text())
        assert entries[0]["state_data"]["location"] == "PETALBURG_CITY"


# ---------------------------------------------------------------------------
# TestBackupCreation
# ---------------------------------------------------------------------------


class TestBackupCreation:
    """create_cache_backup() copies the .state file and returns the new path."""

    def test_backup_file_created(self, tmp_path):
        src = _make_state_file(tmp_path)
        cache_dir = tmp_path / "cache"
        bm = BackupManager(cache_dir=str(cache_dir))
        result = bm.create_cache_backup(str(src), "LITTLEROOT_TOWN")
        assert result is not None
        assert Path(result).exists()

    def test_backup_filename_contains_milestone(self, tmp_path):
        src = _make_state_file(tmp_path)
        cache_dir = tmp_path / "cache"
        bm = BackupManager(cache_dir=str(cache_dir))
        result = bm.create_cache_backup(str(src), "ROUTE_101")
        assert "ROUTE_101" in Path(result).name

    def test_backup_filename_contains_timestamp(self, tmp_path):
        src = _make_state_file(tmp_path)
        cache_dir = tmp_path / "cache"
        bm = BackupManager(cache_dir=str(cache_dir))
        result = bm.create_cache_backup(str(src), "STARTER_CHOSEN")
        # Timestamp format: YYYYMMDD_HHMMSS — check at least date digits present
        name = Path(result).name
        assert any(c.isdigit() for c in name)

    def test_backup_file_has_correct_extension(self, tmp_path):
        src = _make_state_file(tmp_path, name="save.state")
        cache_dir = tmp_path / "cache"
        bm = BackupManager(cache_dir=str(cache_dir))
        result = bm.create_cache_backup(str(src), "OLDALE_TOWN")
        assert Path(result).suffix == ".state"

    def test_backup_content_matches_source(self, tmp_path):
        src = _make_state_file(tmp_path)
        cache_dir = tmp_path / "cache"
        bm = BackupManager(cache_dir=str(cache_dir))
        result = bm.create_cache_backup(str(src), "BIRCH_LAB_VISITED")
        assert Path(result).read_bytes() == src.read_bytes()

    def test_returns_none_for_missing_source(self, tmp_path):
        cache_dir = tmp_path / "cache"
        bm = BackupManager(cache_dir=str(cache_dir))
        result = bm.create_cache_backup("/nonexistent/path.state", "MILESTONE")
        assert result is None

    def test_cache_dir_created_if_missing(self, tmp_path):
        nested_cache = tmp_path / "deeply" / "nested" / "cache"
        bm = BackupManager(cache_dir=str(nested_cache))
        assert nested_cache.is_dir()


# ---------------------------------------------------------------------------
# TestBackupManagerIdempotent
# ---------------------------------------------------------------------------


class TestBackupManagerIdempotent:
    """Two calls for the same milestone produce two distinct files (no overwrite)."""

    def test_two_calls_produce_two_files(self, tmp_path):
        src = _make_state_file(tmp_path)
        cache_dir = tmp_path / "cache"
        bm = BackupManager(cache_dir=str(cache_dir))

        result1 = bm.create_cache_backup(str(src), "LITTLEROOT_TOWN")
        # Sleep briefly to ensure different timestamps (1 second granularity)
        time.sleep(1.1)
        result2 = bm.create_cache_backup(str(src), "LITTLEROOT_TOWN")

        assert result1 is not None
        assert result2 is not None
        assert result1 != result2
        assert Path(result1).exists()
        assert Path(result2).exists()

    def test_two_files_have_different_names(self, tmp_path):
        src = _make_state_file(tmp_path)
        cache_dir = tmp_path / "cache"
        bm = BackupManager(cache_dir=str(cache_dir))

        result1 = bm.create_cache_backup(str(src), "RIVAL_BATTLE_1")
        time.sleep(1.1)
        result2 = bm.create_cache_backup(str(src), "RIVAL_BATTLE_1")

        assert Path(result1).name != Path(result2).name

    def test_both_files_contain_correct_data(self, tmp_path):
        src = _make_state_file(tmp_path)
        original_bytes = src.read_bytes()
        cache_dir = tmp_path / "cache"
        bm = BackupManager(cache_dir=str(cache_dir))

        result1 = bm.create_cache_backup(str(src), "MILESTONE_X")
        time.sleep(1.1)
        result2 = bm.create_cache_backup(str(src), "MILESTONE_X")

        assert Path(result1).read_bytes() == original_bytes
        assert Path(result2).read_bytes() == original_bytes
