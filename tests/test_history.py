"""Tests for evaluation history persistence (pure filesystem I/O)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ai_speech_shadowing.core.feedback import FeedbackReport, build_report
from ai_speech_shadowing.core.fluency import DtwResult, FluencyDiff, PauseInfo
from ai_speech_shadowing.core.history import (
    HistoryEntry,
    cleanup_old_reports,
    delete_report,
    format_summary,
    list_reports,
    load_report,
    report_path,
    save_report,
)
from ai_speech_shadowing.core.phoneme import diff_phonemes
from ai_speech_shadowing.core.prosody import PitchStats, ProsodyDiff


def _report() -> FeedbackReport:
    pitch = PitchStats(
        f0_contour=np.zeros(1, dtype=np.float64),
        times=np.zeros(1, dtype=np.float64),
        mean_hz=200.0,
        median_hz=200.0,
        min_hz=100.0,
        max_hz=300.0,
        range_hz=200.0,
        std_hz=20.0,
        voiced_ratio=1.0,
        pitch_floor=75.0,
        pitch_ceiling=500.0,
    )
    prosody = ProsodyDiff(
        reference=pitch,
        hypothesis=pitch,
        pitch_range_ratio=1.0,
        monotone=False,
        monotone_threshold=0.5,
        score=1.0,
    )
    fluency = FluencyDiff(
        dtw=DtwResult(0.0, 10, 0.0),
        score=1.0,
        reference_pauses=PauseInfo(0, 0.0, ()),
        hypothesis_pauses=PauseInfo(0, 0.0, ()),
        syllable_rate_reference=2.0,
        syllable_rate_hypothesis=2.0,
        syllable_rate_ratio=1.0,
    )
    return build_report(diff_phonemes(["a", "b"], ["a", "b"]), prosody, fluency)


@pytest.fixture
def history_dir(tmp_path: Path) -> Path:
    return tmp_path / "history"


class TestSaveLoad:
    def test_save_writes_json_with_id(self, history_dir: Path) -> None:
        path = save_report(_report(), history_dir=history_dir)
        assert path.is_file()
        assert path.name.startswith("eval_")
        assert path.suffix == ".json"
        data = json.loads(path.read_text())
        assert data["id"].startswith("eval_")
        assert "created_at" in data
        assert data["composite"]["score"] == 100

    def test_load_round_trip(self, history_dir: Path) -> None:
        path = save_report(_report(), history_dir=history_dir)
        rid = path.stem
        data = load_report(rid, history_dir)
        assert data is not None
        assert data["id"] == rid

    def test_load_missing_returns_none(self, history_dir: Path) -> None:
        assert load_report("eval_nope", history_dir) is None


class TestList:
    def test_empty_when_no_dir(self, history_dir: Path) -> None:
        assert list_reports(history_dir) == []

    def test_lists_entries(self, history_dir: Path) -> None:
        save_report(_report(), history_dir=history_dir)
        save_report(_report(), history_dir=history_dir)
        entries = list_reports(history_dir)
        assert len(entries) == 2
        assert all(isinstance(e, HistoryEntry) for e in entries)
        assert all(e.id.startswith("eval_") for e in entries)
        assert all(e.composite_score == 100 for e in entries)

    def test_skips_malformed(self, history_dir: Path) -> None:
        save_report(_report(), history_dir=history_dir)
        (history_dir / "eval_bad.json").write_text("{not json")
        entries = list_reports(history_dir)
        assert len(entries) == 1  # malformed skipped


class TestDelete:
    def test_delete_existing(self, history_dir: Path) -> None:
        path = save_report(_report(), history_dir=history_dir)
        assert delete_report(path.stem, history_dir) is True
        assert not path.is_file()

    def test_delete_missing(self, history_dir: Path) -> None:
        assert delete_report("eval_nope", history_dir) is False


# --------------------------------------------------------------------------- #
# Path safety — report_id traversal must be contained under history_dir
# --------------------------------------------------------------------------- #
class TestReportPathSafety:
    """Regression tests: a report_id containing traversal must not read or
    delete files outside the history (recordings) folder."""

    @pytest.mark.parametrize("report_id", ["..", ".", "../secret", "..%2f..%2fetc", "a/../b"])
    def test_report_path_rejects_traversal(self, history_dir: Path, report_id: str) -> None:
        assert report_path(report_id, history_dir, suffix=".json") is None
        assert report_path(report_id, history_dir, suffix=".wav") is None

    def test_report_path_accepts_clean_id(self, history_dir: Path) -> None:
        p = report_path("eval_abc12345", history_dir, suffix=".json")
        assert p == history_dir / "eval_abc12345.json"

    def test_load_traversal_returns_none(self, history_dir: Path) -> None:
        assert load_report("..", history_dir) is None
        assert load_report("../secret", history_dir) is None

    def test_delete_traversal_returns_false_and_preserves_data(self, history_dir: Path) -> None:
        path = save_report(_report(), history_dir=history_dir)
        # attempt to delete via traversal must fail AND leave real data intact
        assert delete_report("..", history_dir) is False
        assert path.is_file()


class TestFormatSummary:
    def test_contains_scores_and_feedback(self, history_dir: Path) -> None:
        save_report(_report(), history_dir=history_dir)
        data = load_report(list_reports(history_dir)[0].id, history_dir)
        summary = format_summary(data)
        assert "Composite: 100/100" in summary
        assert "Pronunciation" in summary
        assert "Feedback:" in summary


# --------------------------------------------------------------------------- #
# Per-user scoping
# --------------------------------------------------------------------------- #
_UID_A = "a" * 64  # valid 64-hex user id
_UID_B = "b" * 64


class TestUserScoping:
    def test_save_creates_user_subdirectory(self, history_dir: Path) -> None:
        path = save_report(_report(), history_dir=history_dir, user_id=_UID_A)
        assert _UID_A in path.parts
        assert path.is_file()

    def test_list_scoped_to_user(self, history_dir: Path) -> None:
        save_report(_report(), history_dir=history_dir, user_id=_UID_A)
        save_report(_report(), history_dir=history_dir, user_id=_UID_A)
        save_report(_report(), history_dir=history_dir, user_id=_UID_B)
        assert len(list_reports(history_dir, user_id=_UID_A)) == 2
        assert len(list_reports(history_dir, user_id=_UID_B)) == 1

    def test_user_cannot_load_another_users_report(self, history_dir: Path) -> None:
        path = save_report(_report(), history_dir=history_dir, user_id=_UID_A)
        rid = path.stem
        # owner can load
        assert load_report(rid, history_dir, user_id=_UID_A) is not None
        # other user cannot
        assert load_report(rid, history_dir, user_id=_UID_B) is None

    def test_user_cannot_delete_another_users_report(self, history_dir: Path) -> None:
        path = save_report(_report(), history_dir=history_dir, user_id=_UID_A)
        rid = path.stem
        assert delete_report(rid, history_dir, user_id=_UID_B) is False
        assert path.is_file()  # still there

    def test_none_user_sees_all(self, history_dir: Path) -> None:
        save_report(_report(), history_dir=history_dir, user_id=_UID_A)
        save_report(_report(), history_dir=history_dir, user_id=_UID_B)
        save_report(_report(), history_dir=history_dir, user_id="_cli")
        # user_id=None scans every subdirectory (CLI report view)
        assert len(list_reports(history_dir, user_id=None)) == 3

    def test_cli_user_is_the_default_bucket(self, history_dir: Path) -> None:
        path = save_report(_report(), history_dir=history_dir)  # user_id=None
        assert "_cli" in path.parts
        # visible via all-users scan
        assert len(list_reports(history_dir)) == 1

    def test_report_path_with_user_segment(self, history_dir: Path) -> None:
        p = report_path("eval_abc12345", history_dir, user_id=_UID_A, suffix=".json")
        assert p == history_dir / _UID_A / "eval_abc12345.json"

    def test_report_path_rejects_bad_user_id(self, history_dir: Path) -> None:
        assert report_path("eval_abc", history_dir, user_id="../escape", suffix=".json") is None


# --------------------------------------------------------------------------- #
# Retention cleanup
# --------------------------------------------------------------------------- #
class TestCleanup:
    def test_zero_retention_is_noop(self, history_dir: Path) -> None:
        save_report(_report(), history_dir=history_dir, user_id=_UID_A)
        assert cleanup_old_reports(history_dir, retention_days=0) == 0
        assert len(list_reports(history_dir, user_id=_UID_A)) == 1

    def test_deletes_old_reports(self, history_dir: Path) -> None:
        """Reports with created_at older than retention are removed."""
        from datetime import UTC, datetime, timedelta

        old = save_report(_report(), history_dir=history_dir, user_id=_UID_A)
        # rewrite created_at to 30 days ago
        data = json.loads(old.read_text())
        data["created_at"] = (datetime.now(UTC) - timedelta(days=30)).isoformat(timespec="seconds")
        old.write_text(json.dumps(data))
        recent = save_report(_report(), history_dir=history_dir, user_id=_UID_A)

        deleted = cleanup_old_reports(history_dir, retention_days=7)
        assert deleted == 1
        assert not old.is_file()
        assert recent.is_file()

    def test_deletes_wav_alongside_json(self, history_dir: Path) -> None:
        from datetime import UTC, datetime, timedelta

        old = save_report(_report(), history_dir=history_dir, user_id=_UID_A)
        wav = old.with_suffix(".wav")
        wav.write_bytes(b"fake audio")
        data = json.loads(old.read_text())
        data["created_at"] = (datetime.now(UTC) - timedelta(days=30)).isoformat(timespec="seconds")
        old.write_text(json.dumps(data))

        cleanup_old_reports(history_dir, retention_days=7)
        assert not old.is_file()
        assert not wav.is_file()

    def test_removes_empty_user_dirs(self, history_dir: Path) -> None:
        from datetime import UTC, datetime, timedelta

        old = save_report(_report(), history_dir=history_dir, user_id=_UID_A)
        data = json.loads(old.read_text())
        data["created_at"] = (datetime.now(UTC) - timedelta(days=30)).isoformat(timespec="seconds")
        old.write_text(json.dumps(data))

        cleanup_old_reports(history_dir, retention_days=7)
        assert not (history_dir / _UID_A).exists()

    def test_cleans_legacy_flat_files_too(self, history_dir: Path) -> None:
        """Top-level eval_*.json (pre-feature) are also aged out."""
        from datetime import UTC, datetime, timedelta

        old = save_report(_report(), history_dir=history_dir)  # goes to _cli/
        data = json.loads(old.read_text())
        data["created_at"] = (datetime.now(UTC) - timedelta(days=30)).isoformat(timespec="seconds")
        old.write_text(json.dumps(data))

        assert cleanup_old_reports(history_dir, retention_days=7) == 1
        assert not old.is_file()
