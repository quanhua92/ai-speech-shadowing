"""Retention cleanup for user-generated references.

Mirrors the history retention tests: only ``source="user"`` references whose
``updated_at`` is older than the retention window are removed. Seed references
(any other / missing ``source``) are never touched — that is the whole point of
the feature, so it gets its own tests.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ai_speech_shadowing.tts.generator import (
    ReferenceConfig,
    ReferenceManager,
    cleanup_old_references,
)


@pytest.fixture
def manager(tmp_path: Path) -> ReferenceManager:
    return ReferenceManager(ReferenceConfig(base_dir=tmp_path))


def _age(slug: str, manager: ReferenceManager, *, hours: int) -> None:
    """Rewrite a reference's ``updated_at`` to N hours ago."""
    path = manager.metadata_path(slug)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["updated_at"] = (datetime.now(UTC) - timedelta(hours=hours)).isoformat(timespec="seconds")
    path.write_text(json.dumps(data), encoding="utf-8")


class TestCleanup:
    def test_zero_retention_is_noop(self, manager: ReferenceManager) -> None:
        manager.write_metadata("hi", "Hello", "a", "af_heart", source="user")
        assert cleanup_old_references(manager.config.base_dir, retention_hours=0) == 0
        assert manager.metadata_path("hi").is_file()

    def test_deletes_old_user_references(self, manager: ReferenceManager) -> None:
        manager.write_metadata("hi", "Hello", "a", "af_heart", source="user")
        _age("hi", manager, hours=3)

        deleted = cleanup_old_references(manager.config.base_dir, retention_hours=1)
        assert deleted == 1
        assert not manager.slug_path("hi").exists()

    def test_keeps_recent_user_references(self, manager: ReferenceManager) -> None:
        manager.write_metadata("hi", "Hello", "a", "af_heart", source="user")
        # updated_at is "now" -> below the cutoff
        assert cleanup_old_references(manager.config.base_dir, retention_hours=1) == 0
        assert manager.slug_path("hi").is_dir()

    def test_seed_references_are_never_deleted(self, manager: ReferenceManager) -> None:
        """Explicit seed source is protected even when aged out."""
        manager.write_metadata("hi", "Hello", "a", "af_heart", source="seed")
        _age("hi", manager, hours=3)

        assert cleanup_old_references(manager.config.base_dir, retention_hours=1) == 0
        assert manager.slug_path("hi").is_dir()

    def test_missing_source_field_is_protected(self, manager: ReferenceManager) -> None:
        """Legacy seed metadata predating the field defaults to 'seed'."""
        manager.write_metadata("hi", "Hello", "a", "af_heart", source="user")
        # strip the field to simulate a pre-existing bundled reference
        path = manager.metadata_path("hi")
        data = json.loads(path.read_text(encoding="utf-8"))
        data.pop("source", None)
        _age("hi", manager, hours=3)  # also sets old updated_at
        path.write_text(json.dumps(data), encoding="utf-8")

        assert cleanup_old_references(manager.config.base_dir, retention_hours=1) == 0
        assert manager.slug_path("hi").is_dir()

    def test_skips_corrupt_metadata(self, manager: ReferenceManager) -> None:
        manager.write_metadata("hi", "Hello", "a", "af_heart", source="user")
        manager.metadata_path("hi").write_text("{not valid json", encoding="utf-8")

        # Must not crash and must not delete (we can't know its origin).
        assert cleanup_old_references(manager.config.base_dir, retention_hours=1) == 0
        assert manager.slug_path("hi").is_dir()

    def test_skips_dirs_without_metadata(self, manager: ReferenceManager) -> None:
        # an orphan directory with no metadata.json is left alone
        manager.slug_path("orphan").mkdir(parents=True)
        assert cleanup_old_references(manager.config.base_dir, retention_hours=1) == 0
        assert manager.slug_path("orphan").is_dir()

    def test_mixed_batch_only_removes_aged_users(self, manager: ReferenceManager) -> None:
        manager.write_metadata("old-user", "A", "a", "af_heart", source="user")
        _age("old-user", manager, hours=3)
        manager.write_metadata("new-user", "B", "a", "af_heart", source="user")
        manager.write_metadata("old-seed", "C", "a", "af_heart", source="seed")
        _age("old-seed", manager, hours=3)

        assert cleanup_old_references(manager.config.base_dir, retention_hours=1) == 1
        assert not manager.slug_path("old-user").exists()
        assert manager.slug_path("new-user").is_dir()
        assert manager.slug_path("old-seed").is_dir()

    def test_source_setdefault_is_first_write_wins(self, manager: ReferenceManager) -> None:
        """Adding a voice to an existing slug never rewrites its origin."""
        manager.write_metadata("hi", "Hello", "a", "af_heart", source="user")
        # second write omits source (default "seed") but must not flip a user ref
        manager.write_metadata("hi", "Hello", "a", "am_adam")
        meta = manager.read_metadata("hi")
        assert meta["source"] == "user"
