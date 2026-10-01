"""Tests for atomic filesystem writes."""
import os
import tempfile
from pathlib import Path

import pytest

from innovation.core.fsutil import atomic_write_text


class TestAtomicWriteText:
    """atomic_write_text safely writes JSON and other files."""

    def test_writes_content(self, tmp_path):
        """Writing new file succeeds and contains exact text."""
        path = tmp_path / "test.json"
        text = '{"key": "value"}'

        atomic_write_text(path, text)

        assert path.exists()
        assert path.read_text() == text

    def test_overwrites_existing(self, tmp_path):
        """Overwriting an existing file replaces it atomically."""
        path = tmp_path / "test.json"
        old_text = '{"old": "data"}'
        new_text = '{"new": "data"}'

        path.write_text(old_text)
        atomic_write_text(path, new_text)

        assert path.read_text() == new_text

    def test_leaves_no_temp_files(self, tmp_path):
        """No .tmp files remain after successful write."""
        path = tmp_path / "test.json"
        text = '{"data": "content"}'

        atomic_write_text(path, text)

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Unexpected temp files: {tmp_files}"

    def test_creates_missing_parent_dir(self, tmp_path):
        """Parent directories are created if missing."""
        path = tmp_path / "a" / "b" / "c" / "test.json"
        text = '{"nested": "content"}'

        atomic_write_text(path, text)

        assert path.exists()
        assert path.read_text() == text

    def test_failure_cleans_temp_file(self, tmp_path, monkeypatch):
        """On failure, temp file is cleaned up and original unchanged."""
        path = tmp_path / "test.json"
        original_text = '{"original": "content"}'
        path.write_text(original_text)

        # Simulate os.replace failure
        def failing_replace(src, dst):
            raise OSError("Simulated replace failure")

        monkeypatch.setattr(os, "replace", failing_replace)

        with pytest.raises(OSError, match="Simulated replace failure"):
            atomic_write_text(path, '{"new": "content"}')

        # Original file unchanged
        assert path.read_text() == original_text

        # No temp files left behind
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == [], f"Temp files not cleaned up: {tmp_files}"

    def test_large_content(self, tmp_path):
        """Writing large content works correctly."""
        path = tmp_path / "large.json"
        text = '{"data": "' + "x" * 1000000 + '"}'

        atomic_write_text(path, text)

        assert path.read_text() == text

    def test_special_characters(self, tmp_path):
        """Writing text with special characters works."""
        path = tmp_path / "special.json"
        text = '{"emoji": "🎉", "newline": "a\\nb", "quote": "\\"quoted\\""}'

        atomic_write_text(path, text)

        assert path.read_text() == text
