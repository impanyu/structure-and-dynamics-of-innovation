"""Filesystem utilities: atomic writes for cache safety."""
import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically: write to a temp file, flush, fsync, then replace.

    This ensures concurrent readers never see a half-written file. The parent
    directory is created if missing. If anything fails, the original file is
    unchanged and no temp file remains on disk.

    Args:
        path: Target file path.
        text: Text content to write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file in the same directory so os.replace() works atomically
    # on most filesystems (guaranteed on POSIX).
    tmp_file = tempfile.NamedTemporaryFile(
        mode='w',
        dir=path.parent,
        delete=False,
        suffix=".tmp"
    )
    try:
        tmp_file.write(text)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
        tmp_file.close()
        # Atomic rename: os.replace() replaces the target if it exists
        # and is atomic on POSIX systems.
        os.replace(tmp_file.name, path)
    except Exception:
        # Clean up temp file on any failure
        tmp_file.close()
        try:
            os.unlink(tmp_file.name)
        except OSError:
            pass
        raise
