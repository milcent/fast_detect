"""
helpers.py — Utility functions for fast_detect.
"""

from datetime import timedelta
from pathlib import Path

from fast_detect.constants import VIDEO_EXTENSIONS


def collect_video_files(
    folder: str,
    recursive: bool,
) -> tuple[list[Path], list[dict]]:
    """
    Return all readable video files found in *folder*, sorted by name.

    Files that raise ``OSError`` during filesystem stat (e.g. corrupted or
    unreadable entries on DVR volumes, WinError 1392) are silently skipped;
    a warning is printed and each bad path is returned in the ``skipped``
    list so callers can log it to the JSON output.

    Returns:
        (files, skipped)
        - files:   list[Path] — valid, readable video files
        - skipped: list[dict] — {"video": ..., "error": ...} entries
    """
    base = Path(folder)
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    pattern = "**/*" if recursive else "*"
    files:   list[Path] = []
    skipped: list[dict] = []

    for p in base.glob(pattern):
        try:
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                files.append(p)
        except OSError as exc:
            print(f"  [skip] Unreadable path '{p}': {exc}")
            skipped.append({"video": str(p), "error": str(exc)})

    return sorted(files), skipped


def format_timestamp(seconds: float) -> str:
    """Return a human-readable HH:MM:SS.mmm string for a given number of seconds."""
    td            = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    millis        = int((seconds - int(seconds)) * 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs    = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def seconds_to_dict(seconds: float) -> dict:
    """Build a detection entry dict with both raw seconds and a formatted timestamp."""
    return {
        "seconds":   round(seconds, 3),
        "timestamp": format_timestamp(seconds),
    }
