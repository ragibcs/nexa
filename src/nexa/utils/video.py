"""Video format detection, frame counting, and I/O helpers."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
from rich.console import Console

console = Console()

# Common video extensions
VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
}


def is_video(path: str | Path) -> bool:
    """Return ``True`` if *path* has a video extension."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_image(path: str | Path) -> bool:
    """Return ``True`` if *path* has an image extension."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def count_frames(path: str | Path) -> int:
    """Return the number of frames in a video file."""
    try:
        props = iio.improps(str(path), plugin="pyav")
        if hasattr(props, "n_images"):
            return props.n_images
    except Exception:
        pass

    # Fallback: iterate and count
    count = 0
    try:
        for _ in iio.imiter(str(path), plugin="pyav"):
            count += 1
    except Exception:
        pass
    return count


def get_fps(path: str | Path) -> float:
    """Return the FPS of a video file (default 30.0)."""
    try:
        meta = iio.immeta(str(path), plugin="pyav")
        return float(meta.get("fps", 30.0))
    except Exception:
        return 30.0
