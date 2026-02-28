"""FFmpeg audio extract / mux for video processing."""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path

from rich.console import Console

console = Console()


def has_ffmpeg() -> bool:
    """Check whether ``ffmpeg`` is available on PATH."""
    return shutil.which("ffmpeg") is not None


def extract_audio(video_path: str | Path, audio_path: str | Path) -> bool:
    """Extract the audio track from *video_path* to *audio_path*.

    Returns ``True`` if audio was successfully extracted, ``False`` otherwise.
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)

    if not has_ffmpeg():
        console.print("[yellow]FFmpeg not found — skipping audio extraction.[/]")
        return False

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "copy",
        str(audio_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and audio_path.exists():
            console.print(f"[dim]Audio extracted → {audio_path}[/]")
            return True
        else:
            console.print("[yellow]No audio track found or extraction failed.[/]")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def mux_audio(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Mux *audio_path* into *video_path*, writing to *output_path*.

    If muxing fails, the video-only file is returned instead.
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if not has_ffmpeg() or not audio_path.exists():
        # Just copy the video as-is
        shutil.copy2(video_path, output_path)
        return output_path

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0 and output_path.exists():
            console.print(f"[dim]Audio muxed → {output_path}[/]")
            return output_path
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: copy video without audio
    shutil.copy2(video_path, output_path)
    return output_path
