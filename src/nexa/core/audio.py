import subprocess
from pathlib import Path


def extract_audio(input_path: Path, output_audio_path: Path) -> bool:
    """Extract the audio stream from a video file (codec copy, no re-encode)."""
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vn", "-acodec", "copy", str(output_audio_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_audio_path.exists() and output_audio_path.stat().st_size > 0
    except subprocess.CalledProcessError:
        return False


def mux_audio_video(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    """Combine a video file (no audio) with a separate audio file."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False
