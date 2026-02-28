import mimetypes
import subprocess
import json
from pathlib import Path

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v', '.ts', '.mts'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

def is_video(path: Path) -> bool:
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type:
        return mime_type.startswith('video')
    return path.suffix.lower() in VIDEO_EXTENSIONS

def is_image(path: Path) -> bool:
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type:
        return mime_type.startswith('image')
    return path.suffix.lower() in IMAGE_EXTENSIONS

def get_video_info(path: Path) -> dict:
    """Probe video metadata using ffprobe."""
    command = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_streams', '-show_format', str(path)
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}

def get_frame_count(path: Path) -> int:
    """Get total frame count from video metadata."""
    info = get_video_info(path)
    for stream in info.get('streams', []):
        if stream.get('codec_type') == 'video':
            count = stream.get('nb_frames')
            if count and count != 'N/A':
                return int(count)
    return 0
