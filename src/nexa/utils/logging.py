import shutil
from rich.console import Console

console = Console(stderr=True)

def log_info(msg: str):
    console.print(f"[blue][INFO][/blue] {msg}")

def log_success(msg: str):
    console.print(f"[green][OK][/green] {msg}")

def log_warn(msg: str):
    console.print(f"[yellow][WARN][/yellow] {msg}")

def log_error(msg: str):
    console.print(f"[red][ERROR][/red] {msg}")

def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        log_error("FFmpeg is not installed or not found in PATH.")
        log_error("Install it with: sudo apt install ffmpeg  (Linux) or brew install ffmpeg  (macOS)")
        raise SystemExit(1)
