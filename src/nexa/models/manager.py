from pathlib import Path
import urllib.request
from rich.progress import Progress, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn
import os

CACHE_DIR = Path.home() / ".cache" / "nexa" / "models"

MODELS = {
    "gfpgan_1.4": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "ext": ".pth",
    },
    "codeformer": {
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "ext": ".pth",
    },
}


def _download_with_progress(url: str, dest: Path):
    """Download a file with a Rich progress bar."""
    response = urllib.request.urlopen(url)
    total = int(response.headers.get("Content-Length", 0))

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"Downloading {dest.name}", total=total)
        with open(dest, "wb") as f:
            while True:
                chunk = response.read(1024 * 64)
                if not chunk:
                    break
                f.write(chunk)
                progress.update(task, advance=len(chunk))


def download_model(name: str) -> Path:
    """Download model to cache directory if not already present."""
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    info = MODELS[name]
    model_path = CACHE_DIR / f"{name}{info['ext']}"

    if model_path.exists():
        return model_path

    try:
        _download_with_progress(info["url"], model_path)
    except Exception as e:
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(f"Failed to download model '{name}': {e}") from e

    return model_path


def get_model_path(name: str) -> Path:
    return download_model(name)

def ensure_hf_models():
    """Ensure HuggingFace models for IP-Adapter FaceID and LCM are cached."""
    from huggingface_hub import snapshot_download
    from nexa.utils.logging import log_info

    # Make sure we don't symlink in cache (can cause issues with some diffusers loaders)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    log_info("Ensuring HuggingFace models are downloaded (this may take a while on first run)...")

    # 1. Base model (SD1.5) is handled by from_pretrained

    # 2. IP-Adapter FaceID
    snapshot_download(repo_id="h94/IP-Adapter-FaceID", allow_patterns=["*.bin", "*.safetensors", "*.json"])

