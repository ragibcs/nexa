from pathlib import Path
import urllib.request
from rich.progress import Progress, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn

CACHE_DIR = Path.home() / ".cache" / "nexa" / "models"

MODELS = {
    "inswapper_128": {
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
        "ext": ".onnx",
    },
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
