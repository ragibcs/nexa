"""Model download manager — HuggingFace Hub + direct URLs."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download
from rich.console import Console

console = Console()

CACHE_DIR = Path.home() / ".cache" / "nexa" / "models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Direct-download URLs ─────────────────────────────────────────────────────
GFPGAN_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/"
    "GFPGANv1.4.pth"
)

CODEFORMER_URL = (
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/"
    "codeformer.pth"
)


def download_from_url(url: str, filename: str | None = None) -> Path:
    """Download a file from *url* into the local cache directory."""
    if filename is None:
        filename = url.rsplit("/", 1)[-1]
    dest = CACHE_DIR / filename
    if dest.exists():
        console.print(f"[dim]Cache hit:[/] {dest}")
        return dest
    console.print(f"[bold cyan]Downloading[/] {url} → {dest}")
    import urllib.request
    urllib.request.urlretrieve(url, str(dest))
    return dest


def download_ip_adapter_faceid() -> Path:
    """Download ``ip-adapter-faceid_sd15.bin`` from HuggingFace Hub."""
    path = hf_hub_download(
        repo_id="h94/IP-Adapter-FaceID",
        filename="ip-adapter-faceid_sd15.bin",
        cache_dir=str(CACHE_DIR),
    )
    console.print(f"[dim]IP-Adapter FaceID weights:[/] {path}")
    return Path(path)


def download_gfpgan() -> Path:
    """Download GFPGANv1.4 weights."""
    return download_from_url(GFPGAN_URL, "GFPGANv1.4.pth")


def download_codeformer() -> Path:
    """Download CodeFormer weights."""
    return download_from_url(CODEFORMER_URL, "codeformer.pth")


def ensure_sd15_model(model_id: str = "runwayml/stable-diffusion-v1-5") -> str:
    """Return the model ID (downloaded on first use by diffusers)."""
    console.print(f"[dim]SD1.5 model:[/] {model_id}")
    return model_id
