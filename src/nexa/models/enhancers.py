"""GFPGAN / CodeFormer post-processing face enhancement."""

from __future__ import annotations

import sys

import numpy as np
from rich.console import Console

from nexa.models.manager import download_gfpgan, download_codeformer

console = Console()


def _patch_torchvision() -> None:
    """Monkeypatch for ``torchvision.transforms.functional_tensor`` removed
    in torchvision >= 0.17.  GFPGAN still imports it internally."""
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        import torchvision.transforms.functional as _functional
        sys.modules["torchvision.transforms.functional_tensor"] = _functional


class GFPGANEnhancer:
    """Enhance faces using GFPGANv1.4."""

    def __init__(self) -> None:
        _patch_torchvision()
        model_path = download_gfpgan()
        console.print(f"[bold green]Loading GFPGAN[/] from {model_path}")

        from gfpgan import GFPGANer

        self.enhancer = GFPGANer(
            model_path=str(model_path),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        console.print("[green]GFPGAN ready.[/]")

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance all faces in *image* (BGR, uint8).  Returns enhanced BGR."""
        _, _, output = self.enhancer.enhance(
            image, has_aligned=False, only_center_face=False, paste_back=True
        )
        return output


class CodeFormerEnhancer:
    """Temporary CodeFormer wrapper.

    Until native CodeFormer processing is implemented, this enhancer always
    falls back to GFPGAN to avoid silently returning the input image unchanged.
    """

    def __init__(self) -> None:
        _patch_torchvision()
        try:
            from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper  # noqa: F401
            model_path = download_codeformer()
            console.print(f"[bold yellow]CodeFormer weights found[/] at {model_path}")
            console.print(
                "[yellow]CodeFormer inference is not implemented yet — using GFPGAN fallback.[/]"
            )
        except ImportError:
            console.print(
                "[yellow]CodeFormer not available — falling back to GFPGAN.[/]"
            )

        self._fallback = GFPGANEnhancer()

    def enhance(self, image: np.ndarray) -> np.ndarray:
        return self._fallback.enhance(image)


def get_enhancer(name: str | None):
    """Factory: return an enhancer instance or ``None``."""
    if name is None:
        return None
    name = name.lower().strip()
    if name == "gfpgan":
        return GFPGANEnhancer()
    elif name == "codeformer":
        return CodeFormerEnhancer()
    else:
        console.print(f"[red]Unknown enhancer '{name}' — skipping.[/]")
        return None
