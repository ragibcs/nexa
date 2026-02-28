"""InsightFace buffalo_l face detection + ArcFace embeddings."""

from __future__ import annotations

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from rich.console import Console

console = Console()


class FaceAnalyzer:
    """Wrapper around InsightFace ``buffalo_l`` for detection and embedding."""

    def __init__(self, det_size: tuple[int, int] = (640, 640)) -> None:
        console.print("[bold green]Loading InsightFace buffalo_l …[/]")
        # CPU provider is fine — InsightFace detection is fast enough on CPU
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=det_size)
        console.print("[green]InsightFace ready.[/]")

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, image: np.ndarray) -> list:
        """Return list of InsightFace ``Face`` objects (sorted left→right)."""
        faces = self.app.get(image)
        # Sort by horizontal centre of bounding box
        faces = sorted(faces, key=lambda f: (f.bbox[0] + f.bbox[2]) / 2)
        return faces

    @staticmethod
    def embedding(face) -> np.ndarray:
        """Return the 512-d *normed* ArcFace embedding for a detected face."""
        return face.normed_embedding  # already L2-normalised

    @staticmethod
    def landmarks_106(face) -> np.ndarray | None:
        """Return 106-point 2-D landmarks (or ``None`` if unavailable)."""
        lm = getattr(face, "landmark_2d_106", None)
        if lm is None:
            lm = getattr(face, "landmark_2d_68", None)
        return lm

    @staticmethod
    def bbox(face) -> np.ndarray:
        """Return ``[x1, y1, x2, y2]`` bounding box."""
        return face.bbox.astype(int)
