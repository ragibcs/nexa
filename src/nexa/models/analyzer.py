"""InsightFace buffalo_l face detection + ArcFace embeddings."""

from __future__ import annotations

import numpy as np
from insightface.app import FaceAnalysis
from rich.console import Console

console = Console()


class FaceAnalyzer:
    """Wrapper around InsightFace ``buffalo_l`` for detection and embedding."""

    def __init__(
        self,
        det_size: tuple[int, int] = (640, 640),
        device: str = "cuda",
        det_score: float = 0.45,
    ) -> None:
        console.print("[bold green]Loading InsightFace buffalo_l …[/]")

        use_cuda = device.startswith("cuda")
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_cuda
            else ["CPUExecutionProvider"]
        )
        ctx_id = 0 if use_cuda else -1

        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_score)
        console.print("[green]InsightFace ready.[/]")

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, image: np.ndarray) -> list:
        """Return list of InsightFace ``Face`` objects (sorted left→right)."""
        faces = self.app.get(image)
        faces = sorted(faces, key=lambda f: (f.bbox[0] + f.bbox[2]) / 2)
        return faces

    @staticmethod
    def best_face(faces: list):
        """Pick the most reliable face (highest det score, then largest area)."""
        if not faces:
            return None
        return max(
            faces,
            key=lambda f: (
                float(getattr(f, "det_score", 0.0)),
                float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
            ),
        )

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
