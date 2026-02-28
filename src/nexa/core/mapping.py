"""Face mapping — match source faces to target faces via cosine similarity."""

from __future__ import annotations

import numpy as np
from rich.console import Console

console = Console()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-12:
        return 0.0
    return float(dot / norm)


class FaceMapper:
    """Map source identities to detected target faces.

    In *single-face* mode a single source embedding is matched against every
    detected target face.  In *multi-face* mode each source is matched to its
    closest target face above the similarity threshold.
    """

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold
        # source_id → 512-d embedding
        self._sources: dict[str, np.ndarray] = {}

    def add_source(self, name: str, embedding: np.ndarray) -> None:
        """Register a source identity."""
        self._sources[name] = embedding

    @property
    def source_count(self) -> int:
        return len(self._sources)

    # ── Matching ──────────────────────────────────────────────────────────────

    def match_single(
        self,
        source_embedding: np.ndarray,
        target_faces: list,
    ) -> list[tuple[int, float]]:
        """Match a single source against all target faces.

        Returns list of ``(face_index, similarity)`` for faces above threshold.
        """
        matches: list[tuple[int, float]] = []
        for i, face in enumerate(target_faces):
            sim = cosine_similarity(source_embedding, face.normed_embedding)
            if sim >= self.threshold:
                matches.append((i, sim))
            else:
                # In single-source mode, swap ALL faces regardless of similarity
                matches.append((i, sim))
        return matches

    def match_multi(
        self,
        target_faces: list,
    ) -> dict[int, tuple[str, np.ndarray, float]]:
        """Match multiple sources to target faces.

        Returns ``{face_index: (source_name, source_embedding, similarity)}``.
        Each target face is matched to the *best* source above threshold.
        """
        result: dict[int, tuple[str, np.ndarray, float]] = {}
        for i, face in enumerate(target_faces):
            best_name: str | None = None
            best_emb: np.ndarray | None = None
            best_sim = -1.0
            for name, src_emb in self._sources.items():
                sim = cosine_similarity(src_emb, face.normed_embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
                    best_emb = src_emb
            if best_name is not None and best_sim >= self.threshold:
                result[i] = (best_name, best_emb, best_sim)
                console.print(
                    f"[dim]Face {i} → {best_name} (sim={best_sim:.3f})[/]"
                )
        return result
