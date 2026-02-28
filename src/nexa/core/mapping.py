import cv2
import numpy as np
from typing import Optional

from nexa.utils.logging import log_info

DEFAULT_SIMILARITY_THRESHOLD = 0.6


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine similarity safe against zero-norm vectors."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


class FaceMapper:
    """
    Maps source faces to target faces using cosine similarity of embeddings.

    mappings: list of (source_path, target_path | None).
              If target_path is None, the source applies to ALL detected faces.
    """

    def __init__(self, analyzer, mappings: list, threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.analyzer = analyzer
        self.threshold = threshold
        self.default_source_face = None
        self.specific_mappings: list[dict] = []

        self._build(mappings)

    def _build(self, mappings: list):
        for src_path, tgt_path in mappings:
            src_img = cv2.imread(str(src_path))
            if src_img is None:
                raise FileNotFoundError(f"Cannot read source image: {src_path}")
            src_faces = self.analyzer.analyze(src_img)
            if not src_faces:
                raise ValueError(f"No face detected in source image: {src_path}")
            src_face = src_faces[0]

            if tgt_path is None:
                self.default_source_face = src_face
                log_info(f"Default source face loaded from {src_path}")
            else:
                tgt_img = cv2.imread(str(tgt_path))
                if tgt_img is None:
                    raise FileNotFoundError(f"Cannot read target reference image: {tgt_path}")
                tgt_faces = self.analyzer.analyze(tgt_img)
                if not tgt_faces:
                    raise ValueError(f"No face detected in target reference image: {tgt_path}")
                self.specific_mappings.append({
                    "target_emb": tgt_faces[0].embedding,
                    "source_face": src_face,
                })
                log_info(f"Mapped {src_path} -> {tgt_path}")

    def get_source_for_target(self, target_face) -> Optional[object]:
        """
        Return the source Face that should replace *target_face*,
        or None if nothing matches.
        """
        if not self.specific_mappings:
            return self.default_source_face

        best_sim = -1.0
        best_src = None
        target_emb = target_face.embedding

        for m in self.specific_mappings:
            sim = cosine_similarity(target_emb, m["target_emb"])
            if sim > self.threshold and sim > best_sim:
                best_sim = sim
                best_src = m["source_face"]

        return best_src if best_src is not None else self.default_source_face
