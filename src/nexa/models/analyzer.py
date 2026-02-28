import insightface
import numpy as np

from nexa.models.providers import get_providers


class FaceAnalyzer:
    def __init__(self, use_gpu: bool = False):
        providers = get_providers(use_gpu)
        self.app = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def analyze(self, image: np.ndarray) -> list:
        """Detect all faces in an image and return Face objects with embeddings."""
        return self.app.get(image)
