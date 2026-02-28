import insightface
import numpy as np

class FaceAnalyzer:
    def __init__(self, use_gpu: bool = False):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.app = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def analyze(self, image: np.ndarray) -> list:
        """Detect all faces in an image and return Face objects with embeddings."""
        return self.app.get(image)
