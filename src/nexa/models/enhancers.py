import cv2
import numpy as np

from nexa.utils.logging import log_info, log_warn

AVAILABLE_ENHANCERS = ["gfpgan", "codeformer"]


class FaceEnhancer:
    """Wrapper around GFPGAN / CodeFormer for post-swap face restoration."""

    def __init__(self, name: str = "gfpgan"):
        self.name = name.lower()
        self._restorer = None
        self._load()

    def _load(self):
        if self.name == "gfpgan":
            self._load_gfpgan()
        elif self.name == "codeformer":
            self._load_codeformer()
        else:
            raise ValueError(
                f"Unknown enhancer '{self.name}'. Available: {AVAILABLE_ENHANCERS}"
            )

    def _load_gfpgan(self):
        try:
            from gfpgan import GFPGANer
        except ImportError:
            raise ImportError(
                "gfpgan is not installed. Install it with: pip install gfpgan"
            )

        from nexa.models.manager import get_model_path

        model_path = get_model_path("gfpgan_1.4")
        self._restorer = GFPGANer(
            model_path=str(model_path),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
        )
        log_info("GFPGAN enhancer loaded.")

    def _load_codeformer(self):
        log_warn("CodeFormer enhancer is not yet fully integrated; falling back to GFPGAN.")
        self._load_gfpgan()

    def enhance(self, img: np.ndarray) -> np.ndarray:
        """Enhance / restore faces in a BGR uint8 image."""
        if self._restorer is None:
            return img

        _, _, output = self._restorer.enhance(
            img, has_aligned=False, only_center_face=False, paste_back=True
        )
        return output
