import onnxruntime
import numpy as np
import cv2
from insightface.utils import face_align

from nexa.models.manager import get_model_path
from nexa.models.providers import get_providers


class Swapper:
    def __init__(self, model_name: str = "inswapper_128", use_gpu: bool = False):
        self.model_path = get_model_path(model_name)
        providers = get_providers(use_gpu)
        self.session = onnxruntime.InferenceSession(str(self.model_path), providers=providers)
        inputs = self.session.get_inputs()
        self.input_names = [inp.name for inp in inputs]
        outputs = self.session.get_outputs()
        self.output_names = [out.name for out in outputs]
        self.input_shape = inputs[0].shape
        self.input_size = (self.input_shape[2], self.input_shape[3])

    def swap(self, img: np.ndarray, source_face, target_face) -> np.ndarray:
        """
        Swap the target_face region in img with the source_face identity.

        Args:
            img: BGR uint8 numpy array.
            source_face: InsightFace Face object (provides embedding / identity).
            target_face: InsightFace Face object (provides kps / location to replace).

        Returns:
            The image with the face swapped, as BGR uint8.
        """
        # Prepare source embedding as float32
        source_emb = source_face.embedding.reshape((1, -1)).astype(np.float32)
        source_emb = source_emb / np.linalg.norm(source_emb)

        # Align target face
        aimg, M = face_align.norm_crop2(img, target_face.kps, self.input_size[0])
        blob = cv2.dnn.blobFromImage(
            aimg, 1.0 / 255.0, self.input_size, (0.0, 0.0, 0.0), swapRB=True
        )

        pred = self.session.run(
            self.output_names,
            {self.input_names[0]: blob, self.input_names[1]: source_emb},
        )[0][0]

        # Post-process: CHW -> HWC, BGR, scale to 0-255
        pred = pred.transpose((1, 2, 0))[:, :, ::-1] * 255.0
        pred = np.clip(pred, 0, 255).astype(np.float32)

        h, w = img.shape[:2]
        bimg = cv2.warpAffine(
            pred, M, (w, h), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
        )

        # Build soft mask and warp back
        mask = np.ones((self.input_size[0], self.input_size[1], 3), dtype=np.float32)
        mask = cv2.warpAffine(
            mask, M, (w, h), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT
        )

        # Feather edges slightly for a smoother blend
        kernel = np.ones((3, 3), np.float32)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        img_f = img.astype(np.float32)
        res = img_f * (1.0 - mask) + bimg * mask
        return np.clip(res, 0, 255).astype(np.uint8)
