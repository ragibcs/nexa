import os
import torch
import numpy as np
import cv2
from PIL import Image

from diffusers import StableDiffusionInpaintPipeline, LCMScheduler
from huggingface_hub import hf_hub_download
from nexa.models.manager import ensure_hf_models
from nexa.utils.logging import log_info


class Swapper:
    def __init__(self, model_id: str = "SG161222/Realistic_Vision_V5.1_noVAE", use_gpu: bool = False, steps: int = 4):
        """
        Initializes the IP-Adapter FaceID Diffusion Pipeline with LCM for fast inference.
        Uses the official ip_adapter package for reliable FaceID weight loading.
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.steps = steps

        # Ensure all HF models are downloaded
        ensure_hf_models()

        log_info(f"Loading Base SD1.5 Inpainting Pipeline from '{model_id}'...")
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
        ).to(self.device)

        log_info("Applying LCM LoRA for 4-step rapid inference...")
        self.pipeline.scheduler = LCMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        # Fuse LoRA into base model weights so IP-Adapter attention processors don't conflict
        self.pipeline.fuse_lora()

        log_info("Loading IP-Adapter-FaceID via ip_adapter package...")
        ip_ckpt = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid_sd15.bin")

        from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
        self.ip_model = IPAdapterFaceID(self.pipeline, ip_ckpt, self.device, self.dtype)

        # Keep things memory efficient
        if self.device.type == "cuda":
            self.pipeline.enable_model_cpu_offload()

    def _get_expanded_bbox(self, face, img_shape, expand_ratio=1.5):
        """
        Returns an expanded square bounding box around the detected face.
        expand_ratio = 1.0 means tight bounding box, 1.5 adds context.
        """
        bbox = face.bbox
        x1, y1, x2, y2 = bbox

        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2

        # Make it a square crop based on the max dimension
        size = max(w, h) * expand_ratio

        nx1 = max(0, int(cx - size / 2))
        ny1 = max(0, int(cy - size / 2))
        nx2 = min(img_shape[1], int(cx + size / 2))
        ny2 = min(img_shape[0], int(cy + size / 2))

        return nx1, ny1, nx2, ny2

    def swap(self, img: np.ndarray, source_face, target_face) -> np.ndarray:
        """
        Diffusion-based Face Swap.
        Uses IP-Adapter-FaceID to generate a new face that matches the source identity,
        and blends it into the original image via inpainting.
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. Get cropped target region
        x1, y1, x2, y2 = self._get_expanded_bbox(target_face, img.shape, expand_ratio=1.6)
        crop_rgb = img_rgb[y1:y2, x1:x2]

        # Resize to standard 512x512 for SD1.5
        orig_crop_h, orig_crop_w = crop_rgb.shape[:2]
        if orig_crop_h == 0 or orig_crop_w == 0:
            return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        crop_512 = cv2.resize(crop_rgb, (512, 512), interpolation=cv2.INTER_LANCZOS4)

        # 2. Create Inpainting Mask (White for the face we want to replace, Black for background)
        mask_512 = np.zeros((512, 512), dtype=np.uint8)

        kps = target_face.landmark_2d_106 if hasattr(target_face, 'landmark_2d_106') else target_face.kps
        if kps is not None:
            kps_shifted = []
            for pt in kps:
                px = (pt[0] - x1) * (512.0 / orig_crop_w)
                py = (pt[1] - y1) * (512.0 / orig_crop_h)
                kps_shifted.append([px, py])

            kps_shifted = np.array(kps_shifted, dtype=np.int32)

            hull = cv2.convexHull(kps_shifted)
            cv2.fillConvexPoly(mask_512, hull, 255)

            kernel = np.ones((15, 15), np.uint8)
            mask_512 = cv2.dilate(mask_512, kernel, iterations=2)
            mask_512 = cv2.GaussianBlur(mask_512, (51, 51), 0)
        else:
            cv2.circle(mask_512, (256, 256), 180, 255, -1)
            mask_512 = cv2.GaussianBlur(mask_512, (51, 51), 0)

        composite_mask = mask_512.astype(np.float32) / 255.0
        diff_mask = np.where(mask_512 > 10, 255, 0).astype(np.uint8)

        # 3. Prepare Image/Mask for Diffusers
        init_image = Image.fromarray(crop_512)
        mask_image = Image.fromarray(diff_mask)

        # 4. Extract Source Face Embeds
        faceid_embeds = torch.from_numpy(source_face.normed_embedding).unsqueeze(0).to(self.device, dtype=self.dtype)

        # 5. Run Diffusion Pipeline via IP-Adapter FaceID
        prompt = "photorealistic portrait, highly detailed, sharp focus, exactly same lighting and expression"
        n_prompt = "cartoon, 3d, animated, blurry, deformed, disfigured, poorly drawn, bad lighting"

        images = self.ip_model.generate(
            prompt=prompt,
            negative_prompt=n_prompt,
            faceid_embeds=faceid_embeds,
            num_samples=1,
            num_inference_steps=self.steps,
            guidance_scale=1.5,
            image=init_image,
            mask_image=mask_image,
            strength=0.99,
        )

        gen_rgb = np.array(images[0])

        # 6. Composite the generated face back into the original crop using the soft mask
        composite_mask_3d = np.stack([composite_mask]*3, axis=2)
        blended_crop_rgb = (gen_rgb * composite_mask_3d + crop_512 * (1.0 - composite_mask_3d)).astype(np.uint8)

        # Resize blended crop back to original crop resolution
        blended_crop_rgb_orig_res = cv2.resize(blended_crop_rgb, (orig_crop_w, orig_crop_h), interpolation=cv2.INTER_LANCZOS4)

        # 7. Paste back into original full image
        img_rgb[y1:y2, x1:x2] = blended_crop_rgb_orig_res

        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
