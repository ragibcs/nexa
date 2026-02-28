import os
import torch
import numpy as np
import cv2
from PIL import Image

from diffusers import StableDiffusionInpaintPipeline, LCMScheduler
from nexa.models.manager import ensure_hf_models
from nexa.utils.logging import log_info


class Swapper:
    def __init__(self, model_id: str = "SG161222/Realistic_Vision_V5.1_noVAE", use_gpu: bool = False, steps: int = 4):
        """
        Initializes the IP-Adapter FaceID Diffusion Pipeline with LCM for fast inference.
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.steps = steps

        # Ensure all HF models are downloaded
        ensure_hf_models()

        log_info(f"Loading Base SD1.5 Inpainting Pipeline from '{model_id}'...")
        # Since standard models don't have inpainting natively mapped sometimes, we just load
        # the standard weights into an Inpaint Pipeline, it handles it gracefully for img2img inpainting
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
        ).to(self.device)

        log_info("Applying LCM LoRA for 4-step rapid inference...")
        self.pipeline.scheduler = LCMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

        log_info("Loading IP-Adapter-FaceID...")
        # Note: diffusers loads ip adapters by looking at the repo
        try:
            self.pipeline.load_ip_adapter(
                "h94/IP-Adapter-FaceID",
                subfolder="",
                weight_name="ip-adapter-faceid_sd15.bin"
            )
        except Exception:
            # Fallback for diffusers versions that struggle with custom names in the root folder
            # It expects `pytorch_model.bin` or `model.safetensors` by default if weight_name has issues
            try:
                self.pipeline.load_ip_adapter(
                    "h94/IP-Adapter-FaceID",
                    subfolder=None,
                    weight_name="ip-adapter-faceid_sd15.bin"
                )
            except Exception as e:
                log_info(f"Downloading weights manually to bypass diffusers cache bug... ({e})")
                from huggingface_hub import hf_hub_download
                ckpt_path = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid_sd15.bin")
                self.pipeline.load_ip_adapter(os.path.dirname(ckpt_path), subfolder="", weight_name=os.path.basename(ckpt_path))

        # We need to set the scale of the IP-Adapter
        self.pipeline.set_ip_adapter_scale(1.2)

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
        Uses StableDiffusionInpaintPipeline with IP-Adapter-FaceID to generate a new face
        that exactly matches the source identity, and blends it into the original image.
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
        # We use the target_face landmarks to draw a tight polygon around the face features
        mask_512 = np.zeros((512, 512), dtype=np.uint8)

        # Get facial landmarks relative to the cropped 512x512 image
        kps = target_face.landmark_2d_106 if hasattr(target_face, 'landmark_2d_106') else target_face.kps
        if kps is not None:
            # Shift kps to crop coordinates, then scale to 512
            kps_shifted = []
            for pt in kps:
                px = (pt[0] - x1) * (512.0 / orig_crop_w)
                py = (pt[1] - y1) * (512.0 / orig_crop_h)
                kps_shifted.append([px, py])

            kps_shifted = np.array(kps_shifted, dtype=np.int32)

            # Draw convex hull around the face landmarks to define the inpainting region
            hull = cv2.convexHull(kps_shifted)
            cv2.fillConvexPoly(mask_512, hull, 255)

            # Dilate and blur the mask heavily so the diffusion model has freedom to generate new borders
            kernel = np.ones((15, 15), np.uint8)
            mask_512 = cv2.dilate(mask_512, kernel, iterations=2)
            mask_512 = cv2.GaussianBlur(mask_512, (51, 51), 0)
        else:
            # Fallback if no 106 kps available, just do a center circle
            cv2.circle(mask_512, (256, 256), 180, 255, -1)
            mask_512 = cv2.GaussianBlur(mask_512, (51, 51), 0)

        # Threshold to create binary mask for diffusion, but keep a feathered version for compositing later
        composite_mask = mask_512.astype(np.float32) / 255.0
        diff_mask = np.where(mask_512 > 10, 255, 0).astype(np.uint8)

        # 3. Prepare Image/Mask for Diffusers
        init_image = Image.fromarray(crop_512)
        mask_image = Image.fromarray(diff_mask)

        # 4. Extract Source Face Embeds
        # Diffusers expects the embedding to be passed via cross-attention kwargs or image_embeds
        import torch
        emb = torch.tensor(source_face.normed_embedding).view(1, 1, -1)
        faceid_embeds = emb.to(self.device, dtype=self.dtype)

        # 5. Run Diffusion Pipeline (LCM fast inference)
        prompt = "photorealistic portrait, highly detailed, sharp focus, exactly same lighting and expression"
        n_prompt = "cartoon, 3d, animated, blurry, deformed, disfigured, poorly drawn, bad lighting"

        with torch.no_grad():
            try:
                # Diffusers >= 0.27 style
                gen_image = self.pipeline(
                    prompt=prompt,
                    negative_prompt=n_prompt,
                    image=init_image,
                    mask_image=mask_image,
                    ip_adapter_image_embeds=[faceid_embeds],
                    num_inference_steps=self.steps,
                    guidance_scale=1.5,
                    strength=0.99,
                ).images[0]
            except Exception as e:
                # Fallback to single tensor if list fails
                log_info(f"Retrying with un-listed embedding... ({e})")
                gen_image = self.pipeline(
                    prompt=prompt,
                    negative_prompt=n_prompt,
                    image=init_image,
                    mask_image=mask_image,
                    ip_adapter_image_embeds=faceid_embeds,
                    num_inference_steps=self.steps,
                    guidance_scale=1.5,
                    strength=0.99,
                ).images[0]

        gen_rgb = np.array(gen_image)

        # 6. Composite the generated face back into the original crop using the soft mask
        composite_mask_3d = np.stack([composite_mask]*3, axis=2)
        blended_crop_rgb = (gen_rgb * composite_mask_3d + crop_512 * (1.0 - composite_mask_3d)).astype(np.uint8)

        # Resize blended crop back to original crop resolution
        blended_crop_rgb_orig_res = cv2.resize(blended_crop_rgb, (orig_crop_w, orig_crop_h), interpolation=cv2.INTER_LANCZOS4)

        # 7. Paste back into original full image
        img_rgb[y1:y2, x1:x2] = blended_crop_rgb_orig_res

        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)