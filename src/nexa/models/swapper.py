import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from huggingface_hub import hf_hub_download
from nexa.models.manager import ensure_hf_models
from nexa.utils.logging import log_info


# ---------------------------------------------------------------------------
# IP-Adapter FaceID components (self-contained, no external ip_adapter dep)
# ---------------------------------------------------------------------------

class FaceIDProjModel(nn.Module):
    """Projects InsightFace 512-d embeddings into cross-attention token space."""

    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x


class IPAttnProcessor(nn.Module):
    """Cross-attention processor that adds IP-Adapter face identity signals."""

    def __init__(self, hidden_size, cross_attention_dim, scale=1.0, num_tokens=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        ip_hidden_states = None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # Split off IP tokens appended at the end
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Standard cross-attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # IP-Adapter face identity attention
        if ip_hidden_states is not None:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)
            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_out = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            ip_out = ip_out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states + self.scale * ip_out

        # Output projection + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ---------------------------------------------------------------------------
# Main Swapper
# ---------------------------------------------------------------------------

class Swapper:
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5",
                 use_gpu: bool = False, steps: int = 20):
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.steps = steps
        self.num_tokens = 4

        ensure_hf_models()

        log_info(f"Loading SD1.5 Pipeline from '{model_id}'...")
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=self.dtype, safety_checker=None,
        )
        self.pipeline.scheduler = DDIMScheduler.from_config(
            self.pipeline.scheduler.config
        )
        self.pipeline = self.pipeline.to(self.device)

        log_info("Loading IP-Adapter-FaceID (built-in loader)...")
        self._setup_ip_adapter()

        if self.device.type == "cuda":
            self.pipeline.enable_model_cpu_offload()

    # ---- IP-Adapter setup (no external package) ----

    def _setup_ip_adapter(self):
        ip_ckpt = hf_hub_download(
            repo_id="h94/IP-Adapter-FaceID",
            filename="ip-adapter-faceid_sd15.bin",
        )

        unet = self.pipeline.unet
        ca_dim = unet.config.cross_attention_dim  # 768 for SD1.5

        # 1. Image projection model
        self.image_proj_model = FaceIDProjModel(
            cross_attention_dim=ca_dim,
            id_embeddings_dim=512,
            num_tokens=self.num_tokens,
        )

        # 2. Build attention processors
        attn_procs = {}
        for name in unet.attn_processors.keys():
            if name.endswith("attn1.processor"):
                # Self-attention → standard processor
                attn_procs[name] = AttnProcessor2_0()
            else:
                # Cross-attention → IP-Adapter processor
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]
                else:
                    attn_procs[name] = AttnProcessor2_0()
                    continue

                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=ca_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                )

        unet.set_attn_processor(attn_procs)

        # 3. Load pre-trained weights
        state_dict = torch.load(ip_ckpt, map_location="cpu")

        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.image_proj_model = self.image_proj_model.to(
            self.device, dtype=self.dtype
        )

        ip_layers = nn.ModuleList(unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

        # Move IP processors to correct device
        for proc in unet.attn_processors.values():
            if isinstance(proc, IPAttnProcessor):
                proc.to(self.device, dtype=self.dtype)

        log_info("IP-Adapter-FaceID loaded successfully.")

    # ---- Helpers ----

    def _get_expanded_bbox(self, face, img_shape, expand_ratio=1.5):
        bbox = face.bbox
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        size = max(w, h) * expand_ratio
        nx1 = max(0, int(cx - size / 2))
        ny1 = max(0, int(cy - size / 2))
        nx2 = min(img_shape[1], int(cx + size / 2))
        ny2 = min(img_shape[0], int(cy + size / 2))
        return nx1, ny1, nx2, ny2

    # ---- Main swap ----

    def swap(self, img: np.ndarray, source_face, target_face) -> np.ndarray:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. Crop target face region
        x1, y1, x2, y2 = self._get_expanded_bbox(target_face, img.shape, 1.6)
        crop_rgb = img_rgb[y1:y2, x1:x2]
        orig_h, orig_w = crop_rgb.shape[:2]
        if orig_h == 0 or orig_w == 0:
            return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        crop_512 = cv2.resize(crop_rgb, (512, 512), interpolation=cv2.INTER_LANCZOS4)

        # 2. Face mask for compositing
        mask_512 = np.zeros((512, 512), dtype=np.uint8)
        kps = (target_face.landmark_2d_106
               if hasattr(target_face, 'landmark_2d_106') else target_face.kps)
        if kps is not None:
            pts = np.array([
                [(pt[0] - x1) * 512.0 / orig_w, (pt[1] - y1) * 512.0 / orig_h]
                for pt in kps
            ], dtype=np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask_512, hull, 255)
            mask_512 = cv2.dilate(mask_512, np.ones((15, 15), np.uint8), iterations=2)
            mask_512 = cv2.GaussianBlur(mask_512, (51, 51), 0)
        else:
            cv2.circle(mask_512, (256, 256), 180, 255, -1)
            mask_512 = cv2.GaussianBlur(mask_512, (51, 51), 0)
        composite_mask = mask_512.astype(np.float32) / 255.0

        # 3. Project source face embedding
        faceid_embeds = (
            torch.from_numpy(source_face.normed_embedding)
            .unsqueeze(0)
            .to(self.device, dtype=self.dtype)
        )
        with torch.no_grad():
            image_prompt_embeds = self.image_proj_model(faceid_embeds)
            uncond_image_prompt_embeds = self.image_proj_model(
                torch.zeros_like(faceid_embeds)
            )

        # 4. Get text prompt embeddings
        prompt = "photorealistic portrait, highly detailed, sharp focus, same lighting"
        n_prompt = "cartoon, 3d, animated, blurry, deformed, disfigured, poorly drawn"

        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=n_prompt,
        )

        # Concatenate face identity tokens with text tokens
        prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat(
            [negative_prompt_embeds, uncond_image_prompt_embeds], dim=1
        )

        # 5. Run img2img diffusion
        init_image = Image.fromarray(crop_512)

        with torch.no_grad():
            result = self.pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image=init_image,
                strength=0.65,
                guidance_scale=5.0,
                num_inference_steps=self.steps,
            ).images[0]

        gen_rgb = np.array(result)

        # 6. Composite with soft mask
        m3 = np.stack([composite_mask] * 3, axis=2)
        blended = (gen_rgb * m3 + crop_512 * (1.0 - m3)).astype(np.uint8)
        blended = cv2.resize(blended, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

        # 7. Paste back
        img_rgb[y1:y2, x1:x2] = blended
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
