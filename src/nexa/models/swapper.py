"""IP-Adapter FaceID diffusion-based face swap engine (CORE).

This module implements the IP-Adapter FaceID mechanism **directly** using
custom ``nn.Module`` classes, avoiding the deprecated ``ip_adapter`` pip
package and the version-fragile ``pipeline.load_ip_adapter()`` API.

Architecture
------------
1. ``FaceIDProjModel``  — MLP that projects 512-d InsightFace embeddings
   into (batch, num_tokens, cross_attention_dim) for SD1.5.
2. ``IPAttnProcessor``  — Custom cross-attention processor that adds
   scaled IP-Adapter attention using plain ``nn.Linear`` layers.
3. ``FaceSwapper``      — High-level class that loads the SD1.5 pipeline,
   wires up the IP-Adapter processors, and exposes a ``swap()`` method.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionImg2ImgPipeline,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from PIL import Image
from rich.console import Console
from scipy.ndimage import binary_dilation

from nexa.models.manager import download_ip_adapter_faceid

console = Console()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. FaceID Projection Model
# ═══════════════════════════════════════════════════════════════════════════════

class FaceIDProjModel(nn.Module):
    """Project 512-d ArcFace embedding → (B, num_tokens, cross_attention_dim)."""

    def __init__(
        self,
        embedding_dim: int = 512,
        cross_attention_dim: int = 768,
        num_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, cross_attention_dim),
            nn.GELU(),
            nn.Linear(cross_attention_dim, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 512)
        x = self.proj(x)                                       # (B, 768*4)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)  # (B, 4, 768)
        x = self.norm(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 2. IP-Adapter Cross-Attention Processor
# ═══════════════════════════════════════════════════════════════════════════════

class IPAttnProcessor(nn.Module):
    """Cross-attention processor that adds IP-Adapter face identity tokens.

    Uses **only** ``nn.Linear`` (no deprecated ``LoRALinearLayer``).
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        num_tokens: int = 4,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.scale = scale

        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask=None,
        temb=None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # ── Split text tokens and IP tokens ──────────────────────────────────
        end_pos = encoder_hidden_states.shape[1] - self.num_tokens
        text_hs = encoder_hidden_states[:, :end_pos, :]
        ip_hs = encoder_hidden_states[:, end_pos:, :]

        # Standard text cross-attention
        key = attn.to_k(text_hs)
        value = attn.to_v(text_hs)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # IP-Adapter attention
        ip_key = self.to_k_ip(ip_hs)
        ip_value = self.to_v_ip(ip_hs)

        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        ip_hidden = F.scaled_dot_product_attention(
            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states + self.scale * ip_hidden

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Linear projection + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FaceSwapper — high-level swap engine
# ═══════════════════════════════════════════════════════════════════════════════

class FaceSwapper:
    """Loads SD1.5 + IP-Adapter FaceID and exposes ``swap()``."""

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        num_tokens: int = 4,
        ip_scale: float = 1.0,
        steps: int = 20,
        guidance_scale: float = 5.0,
        strength: float = 0.65,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.num_tokens = num_tokens
        self.ip_scale = ip_scale
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.strength = strength

        console.print(f"[bold cyan]Loading SD1.5 pipeline:[/] {model_id}")
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
        )

        # ── Build IP-Adapter components ──────────────────────────────────────
        cross_dim = self.pipe.unet.config.cross_attention_dim  # 768 for SD1.5
        self.image_proj_model = FaceIDProjModel(
            embedding_dim=512,
            cross_attention_dim=cross_dim,
            num_tokens=num_tokens,
        )

        # Set attention processors
        self._set_ip_adapter_processors(cross_dim)

        # Load weights
        self._load_ip_adapter_weights()

        # Move image_proj_model to device
        self.image_proj_model = self.image_proj_model.to(self.device, dtype=self.dtype)

        # Enable CPU offload AFTER processor setup
        if self.device.type == "cuda":
            self.pipe.enable_model_cpu_offload()
            console.print("[green]CPU offload enabled.[/]")

        # Pre-encode the text prompts (reused for every swap)
        self._encode_text_prompts()

        console.print("[bold green]FaceSwapper ready.[/]")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _set_ip_adapter_processors(self, cross_dim: int) -> None:
        """Assign IPAttnProcessor to every cross-attention layer in UNet."""
        attn_procs: dict[str, nn.Module] = {}
        unet = self.pipe.unet
        for name in unet.attn_processors.keys():
            if name.endswith("attn1.processor"):
                # Self-attention → standard processor
                attn_procs[name] = AttnProcessor2_0()
            else:
                # Cross-attention → IP-Adapter processor
                # Determine hidden_size from the layer
                parts = name.split(".")
                # Walk the module tree to find hidden_size
                hidden_size = self._get_hidden_size(unet, parts)
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_dim,
                    num_tokens=self.num_tokens,
                    scale=self.ip_scale,
                )
        unet.set_attn_processor(attn_procs)
        console.print(
            f"[dim]Installed {sum(1 for v in attn_procs.values() if isinstance(v, IPAttnProcessor))} "
            f"IPAttnProcessors.[/]"
        )

    @staticmethod
    def _get_hidden_size(unet: nn.Module, name_parts: list[str]) -> int:
        """Resolve the hidden_size of a cross-attention layer from its name."""
        # Navigate to the parent attention module
        module = unet
        # name_parts looks like: ['down_blocks', '0', 'attentions', '0', 'transformer_blocks', '0', 'attn2', 'processor']
        for part in name_parts[:-1]:  # skip 'processor'
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        # module is now the Attention layer — inner_dim == to_q.out_features
        return module.to_q.out_features

    def _load_ip_adapter_weights(self) -> None:
        """Download and load ``ip-adapter-faceid_sd15.bin`` weights."""
        ckpt_path = download_ip_adapter_faceid()
        state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)

        # 1. image_proj weights
        proj_sd = state_dict["image_proj"]
        # The checkpoint keys may be flat — map them
        new_proj_sd = {}
        for k, v in proj_sd.items():
            new_proj_sd[k] = v
        self.image_proj_model.load_state_dict(new_proj_sd, strict=False)
        console.print("[dim]Loaded image_proj weights.[/]")

        # 2. ip_adapter weights → load into cross-attention processors
        ip_sd = state_dict["ip_adapter"]
        # ip_sd keys are like "0.to_k_ip.weight", "0.to_v_ip.weight", …
        # Collect all IPAttnProcessors in order
        ip_processors = nn.ModuleList(
            [
                proc
                for proc in self.pipe.unet.attn_processors.values()
                if isinstance(proc, IPAttnProcessor)
            ]
        )
        ip_processors.load_state_dict(ip_sd, strict=False)
        console.print(f"[dim]Loaded ip_adapter weights into {len(ip_processors)} processors.[/]")

    def _encode_text_prompts(self) -> None:
        """Pre-encode the text prompt and negative prompt."""
        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder

        # Move text encoder to device temporarily
        text_encoder = text_encoder.to(self.device)

        prompt = "a high quality, detailed, photorealistic face"
        neg_prompt = (
            "monochrome, lowres, bad anatomy, worst quality, low quality, "
            "blurry, deformed, disfigured, extra limbs, missing limbs"
        )

        with torch.no_grad():
            tok = tokenizer(
                prompt, padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            )
            self._prompt_embeds = text_encoder(tok.input_ids.to(self.device))[0]

            tok_neg = tokenizer(
                neg_prompt, padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            )
            self._neg_prompt_embeds = text_encoder(tok_neg.input_ids.to(self.device))[0]

        # Move text encoder back to CPU to free VRAM
        text_encoder = text_encoder.to("cpu")
        torch.cuda.empty_cache() if self.device.type == "cuda" else None

    # ── Public API ────────────────────────────────────────────────────────────

    def swap(
        self,
        target_crop: np.ndarray,
        source_embedding: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run face swap on a 512x512 crop.

        Parameters
        ----------
        target_crop : np.ndarray
            BGR image crop (H, W, 3) — will be resized to 512x512.
        source_embedding : np.ndarray
            512-d normed ArcFace embedding of the source face.
        mask : np.ndarray | None
            Optional soft mask (H, W) float32 [0, 1].

        Returns
        -------
        np.ndarray
            BGR result image at 512x512.
        """
        # Prepare init image
        crop_rgb = cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb).resize((512, 512), Image.LANCZOS)

        # Project face embedding
        emb = torch.from_numpy(source_embedding).unsqueeze(0).to(self.device, dtype=self.dtype)
        with torch.no_grad():
            face_tokens = self.image_proj_model(emb)  # (1, 4, 768)

        # Concatenate face tokens with text prompt embeddings
        prompt_embeds = torch.cat([self._prompt_embeds, face_tokens], dim=1)
        # For negative prompt, pad with zeros
        neg_pad = torch.zeros_like(face_tokens)
        neg_prompt_embeds = torch.cat([self._neg_prompt_embeds, neg_pad], dim=1)

        # Run img2img
        with torch.no_grad():
            result = self.pipe(
                image=crop_pil,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_prompt_embeds,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance_scale,
                strength=self.strength,
                output_type="pil",
            ).images[0]

        # Convert back to BGR numpy
        result_np = np.array(result)
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
        return result_bgr

    def swap_face_in_image(
        self,
        full_image: np.ndarray,
        face,
        source_embedding: np.ndarray,
        expand_ratio: float = 1.6,
        mask_dilate: int = 2,
        mask_blur: tuple[int, int] = (51, 51),
    ) -> np.ndarray:
        """Swap a single face in a full image.

        Parameters
        ----------
        full_image : np.ndarray
            Full BGR image.
        face
            InsightFace ``Face`` object detected in the target image.
        source_embedding : np.ndarray
            512-d normed embedding of the source identity.
        expand_ratio : float
            How much to expand the bounding box (1.0 = tight crop).
        mask_dilate : int
            Number of dilation iterations for the face mask.
        mask_blur : tuple[int, int]
            Gaussian blur kernel for the soft mask boundary.

        Returns
        -------
        np.ndarray
            Full image with the face swapped.
        """
        h, w = full_image.shape[:2]
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        # Expand bounding box
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half_w = int(bw * expand_ratio / 2)
        half_h = int(bh * expand_ratio / 2)
        ex1 = max(0, cx - half_w)
        ey1 = max(0, cy - half_h)
        ex2 = min(w, cx + half_w)
        ey2 = min(h, cy + half_h)

        crop = full_image[ey1:ey2, ex1:ex2].copy()
        crop_h, crop_w = crop.shape[:2]

        # Create soft mask from landmarks
        mask = self._create_face_mask(
            face, (crop_h, crop_w), (ex1, ey1), mask_dilate, mask_blur
        )

        # Run diffusion swap
        swapped = self.swap(crop, source_embedding, mask)

        # Resize swapped back to crop dimensions
        swapped = cv2.resize(swapped, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)

        # Alpha-blend using mask
        mask_3ch = np.stack([mask] * 3, axis=-1)
        blended = (swapped.astype(np.float32) * mask_3ch +
                   crop.astype(np.float32) * (1.0 - mask_3ch))
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # Paste back
        result = full_image.copy()
        result[ey1:ey2, ex1:ex2] = blended
        return result

    @staticmethod
    def _create_face_mask(
        face,
        crop_shape: tuple[int, int],
        crop_offset: tuple[int, int],
        dilate_iters: int = 2,
        blur_kernel: tuple[int, int] = (51, 51),
    ) -> np.ndarray:
        """Create a soft face mask from 106-point landmarks."""
        crop_h, crop_w = crop_shape
        ox, oy = crop_offset
        mask = np.zeros((crop_h, crop_w), dtype=np.float32)

        lm = getattr(face, "landmark_2d_106", None)
        if lm is None:
            lm = getattr(face, "landmark_2d_68", None)

        if lm is not None:
            # Offset landmarks to crop coordinates
            pts = lm.copy()
            pts[:, 0] -= ox
            pts[:, 1] -= oy
            pts = pts.astype(np.int32)

            # Convex hull
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 1.0)
        else:
            # Fallback: use bbox-based ellipse mask
            bbox = face.bbox.astype(int)
            bx1, by1, bx2, by2 = bbox
            cx = (bx1 + bx2) // 2 - ox
            cy = (by1 + by2) // 2 - oy
            rw = (bx2 - bx1) // 2
            rh = (by2 - by1) // 2
            cv2.ellipse(mask, (cx, cy), (rw, rh), 0, 0, 360, 1.0, -1)

        # Dilate
        if dilate_iters > 0:
            mask = binary_dilation(mask > 0.5, iterations=dilate_iters).astype(np.float32)

        # Gaussian blur for soft edges
        mask = cv2.GaussianBlur(mask, blur_kernel, 0)
        return mask
