"""IP-Adapter FaceID diffusion-based face swap engine (CORE).

This module re-implements the IP-Adapter FaceID mechanism faithfully
following the official ``tencent-ailab/IP-Adapter`` repository, but
without depending on the ``ip_adapter`` pip package or the deprecated
``LoRALinearLayer`` import.

Architecture
------------
1. ``MLPProjModel``         — MLP that projects 512-d InsightFace embeddings
   into (batch, num_tokens, cross_attention_dim) for SD1.5.
2. ``LoRALinearLayerSimple``— Drop-in replacement for the removed
   ``diffusers.models.lora.LoRALinearLayer``.
3. ``LoRAAttnProcessor``    — Self-attention processor with LoRA.
4. ``LoRAIPAttnProcessor``  — Cross-attention processor with LoRA + IP-Adapter.
5. ``FaceSwapper``          — High-level class that loads the SD1.5 pipeline,
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
from PIL import Image
from rich.console import Console
from scipy.ndimage import binary_dilation

from nexa.models.manager import download_ip_adapter_faceid

console = Console()

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Drop-in LoRA Linear Layer (replaces deprecated diffusers LoRALinearLayer)
# ═══════════════════════════════════════════════════════════════════════════════

class LoRALinearLayerSimple(nn.Module):
    """Minimal LoRA layer: down-project then up-project."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: float | None = None,
    ) -> None:
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        self.network_alpha = network_alpha
        self.rank = rank
        # Initialize: down with kaiming, up with zeros (standard LoRA init)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down = self.down(x)
        up = self.up(down)
        if self.network_alpha is not None:
            up = up * (self.network_alpha / self.rank)
        return up


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MLPProjModel — matches official checkpoint exactly
# ═══════════════════════════════════════════════════════════════════════════════

class MLPProjModel(nn.Module):
    """Project 512-d ArcFace embedding → (B, num_tokens, cross_attention_dim).

    Architecture matches the official ``MLPProjModel`` in
    ``tencent-ailab/IP-Adapter/ip_adapter/ip_adapter_faceid.py``:
        Linear(512 → 1024) → GELU → Linear(1024 → 768*4)
    """

    def __init__(
        self,
        cross_attention_dim: int = 768,
        id_embeddings_dim: int = 512,
        num_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds: torch.Tensor) -> torch.Tensor:
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Attention Processors — match official checkpoint key structure
# ═══════════════════════════════════════════════════════════════════════════════

class LoRAAttnProcessor(nn.Module):
    """Self-attention processor with LoRA (replaces LoRAAttnProcessor2_0).

    Parameters in state_dict:
        to_q_lora.down.weight, to_q_lora.up.weight,
        to_k_lora.down.weight, to_k_lora.up.weight,
        to_v_lora.down.weight, to_v_lora.up.weight,
        to_out_lora.down.weight, to_out_lora.up.weight
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int | None = None,
        rank: int = 128,
        network_alpha: float | None = None,
        lora_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.lora_scale = lora_scale
        self.to_q_lora = LoRALinearLayerSimple(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayerSimple(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayerSimple(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayerSimple(hidden_size, hidden_size, rank, network_alpha)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) + self.lora_scale * self.to_q_lora(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states) + self.lora_scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + self.lora_scale * self.to_v_lora(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj + LoRA
        hidden_states = attn.to_out[0](hidden_states) + self.lora_scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class LoRAIPAttnProcessor(nn.Module):
    """Cross-attention processor with LoRA + IP-Adapter face tokens.

    Parameters in state_dict:
        to_q_lora.down.weight, to_q_lora.up.weight,
        to_k_lora.down.weight, to_k_lora.up.weight,
        to_v_lora.down.weight, to_v_lora.up.weight,
        to_out_lora.down.weight, to_out_lora.up.weight,
        to_k_ip.weight, to_v_ip.weight
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int | None = None,
        rank: int = 128,
        network_alpha: float | None = None,
        lora_scale: float = 1.0,
        scale: float = 1.0,
        num_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.lora_scale = lora_scale
        self.num_tokens = num_tokens
        self.scale = scale

        self.to_q_lora = LoRALinearLayerSimple(hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayerSimple(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayerSimple(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayerSimple(hidden_size, hidden_size, rank, network_alpha)

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) + self.lora_scale * self.to_q_lora(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # Split text tokens and IP tokens
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # Standard text cross-attention with LoRA
        key = attn.to_k(encoder_hidden_states) + self.lora_scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + self.lora_scale * self.to_v_lora(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # IP-Adapter attention
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        ip_hidden_states = F.scaled_dot_product_attention(
            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_hidden_states = ip_hidden_states.to(query.dtype)

        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj + LoRA
        hidden_states = attn.to_out[0](hidden_states) + self.lora_scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FaceSwapper — high-level swap engine
# ═══════════════════════════════════════════════════════════════════════════════

class FaceSwapper:
    """Loads SD1.5 + IP-Adapter FaceID and exposes ``swap()``."""

    LORA_RANK = 128  # must match the checkpoint

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

        # Move pipe to device FIRST (official does this)
        self.pipe = self.pipe.to(self.device)
        console.print(f"[green]Pipeline loaded on {self.device}.[/]")

        # ── Build IP-Adapter components ──────────────────────────────────────
        cross_dim = self.pipe.unet.config.cross_attention_dim  # 768 for SD1.5

        # Set attention processors (LoRA + IP-Adapter)
        self._set_ip_adapter_processors(cross_dim)

        # Image projection model
        self.image_proj_model = MLPProjModel(
            cross_attention_dim=cross_dim,
            id_embeddings_dim=512,
            num_tokens=num_tokens,
        ).to(self.device, dtype=self.dtype)

        # Load weights from checkpoint
        self._load_ip_adapter_weights()

        console.print("[bold green]FaceSwapper ready.[/]")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _set_ip_adapter_processors(self, cross_dim: int) -> None:
        """Assign LoRA + IP-Adapter processors exactly like the official code."""
        unet = self.pipe.unet
        attn_procs: dict[str, nn.Module] = {}

        for name in unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            )

            # Resolve hidden_size from block config (official method)
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                # Fallback — shouldn't happen for SD1.5
                hidden_size = cross_dim

            if cross_attention_dim is None:
                # Self-attention → LoRA only
                attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=self.LORA_RANK,
                ).to(self.device, dtype=self.dtype)
            else:
                # Cross-attention → LoRA + IP-Adapter
                attn_procs[name] = LoRAIPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=self.LORA_RANK,
                    scale=self.ip_scale,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=self.dtype)

        unet.set_attn_processor(attn_procs)

        n_ip = sum(1 for v in attn_procs.values() if isinstance(v, LoRAIPAttnProcessor))
        n_lora = sum(1 for v in attn_procs.values() if isinstance(v, LoRAAttnProcessor))
        console.print(
            f"[dim]Installed {n_lora} LoRA self-attn + {n_ip} LoRA+IP cross-attn processors.[/]"
        )

    def _load_ip_adapter_weights(self) -> None:
        """Download and load ``ip-adapter-faceid_sd15.bin`` weights.

        The official loading strategy:
            ip_layers = nn.ModuleList(pipe.unet.attn_processors.values())
            ip_layers.load_state_dict(state_dict["ip_adapter"])
        This means the state_dict keys are indexed over ALL processors
        (both self-attn and cross-attn) in order.
        """
        ckpt_path = download_ip_adapter_faceid()
        console.print(f"[dim]Loading checkpoint: {ckpt_path}[/]")

        state_dict = torch.load(str(ckpt_path), map_location="cpu")

        # 1. image_proj weights
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        console.print("[dim]✓ Loaded image_proj weights.[/]")

        # 2. ip_adapter weights → load into ALL processors (official method)
        ip_layers = nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])
        console.print(f"[dim]✓ Loaded ip_adapter weights into {len(ip_layers)} processors.[/]")

    def set_scale(self, scale: float) -> None:
        """Update the IP-Adapter scale on all cross-attention processors."""
        for proc in self.pipe.unet.attn_processors.values():
            if isinstance(proc, LoRAIPAttnProcessor):
                proc.scale = scale

    @torch.inference_mode()
    def get_image_embeds(
        self, faceid_embeds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project face embedding and create unconditional (zero) embedding."""
        faceid_embeds = faceid_embeds.to(self.device, dtype=self.dtype)
        image_prompt_embeds = self.image_proj_model(faceid_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(faceid_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    # ── Public API ────────────────────────────────────────────────────────────

    def swap(
        self,
        target_crop: np.ndarray,
        source_embedding: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run face swap on a crop using img2img.

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

        # Get face embeddings
        emb = torch.from_numpy(source_embedding).unsqueeze(0)
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(emb)

        # Encode text prompts
        prompt = "a high quality, detailed, photorealistic face"
        neg_prompt = (
            "monochrome, lowres, bad anatomy, worst quality, low quality, "
            "blurry, deformed, disfigured, extra limbs, missing limbs"
        )

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=neg_prompt,
            )
            # Concatenate face tokens with text prompt embeddings
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat(
                [negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1
            )

        # Run img2img
        with torch.inference_mode():
            result = self.pipe(
                image=crop_pil,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
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
            pts = lm.copy()
            pts[:, 0] -= ox
            pts[:, 1] -= oy
            pts = pts.astype(np.int32)
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

        if dilate_iters > 0:
            mask = binary_dilation(mask > 0.5, iterations=dilate_iters).astype(np.float32)

        mask = cv2.GaussianBlur(mask, blur_kernel, 0)
        return mask
