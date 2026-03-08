"""Pipeline — orchestrates image / video face-swap processing."""

from __future__ import annotations

import shutil
import tempfile
import traceback
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

from nexa.core.audio import extract_audio, mux_audio
from nexa.core.mapping import FaceMapper
from nexa.models.analyzer import FaceAnalyzer
from nexa.models.swapper import FaceSwapper
from nexa.models.enhancers import get_enhancer
from nexa.utils.video import count_frames, get_fps

console = Console()


class NexaPipeline:
    """End-to-end face-swap pipeline for images and videos."""

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        steps: int = 20,
        enhancer_name: str | None = None,
        threshold: float = 0.6,
        ip_scale: float = 1.0,
        strength: float = 0.65,
        guidance_scale: float = 5.0,
        det_size: int = 640,
        det_score: float = 0.45,
    ) -> None:
        console.print("[bold cyan]Initializing Nexa Pipeline…[/]")

        console.print("[dim]Step 1/3: Loading face analyzer…[/]")
        self.analyzer = FaceAnalyzer(
            device=device,
            det_size=(det_size, det_size),
            det_score=det_score,
        )

        console.print("[dim]Step 2/3: Loading face swapper…[/]")
        self.swapper = FaceSwapper(
            model_id=model_id,
            device=device,
            steps=steps,
            ip_scale=ip_scale,
            strength=strength,
            guidance_scale=guidance_scale,
        )

        console.print("[dim]Step 3/3: Loading enhancer…[/]")
        self.enhancer = get_enhancer(enhancer_name)
        self.mapper = FaceMapper(threshold=threshold)

        console.print("[bold green]Pipeline ready.[/]\n")

    # ── Public API ────────────────────────────────────────────────────────────

    def process_image_single(
        self,
        source_path: str | Path,
        target_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """Single-source face swap on an image."""
        source_img = cv2.imread(str(source_path))
        target_img = cv2.imread(str(target_path))

        if source_img is None:
            raise FileNotFoundError(f"Cannot read source: {source_path}")
        if target_img is None:
            raise FileNotFoundError(f"Cannot read target: {target_path}")

        console.print(f"[dim]Source image: {source_img.shape[1]}x{source_img.shape[0]}[/]")
        console.print(f"[dim]Target image: {target_img.shape[1]}x{target_img.shape[0]}[/]")

        # Detect source face
        console.print("[dim]Detecting source face…[/]")
        src_faces = self.analyzer.detect(source_img)
        if not src_faces:
            raise RuntimeError(f"No face detected in source: {source_path}")
        console.print(f"[dim]Found {len(src_faces)} face(s) in source.[/]")
        src_face = self.analyzer.best_face(src_faces)
        src_emb = self.analyzer.embedding(src_face)

        # Detect target faces
        console.print("[dim]Detecting target face(s)…[/]")
        tgt_faces = self.analyzer.detect(target_img)
        if not tgt_faces:
            console.print("[yellow]No faces detected in target — copying as-is.[/]")
            if not cv2.imwrite(str(output_path), target_img):
                raise RuntimeError(f"Failed to write output image: {output_path}")
            return Path(output_path)
        console.print(f"[dim]Found {len(tgt_faces)} face(s) in target.[/]")

        # Swap each target face
        result = target_img.copy()
        for i, face in enumerate(tgt_faces):
            console.print(f"[bold cyan]Swapping face {i + 1}/{len(tgt_faces)}…[/]")
            try:
                result = self.swapper.swap_face_in_image(result, face, src_emb)
                console.print(f"[green]✓ Face {i + 1} swapped.[/]")
            except Exception as e:
                console.print(f"[red]✗ Face {i + 1} failed: {e}[/]")
                traceback.print_exc()

        # Enhance
        if self.enhancer is not None:
            console.print("[dim]Enhancing faces…[/]")
            try:
                result = self.enhancer.enhance(result)
            except Exception as e:
                console.print(f"[yellow]Enhancement failed: {e}[/]")

        if not cv2.imwrite(str(output_path), result):
            raise RuntimeError(f"Failed to write output image: {output_path}")
        console.print(f"[bold green]Saved:[/] {output_path}")
        return Path(output_path)

    def process_image_multi(
        self,
        mappings: dict[str, str],
        target_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """Multi-source face swap on an image.

        *mappings* is ``{source_path: reference_face_path}``.
        """
        target_img = cv2.imread(str(target_path))
        if target_img is None:
            raise FileNotFoundError(f"Cannot read target: {target_path}")

        # Build source/reference pairs where reference embedding is only used
        # for matching and source embedding is used for swapping.
        source_refs: list[tuple[str, np.ndarray, np.ndarray]] = []
        for src_path, ref_path in mappings.items():
            src_img = cv2.imread(str(src_path))
            ref_img = cv2.imread(str(ref_path))
            if src_img is None:
                raise FileNotFoundError(f"Cannot read source: {src_path}")
            if ref_img is None:
                raise FileNotFoundError(f"Cannot read reference: {ref_path}")

            src_faces = self.analyzer.detect(src_img)
            ref_faces = self.analyzer.detect(ref_img)
            if not src_faces:
                raise RuntimeError(f"No face in source: {src_path}")
            if not ref_faces:
                raise RuntimeError(f"No face in reference: {ref_path}")

            source_refs.append(
                (
                    str(src_path),
                    self.analyzer.embedding(self.analyzer.best_face(src_faces)),
                    self.analyzer.embedding(self.analyzer.best_face(ref_faces)),
                )
            )

        # Detect target faces
        tgt_faces = self.analyzer.detect(target_img)
        if not tgt_faces:
            console.print("[yellow]No faces in target.[/]")
            if not cv2.imwrite(str(output_path), target_img):
                raise RuntimeError(f"Failed to write output image: {output_path}")
            return Path(output_path)

        # Match each target face to the closest reference identity, then swap
        # using that identity's source embedding.
        result = target_img.copy()
        for face_idx, tgt_face in enumerate(tgt_faces):
            best_name: str | None = None
            best_src_emb: np.ndarray | None = None
            best_sim = -1.0

            for src_name, src_emb, ref_emb in source_refs:
                sim = float(np.dot(ref_emb, tgt_face.normed_embedding))
                if sim > best_sim:
                    best_sim = sim
                    best_name = src_name
                    best_src_emb = src_emb

            if best_src_emb is None or best_sim < self.mapper.threshold:
                console.print(
                    f"[dim]Face {face_idx} skipped (best similarity={best_sim:.3f}).[/]"
                )
                continue

            console.print(
                f"[dim]Face {face_idx} → {best_name} (similarity={best_sim:.3f})[/]"
            )
            result = self.swapper.swap_face_in_image(result, tgt_face, best_src_emb)

        if self.enhancer is not None:
            result = self.enhancer.enhance(result)

        if not cv2.imwrite(str(output_path), result):
            raise RuntimeError(f"Failed to write output image: {output_path}")
        console.print(f"[bold green]Saved:[/] {output_path}")
        return Path(output_path)

    def process_video(
        self,
        source_path: str | Path,
        target_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """Single-source face swap on a video."""
        import imageio.v3 as iio

        source_img = cv2.imread(str(source_path))
        if source_img is None:
            raise FileNotFoundError(f"Cannot read source: {source_path}")

        src_faces = self.analyzer.detect(source_img)
        if not src_faces:
            raise RuntimeError(f"No face in source: {source_path}")
        src_face = self.analyzer.best_face(src_faces)
        src_emb = self.analyzer.embedding(src_face)

        target_path = Path(target_path)
        output_path = Path(output_path)

        # Get video metadata
        fps = get_fps(target_path)
        total = count_frames(target_path)
        console.print(f"[dim]Video: {total} frames @ {fps:.1f} FPS[/]")

        # Extract audio and process frames
        tmp_dir = Path(tempfile.mkdtemp(prefix="nexa_"))
        try:
            audio_file = tmp_dir / "audio.aac"
            has_audio = extract_audio(target_path, audio_file)

            tmp_video = tmp_dir / "video_noaudio.mp4"
            writer = None
            try:
                writer = iio.imopen(str(tmp_video), "w", plugin="pyav")
                writer.init_video_stream("libx264", fps=fps)

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total}"),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task("Swapping faces", total=total)

                    for frame_idx, frame in enumerate(iio.imiter(str(target_path), plugin="pyav")):
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        faces = self.analyzer.detect(frame_bgr)
                        if faces:
                            for face_idx, face in enumerate(faces):
                                try:
                                    frame_bgr = self.swapper.swap_face_in_image(
                                        frame_bgr, face, src_emb
                                    )
                                except Exception as e:
                                    console.print(
                                        f"[yellow]Swap failed at frame {frame_idx}, face {face_idx}: {e}[/]"
                                    )

                            if self.enhancer is not None:
                                try:
                                    frame_bgr = self.enhancer.enhance(frame_bgr)
                                except Exception as e:
                                    console.print(
                                        f"[yellow]Enhancer failed at frame {frame_idx}: {e}[/]"
                                    )

                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        writer.write_frame(frame_rgb)
                        progress.advance(task)
            finally:
                if writer is not None:
                    writer.close()

            # Mux audio
            if has_audio:
                mux_audio(tmp_video, audio_file, output_path)
            else:
                shutil.copy2(tmp_video, output_path)

            console.print(f"[bold green]Saved:[/] {output_path}")
            return output_path
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
