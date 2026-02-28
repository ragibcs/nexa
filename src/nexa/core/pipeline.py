import cv2
import imageio
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from nexa.models.analyzer import FaceAnalyzer
from nexa.models.swapper import Swapper
from nexa.models.enhancers import FaceEnhancer
from nexa.core.mapping import FaceMapper
from nexa.core.audio import extract_audio, mux_audio_video
from nexa.utils.video import is_video, get_frame_count
from nexa.utils.logging import log_info, log_warn, log_success, log_error, check_ffmpeg


def process_media(
    target_path: Path,
    output_path: Path,
    mappings: list,
    enhancer: Optional[str],
    use_gpu: bool = False,
    similarity_threshold: float = 0.6,
):
    """
    Main entry point.  Detects whether *target_path* is a video or image
    and delegates to the appropriate handler.  All video frames are
    processed in-memory -- no intermediate frame files are created.
    """
    check_ffmpeg()

    log_info("Initialising models ...")
    analyzer = FaceAnalyzer(use_gpu=use_gpu)
    mapper = FaceMapper(analyzer, mappings, threshold=similarity_threshold)
    swapper = Swapper(use_gpu=use_gpu)

    face_enhancer = None
    if enhancer:
        log_info(f"Loading face enhancer: {enhancer}")
        face_enhancer = FaceEnhancer(name=enhancer)

    if is_video(target_path):
        _process_video(target_path, output_path, mapper, analyzer, swapper, face_enhancer)
    else:
        _process_image(target_path, output_path, mapper, analyzer, swapper, face_enhancer)


def _process_video(
    target_path: Path,
    output_path: Path,
    mapper: FaceMapper,
    analyzer: FaceAnalyzer,
    swapper: Swapper,
    enhancer: Optional[FaceEnhancer],
):
    log_info("Processing video in-memory (no frame extraction to disk) ...")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        tmp_audio = tmp_dir / "audio.m4a"
        tmp_video = tmp_dir / "video.mp4"

        log_info("Extracting audio track ...")
        has_audio = extract_audio(target_path, tmp_audio)

        reader = imageio.get_reader(target_path)
        fps = reader.get_meta_data()["fps"]
        writer = imageio.get_writer(tmp_video, fps=fps, codec="libx264", format="FFMPEG")

        total_frames = get_frame_count(target_path) or None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}[/cyan]" if total_frames else ""),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Swapping faces", total=total_frames)

            for frame in reader:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                for face in analyzer.analyze(frame_bgr):
                    src = mapper.get_source_for_target(face)
                    if src:
                        frame_bgr = swapper.swap(frame_bgr, src, face)

                if enhancer:
                    frame_bgr = enhancer.enhance(frame_bgr)

                writer.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                progress.advance(task)

        writer.close()
        reader.close()

        if has_audio:
            log_info("Muxing audio back into output ...")
            if not mux_audio_video(tmp_video, tmp_audio, output_path):
                log_warn("Audio mux failed; saving video without audio.")
                shutil.copy(tmp_video, output_path)
        else:
            log_info("No audio track found in source; copying video.")
            shutil.copy(tmp_video, output_path)

    log_success(f"Video saved to {output_path}")


def _process_image(
    target_path: Path,
    output_path: Path,
    mapper: FaceMapper,
    analyzer: FaceAnalyzer,
    swapper: Swapper,
    enhancer: Optional[FaceEnhancer],
):
    log_info("Processing image ...")
    img = cv2.imread(str(target_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read target image: {target_path}")

    faces = analyzer.analyze(img)
    if not faces:
        log_warn("No faces detected in target image. Saving original.")
        cv2.imwrite(str(output_path), img)
        return

    for face in faces:
        src = mapper.get_source_for_target(face)
        if src:
            img = swapper.swap(img, src, face)

    if enhancer:
        img = enhancer.enhance(img)

    cv2.imwrite(str(output_path), img)
    log_success(f"Image saved to {output_path}")
