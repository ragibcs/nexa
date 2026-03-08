"""Typer CLI entry point for Nexa face swapper."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from nexa.utils.logging import setup_logging, check_ffmpeg
from nexa.utils.video import is_video

console = Console()

app = typer.Typer(
    name="nexa",
    help="Diffusion-based face swapper using IP-Adapter FaceID + SD1.5.",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command()
def main(
    source: Annotated[
        Optional[Path],
        typer.Option("--source", "-s", help="Source face image (single-face mode)."),
    ] = None,
    target: Annotated[
        Path,
        typer.Option("--target", "-t", help="Target image or video."),
    ] = ...,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path."),
    ] = ...,
    map: Annotated[
        Optional[list[str]],
        typer.Option("--map", "-m", help="Multi-face mapping src.jpg:tgt.jpg (repeatable)."),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", "-M", help="HuggingFace SD1.5 model ID."),
    ] = "runwayml/stable-diffusion-v1-5",
    steps: Annotated[
        int,
        typer.Option("--steps", help="Diffusion inference steps (20-35 recommended)."),
    ] = 28,
    enhancer: Annotated[
        Optional[str],
        typer.Option("--enhancer", "-e", help="Face enhancer: gfpgan or codeformer."),
    ] = None,
    gpu: Annotated[
        bool,
        typer.Option("--gpu", help="Use CUDA acceleration."),
    ] = False,
    threshold: Annotated[
        float,
        typer.Option("--threshold", help="Cosine similarity threshold for face matching."),
    ] = 0.6,
    strength: Annotated[
        float,
        typer.Option("--strength", help="Diffusion strength (lower = more realistic, 0=none, 1=full)."),
    ] = 0.45,
    guidance_scale: Annotated[
        float,
        typer.Option("--guidance-scale", help="Classifier-free guidance scale."),
    ] = 4.0,
    ip_scale: Annotated[
        float,
        typer.Option("--ip-scale", help="IP-Adapter face identity strength."),
    ] = 1.0,
) -> None:
    """Run Nexa face swap."""
    setup_logging()
    check_ffmpeg()

    device = "cuda" if gpu else "cpu"

    console.print("[bold]Nexa Face Swapper[/]")
    console.print(f"  Device   : {device}")
    console.print(f"  Model    : {model}")
    console.print(f"  Steps    : {steps}")
    console.print(f"  Strength : {strength}")
    console.print(f"  Guidance : {guidance_scale}")
    console.print(f"  IP Scale : {ip_scale}")
    console.print(f"  Enhancer : {enhancer or 'none'}")
    console.print()

    # Validate inputs
    if source is None and not map:
        console.print("[red]Error: provide --source or --map.[/]")
        raise typer.Exit(1)

    if not target.exists():
        console.print(f"[red]Target not found: {target}[/]")
        raise typer.Exit(1)

    # Lazy import to avoid loading heavy models at CLI parse time
    from nexa.core.pipeline import NexaPipeline

    pipeline = NexaPipeline(
        model_id=model,
        device=device,
        steps=steps,
        enhancer_name=enhancer,
        threshold=threshold,
        ip_scale=ip_scale,
        strength=strength,
        guidance_scale=guidance_scale,
    )

    if map:
        # Multi-face mode
        mappings: dict[str, str] = {}
        for pair in map:
            if ":" not in pair:
                console.print(f"[red]Invalid mapping (expected src:tgt): {pair}[/]")
                raise typer.Exit(1)
            src, tgt = pair.split(":", 1)
            mappings[src] = tgt
        pipeline.process_image_multi(mappings, target, output)
    elif is_video(target):
        # Video mode
        if source is None:
            console.print("[red]Error: --source required for video mode.[/]")
            raise typer.Exit(1)
        pipeline.process_video(source, target, output)
    else:
        # Single image mode
        if source is None:
            console.print("[red]Error: --source required for image mode.[/]")
            raise typer.Exit(1)
        pipeline.process_image_single(source, target, output)


if __name__ == "__main__":
    app()
