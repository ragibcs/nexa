import typer
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated

from nexa.utils.logging import log_info, log_error

app = typer.Typer(
    name="nexa",
    help="Nexa - Enhanced Headless Face Swapper",
    add_completion=False,
    rich_markup_mode="rich",
)


def _validate_path(path: Path, label: str):
    if not path.exists():
        log_error(f"{label} file not found: {path}")
        raise typer.Exit(code=1)


@app.command()
def swap(
    source: Annotated[
        Optional[Path],
        typer.Option("--source", "-s", help="Source face image (single-face mode)"),
    ] = None,
    target: Annotated[
        Path,
        typer.Option("--target", "-t", help="Target video or image to process"),
    ] = ...,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output file path"),
    ] = ...,
    face_map: Annotated[
        Optional[list[str]],
        typer.Option("--map", "-m", help="Multi-face mapping  src.jpg:tgt.jpg  (repeatable)"),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", "-M", help="Base Stable Diffusion 1.5 model ID from HuggingFace (e.g. SG161222/Realistic_Vision_V5.1_noVAE)"),
    ] = "SG161222/Realistic_Vision_V5.1_noVAE",
    steps: Annotated[
        int,
        typer.Option("--steps", help="Number of diffusion inference steps (LCM enables 4-8 steps)"),
    ] = 4,
    enhancer: Annotated[
        Optional[str],
        typer.Option("--enhancer", "-e", help="Face enhancer: gfpgan | codeformer"),
    ] = None,
    gpu: Annotated[
        bool,
        typer.Option("--gpu", help="Use GPU (CUDA) acceleration"),
    ] = False,
    threshold: Annotated[
        float,
        typer.Option("--threshold", help="Cosine-similarity threshold for face matching (0-1)"),
    ] = 0.6,
):
    """Swap faces in a target video or image."""

    # --- validation ---
    if not source and not face_map:
        log_error("Provide either --source (single swap) or --map (multi-face).")
        raise typer.Exit(code=1)

    if source and face_map:
        log_error("Cannot combine --source and --map. Pick one.")
        raise typer.Exit(code=1)

    _validate_path(target, "Target")

    # --- build mapping list ---
    mappings: list[tuple[Path, Optional[Path]]] = []
    if face_map:
        for entry in face_map:
            if ":" not in entry:
                log_error(f"Invalid map format '{entry}'. Expected 'source.jpg:target.jpg'.")
                raise typer.Exit(code=1)
            src_str, tgt_str = entry.split(":", 1)
            sp, tp = Path(src_str), Path(tgt_str)
            _validate_path(sp, "Map source")
            _validate_path(tp, "Map target")
            mappings.append((sp, tp))
    else:
        assert source is not None
        _validate_path(source, "Source")
        mappings = [(source, None)]

    log_info(f"Target  : {target}")
    log_info(f"Output  : {output}")
    log_info(f"Model   : {model}")
    log_info(f"Steps   : {steps}")
    log_info(f"GPU     : {'yes' if gpu else 'no'}")
    if enhancer:
        log_info(f"Enhancer: {enhancer}")

    from nexa.core.pipeline import process_media

    try:
        process_media(
            target_path=target,
            output_path=output,
            mappings=mappings,
            model_id=model,
            steps=steps,
            enhancer=enhancer,
            use_gpu=gpu,
            similarity_threshold=threshold,
        )
    except Exception as e:
        log_error(str(e))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
