# Nexa — Diffusion-Based Face Swapper

**IP-Adapter FaceID + Stable Diffusion 1.5** headless CLI tool for photorealistic face swapping in images and videos.

## Features

- **Diffusion-based face swap** using IP-Adapter FaceID (not traditional GAN-based)
- **Single-face** and **multi-face** mapping modes
- **Video processing** with audio preservation
- **Face enhancement** via GFPGAN post-processing (`codeformer` currently routes to GFPGAN fallback)
- **Colab-ready** — runs on free T4 GPU (~6-7 GB VRAM)
- **CLI + Python API** — use from terminal or import in your code

## Architecture

```
nexa/
├── src/nexa/
│   ├── main.py              # Typer CLI entry point
│   ├── core/
│   │   ├── pipeline.py      # Orchestrates image/video processing
│   │   ├── mapping.py       # Maps source→target faces via embeddings
│   │   └── audio.py         # FFmpeg audio extract/mux for video
│   ├── models/
│   │   ├── analyzer.py      # InsightFace buffalo_l face detection
│   │   ├── swapper.py       # IP-Adapter FaceID diffusion engine (CORE)
│   │   ├── enhancers.py     # GFPGAN / CodeFormer post-processing
│   │   └── manager.py       # Model download manager
│   └── utils/
│       ├── video.py         # Video format detection, frame counting
│       └── logging.py       # Rich-based colored logging
├── new.ipynb                # Google Colab notebook
├── pyproject.toml           # Package config with [gpu] extras
├── requirements.txt         # Flat dependency list
└── README.md
```

## Quick Start (Google Colab)

1. **Upload** the `nexa` folder to `/content/` in Colab
2. **Open** `new.ipynb` and follow the cells
3. **Set runtime** to GPU (Runtime → Change runtime type → T4 GPU)
4. Run cells in order:
   - Install dependencies
   - Install FFmpeg
   - Upload source & target images
   - Run face swap

Or use the CLI directly:

```bash
%cd /content/nexa
!pip install -e ".[gpu]"
!nexa --source /content/source.jpg --target /content/target.jpg --output /content/output.jpg --steps 20 --gpu
```

## CLI Usage

```bash
# Single face swap
nexa --source face.jpg --target photo.jpg --output result.jpg --gpu

# Multi-face mapping (source:reference-in-target)
nexa -m alice.jpg:person1.jpg -m bob.jpg:person2.jpg -t group.jpg -o out.jpg --gpu

# Video processing
nexa -s face.jpg -t video.mp4 -o output.mp4 --gpu --enhancer gfpgan

# Custom parameters for better quality
nexa -s face.jpg -t photo.jpg -o result.jpg --gpu \
  --steps 25 --strength 0.6 --guidance-scale 6.0 --ip-scale 1.2
```

### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--source / -s` | None | Source face image (single-face mode) |
| `--target / -t` | *required* | Target image or video |
| `--output / -o` | *required* | Output file path |
| `--map / -m` | None | Multi-face mapping `source.jpg:reference.jpg` (repeatable) |
| `--model / -M` | `runwayml/stable-diffusion-v1-5` | HuggingFace SD1.5 model ID |
| `--steps` | 20 | Diffusion inference steps (15-30 recommended) |
| `--strength` | 0.65 | How much to change the init image (0=none, 1=full) |
| `--guidance-scale` | 5.0 | Classifier-free guidance scale |
| `--ip-scale` | 1.0 | IP-Adapter face identity strength |
| `--enhancer / -e` | None | Face enhancer: `gfpgan` (or `codeformer`, currently using GFPGAN fallback) |
| `--gpu` | False | Use CUDA acceleration |
| `--threshold` | 0.6 | Cosine similarity threshold for multi-face reference matching |

## Python API

```python
from nexa.core.pipeline import NexaPipeline

pipeline = NexaPipeline(
    model_id="runwayml/stable-diffusion-v1-5",
    device="cuda",
    steps=20,
    enhancer_name="gfpgan",
    ip_scale=1.0,
    strength=0.65,
    guidance_scale=5.0,
)

# Single face swap
pipeline.process_image_single("source.jpg", "target.jpg", "output.jpg")

# Video
pipeline.process_video("source.jpg", "video.mp4", "output.mp4")
```

## How It Works

### Face Swap Pipeline (per frame)

1. **Detect faces** in target image using InsightFace (`buffalo_l`)
2. **Extract** 512-d ArcFace embeddings from source and target faces
3. For each face:
   - **Crop** expanded bounding box (1.6×) around target face
   - **Create soft mask** from landmark convex hull (dilated + blurred)
   - **Project** source embedding through FaceIDProjModel → 4 tokens
   - **Concatenate** face tokens with text prompt embeddings
   - **Run SD1.5 img2img** with cropped region as init image
   - **Composite** result back using soft mask (alpha blending)
4. Optionally **enhance** all faces with GFPGAN

### Core Technology

- **IP-Adapter FaceID** — custom implementation with `nn.Linear` attention processors (no deprecated `LoRALinearLayer`)
- **DDIM Scheduler** — 20 steps, guidance scale 5.0, strength 0.65
- **CPU Offload** — keeps peak VRAM under 8 GB on T4

## Quality Tuning

| Parameter | Effect | Recommended |
|-----------|--------|-------------|
| `strength` | How much to change the init image | 0.5-0.7 |
| `guidance_scale` | How strongly to follow the prompt | 4.0-7.0 |
| `steps` | More steps = better quality, slower | 15-30 |
| `ip_scale` | Face identity strength | 0.8-1.5 |

## Memory Budget (T4 — 15 GB VRAM)

| Component | VRAM |
|-----------|------|
| SD1.5 UNet (float16) | ~3.4 GB |
| Text Encoder (float16) | ~0.5 GB |
| VAE (float16) | ~0.3 GB |
| IP-Adapter processors | ~0.2 GB |
| Inference overhead | ~2-3 GB |
| **Total** | **~6-7 GB** |

## Installation (Local)

> Note: dependencies are pinned to `numpy<2` and `onnxruntime<2` for compatibility with the current InsightFace + ONNX runtime stack.

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/nexa.git
cd nexa

# Install (CPU)
pip install -e .

# Install (GPU)
pip uninstall -y onnxruntime
pip install -e ".[gpu]"

# FFmpeg (for video)
sudo apt install ffmpeg
```

## License

This project is for **research purposes only**. InsightFace models are licensed for non-commercial use.
