<div align="center">

# Nexa 🎭

**Improved Headless Face Swapper**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Typer CLI](https://img.shields.io/badge/CLI-Typer-brightgreen.svg)](https://typer.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/Backend-ONNXRuntime-blueviolet.svg)](https://onnxruntime.ai/)

A modern, quick, and completely headless command-line tool for swapping faces. Nexa is a new version of the old "roop" project. It was built from the ground up to get rid of huge disk I/O bottlenecks by processing video streams completely in memory. It also has advanced features like being able to map multiple faces in one video.

</div>

---

## ✨ Main Features

- **🚀 In-Memory Processing**: Unlike older tools that save thousands of images to your hard drive, Nexa streams video frames directly into memory, processes them, and writes them out. No more SSD thrashing.
- **👯 Multi-Face Mapping**: Do you want to switch out a bunch of different people in the same video? Nexa can do specific source-to-target face mapping using cosine similarity.
- **💻 Fully Headless**: Made to be a command line interface (CLI) tool. Simple to add to scripts, pipelines, or run on remote servers or Colab without needing a user interface.
- **🔊 Audio Preservation**: Automatically takes the original audio track and adds it back into the final video that has been swapped.
- **📦 Automatic Model Management**: When you first run the program, it automatically downloads the models it needs (InsightFace, Inswapper) and stores them in memory.

## ⚙️ How It Works

Nexa uses `imageio` and `ffmpeg` to handle video streams quickly and easily. It uses the `insightface` (buffalo_l model) to find and recognize faces. The `inswapper_128.onnx` model running on `onnxruntime` does the actual face swapping.

---

## 💻 Installation (Local)

### What You Need

- Python 3.10 or higher
- FFmpeg is installed and in your system PATH (`sudo apt install ffmpeg` or `brew install ffmpeg`)

### Use pip (or uv) to install

1. Copy the repository:
   ```bash
   git clone https://github.com/ragibcs/nexa.git
   cd nexa
   ```

2. Set up the package:
   ```bash
   pip install -e .
   # OR with uv:
   uv pip install -e .
   ```

*(Note: For hardware acceleration, you might want to install the `onnxruntime-gpu` package instead of the default CPU version, depending on your hardware.)*

---

## 🚀 How to Use

You can only control Nexa from the command line.

### Simple Face Swap

Use the following command to replace all the faces in a target video or image with a single source face:

```bash
nexa --source person.jpg --target input_video.mp4 --output swapped_video.mp4
```

### Mapping Multiple Faces

In a video with more than one person, tell exactly which source face goes on which target face:

```bash
nexa \
  --map "actor1.jpg:target_face1.jpg" \
  --map "actor2.jpg:target_face2.jpg" \
  --target group_video.mp4 \
  --output mapped_video.mp4
```

*(How it works: Nexa looks at `target_face1.jpg` and `target_face2.jpg` to get reference embeddings. When it processes `group_video.mp4`, it uses these references to find the right source face by comparing the faces it finds.)*

### Menu of Help

To see all the options:

```bash
nexa --help
```

---

## ☁️ How to Run in Google Colab

You can easily run Nexa in a free Google Colab notebook that has GPU support.

1. Make a new notebook in Google Colab.
2. Choose **T4 GPU** from the **Runtime > Change runtime type** menu.
3. Make a cell and run these setup commands:

```python
# 1. Install prerequisites and FFmpeg
!apt-get update && apt-get install -y ffmpeg
!pip install -q onnxruntime-gpu insightface imageio[ffmpeg] ffmpeg-python typer rich opencv-python-headless tqdm requests

# 2. Copy the repository (if you need to, change the URL to your own)
# !git clone https://github.com/ragibcs/nexa.git
# %cd nexa
# !pip install -e .

# If you only uploaded the source code folder to Colab, you can also do this:
# %cd /content/nexa
# !pip install -e .
```

4. Send in your media:
   - On the left sidebar, click the **Folder** icon.
   - Put your `source.jpg` and `target.mp4` files online.

5. In a new cell, run Nexa with the command:
   ```bash
   !nexa --source source.jpg --target target.mp4 --output result.mp4
   ```

6. Get the result!

---

## 📁 Structure of the Project

```text
nexa/
├── pyproject.toml              # Packaging for modern Python
├── requirements.txt            # List of fallback dependencies
└── src/
    └── nexa/
        ├── __init__.py
        ├── main.py             # Typer CLI entrypoint
        ├── core/
        │   ├── pipeline.py     # In-memory video/image processing loop
        │   ├── mapping.py      # Cosine-similarity face matching engine
        │   └── audio.py        # Audio extraction and muxing via FFmpeg
        ├── models/
        │   ├── manager.py      # Automatic model downloading with progress bar
        │   ├── providers.py    # GPU / CPU provider auto-detection
        │   ├── analyzer.py     # InsightFace detection wrapper
        │   ├── swapper.py      # Inswapper ONNX execution wrapper
        │   └── enhancers.py    # GFPGAN / CodeFormer face restoration
        └── utils/
            ├── video.py        # FFprobe helpers and format detection
            └── logging.py      # Rich console logger and FFmpeg check
```

## ⚠️ Warning

This software is only meant for school, research, and fun. Users are not allowed to use this tool to make deepfakes or change media for bad reasons, such as harassment, spreading false information, or any other activity that violates the rights of others. Please use this tool responsibly and make sure you have permission from the people whose faces you are switching.
