# fast_detect

YOLO-based object detection for batch video files. Scans one or more videos (including DVR `.dav` files), detects objects of interest, and exports a JSON report with the timestamps of every detection.

## Features

- **Batch processing** — analyse individual files or entire folders (with optional recursion)
- **`--compress` mode** — process only 1 out of every N frames for dramatically faster throughput on long recordings
- **Highlight clips** (`--clip`) — automatically extract a merged highlight video containing only the segments with detected objects
- **Source transitions** — black screen title cards between different source videos in highlight clips for clear visual separation
- **Bounding-box overlays** (`--clip-boxes`) — optionally draw YOLO bounding boxes with class labels and confidence on highlight frames
- **Custom models** — use any YOLOv8 weights (standard or fine-tuned with custom classes); standard weights are auto-downloaded on first run
- **Flexible class filtering** — specify any class names your model recognises (COCO classes for pretrained models, custom names for fine-tuned models)
- **JSON output** — structured report with per-video stats and per-class timestamped detections
- **DVR support** — natively handles `.dav` surveillance footage alongside standard formats
- **Cross-platform** — runs on Windows, macOS, and Linux

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for environment management.

```bash
# Clone the repository
git clone https://github.com/your-org/fast_detect.git
cd fast_detect

# Install dependencies (creates .venv automatically)
uv sync
```

### GPU Acceleration (CUDA)

By default, `uv sync` installs PyTorch from PyPI, which ships with the latest supported CUDA version. This works out of the box for most modern NVIDIA GPUs.

**If you have an older GPU** that requires a specific CUDA version (e.g. CUDA 11.8), override PyTorch after syncing:

```bash
uv sync
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Replace `cu118` with your CUDA version (`cu121`, `cu124`, etc.). Check your CUDA version with `nvidia-smi`.

**macOS** users get CPU-only PyTorch (with MPS support on Apple Silicon) automatically — no extra steps needed.

## Quick Start

```bash
# Detect persons and cars in a single file, analysing 1 frame per second (30 fps ÷ 30)
uv run fast-detect \
    --videos footage/cam01.dav \
    --objects person car \
    --compress 30

# Process an entire folder recursively using the large model
uv run fast-detect \
    --folder D:/recordings/2026-04-20 \
    --recursive \
    --objects person car truck \
    --compress 60 \
    --model yolov8l.pt \
    --output results.json

# Detection + highlight clip with bounding boxes
uv run fast-detect \
    --videos cam01.dav cam02.dav \
    --objects person car \
    --compress 30 \
    --clip --clip-boxes
```

## Custom / Fine-Tuned Models

fast_detect works with **any YOLO model**, not just the COCO-pretrained ones. If you've fine-tuned a YOLOv8 model on a custom dataset (e.g. `helmet`, `forklift`, `hard_hat`), simply point `--model` at your `.pt` file:

```bash
# Use a custom fine-tuned model
uv run fast-detect \
    --videos factory_cam.mp4 \
    --model /path/to/my_custom_model.pt \
    --objects helmet forklift \
    --compress 30

# Discover which classes a model recognises
uv run fast-detect --list-classes --model /path/to/my_custom_model.pt
```

The `--list-classes` flag loads the model, prints every class name it knows, and exits. This is especially useful when working with models you didn't train yourself.

### Model Path Resolution

The `--model` argument is resolved in this order:
1. The path as-is (absolute or CWD-relative)
2. `./models/<model_arg>`
3. Bare name → Ultralytics auto-downloads to `./models/`

## CLI Reference

```
usage: fast-detect [-h]
                   [--videos VIDEOS [VIDEOS ...]]
                   [--folder FOLDER]
                   [--recursive]
                   --objects OBJECTS [OBJECTS ...]
                   [--compress N]
                   [--output OUTPUT]
                   [--model MODEL]
                   [--confidence CONFIDENCE]
                   [--verbose]
                   [--list-classes]
                   [--clip]
                   [--clip-buffer SECONDS]
                   [--clip-boxes]
                   [--clip-transition SECONDS]
```

| Argument | Default | Description |
|---|---|---|
| `--videos` | — | One or more video file paths |
| `--folder` | — | Folder of videos to analyse |
| `--recursive` | `False` | Search `--folder` recursively |
| `--objects` | *(required)* | Object class names to detect (e.g. `person car`). Use `--list-classes` to discover available names |
| `--compress N` | `1` | Analyse 1 frame every N frames |
| `--model` | `yolov8n.pt` | YOLO weights file — standard or fine-tuned `.pt` (auto-downloaded if standard and not found) |
| `--confidence` | `0.5` | Minimum detection confidence (0–1) |
| `--output` | `results/<timestamp>.json` | Output JSON file path |
| `--verbose` | `False` | Print per-frame progress |
| `--list-classes` | `False` | Load model, print all known class names, and exit |
| `--clip` | `False` | Extract a merged highlight video of detection segments |
| `--clip-buffer` | `2.0` | Seconds of padding before/after each detected frame |
| `--clip-boxes` | `False` | Draw YOLO bounding boxes on highlight frames (requires `--clip`) |
| `--clip-transition` | `1.0` | Seconds of black screen transition between source videos in highlights (0 to disable) |

At least one of `--videos` or `--folder` is required (unless using `--list-classes`).

## The `--compress` Parameter

`--compress N` is the key performance lever for batch processing.

| `--compress` | Frames analysed (30 fps video) | Speed gain (approx.) |
|---|---|---|
| `1` | Every frame | 1× (baseline) |
| `5` | 6 per second | ~5× |
| `15` | 2 per second | ~15× |
| `30` | 1 per second | ~30× |
| `60` | 1 every 2 seconds | ~60× |

**Trade-off:** higher values are faster but may miss very short events. For surveillance use-cases where objects are typically present for several seconds, `--compress 30` or `--compress 60` is usually a good starting point.

## Highlight Clips (`--clip`)

Pass `--clip` to produce a merged highlight video after detection. Only the segments where target objects were detected are included, with the source filename burned into every frame.

```bash
uv run fast-detect --videos cam01.dav cam02.dav --objects person --compress 30 --clip
```

- Output: `results/highlights_<timestamp>.mp4`
- One output file per run, spanning all source videos that had detections
- `--clip-buffer N` (default: `2.0`) adds N seconds of padding before/after each detection; overlapping segments are merged

### Source Transitions

When highlight clips contain segments from **multiple source videos**, a black screen transition with a **title card** (showing the next video's filename) is automatically inserted between sources. This makes it easy to tell when the footage switches cameras.

- `--clip-transition N` (default: `1.0`) controls the duration in seconds
- Set `--clip-transition 0` to disable transitions and get seamless joining (old behaviour)

### Bounding-Box Overlays (`--clip-boxes`)

Add `--clip-boxes` to draw YOLO bounding boxes with class labels and confidence scores on every frame of the highlight video.

```bash
uv run fast-detect --videos cam01.dav --objects person car --compress 30 --clip --clip-boxes
```

> ⚠️ **Performance warning:** `--clip-boxes` **re-runs the YOLO model on every frame** of the highlight video. This is a separate inference pass on top of the initial detection scan and adds significant time and GPU/CPU compute. The cost scales with the total number of highlight frames (which depends on detection density and `--clip-buffer`). For large batches or long highlights, expect a noticeable increase in processing time.

## Output Format

```json
{
  "model": "yolov8l",
  "confidence_threshold": 0.5,
  "compress_rate": 60,
  "target_objects": ["person", "car"],
  "videos": [
    {
      "video": "D:\\recordings\\cam01.dav",
      "fps": 30.0,
      "total_frames": 26970,
      "frames_analysed": 450,
      "detections": {
        "person": [
          {"seconds": 12.0, "timestamp": "00:00:12.000"},
          {"seconds": 74.0, "timestamp": "00:01:14.000"}
        ],
        "car": []
      }
    }
  ]
}
```

Each entry in a detections list corresponds to one analysed frame where that class was detected.

## Supported Video Formats

`.mp4` · `.mkv` · `.avi` · `.mov` · `.wmv` · `.flv` · `.webm` · `.m4v` · `.mpeg` · `.mpg` · `.dav`

## Available YOLO Models

| Model | File | Size | Speed | Accuracy |
|---|---|---|---|---|
| YOLOv8 Nano | `yolov8n.pt` | ~6 MB | Fastest | Lowest |
| YOLOv8 Small | `yolov8s.pt` | ~22 MB | Fast | Low |
| YOLOv8 Medium | `yolov8m.pt` | ~52 MB | Medium | Medium |
| YOLOv8 Large | `yolov8l.pt` | ~88 MB | Slow | High |
| YOLOv8 XLarge | `yolov8x.pt` | ~131 MB | Slowest | Highest |

Standard models are auto-downloaded from Ultralytics on first use. Fine-tuned `.pt` files are loaded directly — use `--list-classes` to inspect their classes.

## Project Structure

```
fast_detect/
├── src/
│   └── fast_detect/
│       ├── __init__.py       # package marker + version
│       ├── __main__.py       # python -m fast_detect support
│       ├── cli.py            # CLI entry point (argparse + main)
│       ├── detector.py       # core YOLO inference loop
│       ├── clipper.py        # highlight video extraction
│       ├── helpers.py        # utility functions
│       └── constants.py      # VIDEO_EXTENSIONS + COCO_CLASSES
├── models/                   # (gitignored) YOLO weights
├── results/                  # (gitignored) JSON output + highlight clips
├── videos/                   # (gitignored) local test videos
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
```

## Requirements

- Python ≥ 3.13
- NVIDIA GPU with CUDA drivers (optional — CPU inference is supported but slower)
- `uv` package manager

## License

See [LICENSE](LICENSE).