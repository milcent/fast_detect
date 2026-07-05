# fast_detect — Project Context for AI Assistants

## Project Overview

**fast_detect** is a YOLO-based object detection tool that scans video files (including batch processing of entire folders) and produces timestamped JSON reports of when specific objects appear. The primary use-case is surveillance / CCTV footage review (`.dav` files from DVR systems, as well as standard video formats).

**Key innovation:** the `--compress N` parameter allows skipping N-1 frames between each inference call, making batch processing of large video archives feasible without GPU-level acceleration.

**Custom model support:** fast_detect works with any YOLO model, not just COCO-pretrained ones. Fine-tuned `.pt` files with custom class names are fully supported. The `--list-classes` flag lets users discover what classes a model knows.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python ≥ 3.13 |
| Package manager | `uv` (mandatory — do NOT use pip directly) |
| Object detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Video decoding | OpenCV (`opencv-python`) |
| Data manipulation | Pandas |
| GPU acceleration | PyTorch (default PyPI — includes latest CUDA on Windows/Linux, CPU on macOS) |

### PyTorch / CUDA Configuration

PyTorch is installed from the standard PyPI index, which ships with the latest supported CUDA version on Windows/Linux and CPU-only (with MPS) on macOS.

Users with older GPUs requiring a specific CUDA version can override after sync:

```bash
uv sync
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Project Structure

```
fast_detect/
├── src/
│   └── fast_detect/
│       ├── __init__.py       # package marker + __version__
│       ├── __main__.py       # python -m fast_detect support
│       ├── cli.py            # CLI entry point — argparse + main()
│       ├── detector.py       # detect_in_video() — core YOLO inference loop
│       ├── clipper.py        # extract_highlights() — highlight video extraction
│       ├── helpers.py        # collect_video_files, format_timestamp, seconds_to_dict
│       └── constants.py      # VIDEO_EXTENSIONS + COCO_CLASSES dict (80 classes, index 0–79)
├── models/                   # (untracked) YOLO weights — all downloads go here
├── results/                  # (untracked) timestamped JSON output files + highlight clips
├── videos/                   # (untracked) local test videos
├── pyproject.toml            # uv project definition + [project.scripts] entry point
└── uv.lock                   # Deterministic dependency lock file
```

### Module responsibilities

| File | Responsibility |
|---|---|
| `cli.py` | CLI parsing (`argparse`), GPU probe, model loading, `--list-classes`, batched/sequential routing, JSON output |
| `detector.py` | `probe_batch_size()` — VRAM probe; `detect_in_video()` — sequential loop; `detect_videos_batched()` — lock-step batched loop |
| `clipper.py` | `extract_highlights()` — builds time segments, merges, writes highlight video with source transitions |
| `helpers.py` | `collect_video_files()`, `format_timestamp()`, `seconds_to_dict()` |
| `constants.py` | `VIDEO_EXTENSIONS` set; `COCO_CLASSES` dict (index → name for all 80 COCO classes) |

---

## CLI Entry Point

The project is installed as a CLI command via `[project.scripts]` in `pyproject.toml`:

```toml
[project.scripts]
fast-detect = "fast_detect.cli:main"
```

After `uv sync`, the command `fast-detect` is available in the virtual environment:

```bash
uv run fast-detect --videos vid1.dav --objects person car --compress 60
# or, if the venv is activated:
fast-detect --videos vid1.dav --objects person car --compress 60
```

---

## Primary Script: `cli.py`

### CLI Interface

```bash
fast-detect \
    --videos vid1.dav vid2.mp4 \
    --folder /path/to/videos \
    --recursive \
    --objects person car truck \
    --compress 60 \
    --model yolov8l.pt \
    --confidence 0.5 \
    --output results/custom.json \
    --clip \
    --clip-buffer 2.0 \
    --clip-boxes \
    --clip-transition 1.0 \
    --verbose
```

### Custom / Fine-Tuned Models

fast_detect accepts **any YOLO `.pt` file**, including fine-tuned models with custom class names:

```bash
# Discover what classes a model knows
fast-detect --list-classes --model my_custom_model.pt

# Use a fine-tuned model
fast-detect --videos factory.mp4 --model my_custom_model.pt --objects helmet forklift
```

The `--list-classes` flag loads the model, prints all class names and indices, and exits. `--objects` is not required when using `--list-classes`.

Class names are resolved dynamically from `model.names` at runtime, so custom models work transparently.

### Highlight Clip Extraction (`--clip`)

Pass `--clip` to produce a merged highlight video after detection:

```bash
fast-detect --videos cam01.dav cam02.dav --objects person --compress 30 --clip
```

- Output: `results/highlights_<run_ts>.mp4` (same timestamp as the JSON)
- One output file per run, spanning **all source videos** that had detections
- Each frame has the **source filename** burned into the bottom-left corner with a semi-transparent dark background
- `--clip-buffer N` (default: `2.0`) adds N seconds of padding before/after each detected frame; overlapping segments are merged automatically
- By default, raw frames only — no YOLO bounding boxes (faster extraction)
- All source videos are resized to match the resolution of the first video with hits

#### Source Transitions (`--clip-transition`)

When segments from multiple source videos are joined, a black screen transition with a centered title card (displaying the next video's filename) is automatically inserted.

- `--clip-transition N` (default: `1.0`) controls the duration in seconds
- Set `--clip-transition 0` to disable transitions

#### Bounding Boxes on Highlights (`--clip-boxes`)

Pass `--clip-boxes` (requires `--clip`) to overlay YOLO bounding boxes on the highlight video. Each target object is drawn with a vivid green rectangle and a label showing the class name and confidence score.

```bash
fast-detect --videos cam01.dav --objects person car --compress 30 --clip --clip-boxes
```

> **⚠️ Performance warning:** `--clip-boxes` **re-runs the YOLO model on every frame** of the highlight video. This is a separate inference pass from the initial detection scan, and adds significant time and GPU/CPU compute. The cost scales with the total number of highlight frames (which depends on how many detections there are and the `--clip-buffer` value). For large batches or long highlight durations, expect a noticeable increase in processing time.

### `--compress` Parameter (Core Feature)

`--compress N` causes the detector to **analyse only 1 out of every N frames**.

- **Default:** `1` (every frame — full accuracy, slowest)
- **Example:** `--compress 60` on a 30 fps video → 1 frame analysed per 2 seconds of footage
- **Implementation:** `frame_idx % compress == 0` inside the main read loop in `detector.py`
- **Speed vs. accuracy tradeoff:** higher values = faster, but may miss short-duration events
- **I/O optimisation:** `detector.py` uses `cap.grab()` for skipped frames (no pixel decode) and `cap.read()` only for inference frames, reducing CPU I/O proportionally to the compress ratio.

### Output Format

Results are written to `results/YYYYMMDD_HHMMSS.json` by default (one file per run, named with the run's local timestamp).

```json
{
  "run_timestamp": "20260620_103200",
  "model": "yolov8l",
  "confidence_threshold": 0.5,
  "compress_rate": 60,
  "target_objects": ["person", "car"],
  "videos": [
    {
      "video": "D:\\footage\\cam01.dav",
      "fps": 30.0,
      "total_frames": 26970,
      "frames_analysed": 450,
      "detections": {
        "person": [
          {"seconds": 12.0, "timestamp": "00:00:12.000"},
          {"seconds": 14.0, "timestamp": "00:00:14.000"}
        ],
        "car": []
      }
    }
  ]
}
```

**One entry per analysed frame where the object was detected.** If an object appears in N consecutive analysed frames, it will have N entries.

### Model Path Resolution

`cli.py` resolves the `--model` argument in this order:
1. The argument as-is (absolute or CWD-relative path that exists)
2. `./models/<model_arg>` — the local models directory
3. Bare name handed to YOLO for auto-download into `./models/`

Ultralytics `weights_dir` is also updated to `./models/` at startup via `SETTINGS.update()`.

### Supported Video Formats

```python
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv",
    ".flv", ".webm", ".m4v", ".mpeg", ".mpg",
    ".dav"   # ← DVR/surveillance format
}
```

---

## Constants (`constants.py`)

- `VIDEO_EXTENSIONS` — set of supported video file suffixes
- `COCO_CLASSES` — `dict[int, str]` mapping index 0–79 to COCO class names (the 80 standard classes YOLOv8 was trained on). Use this for reference or validation; `detector.py` resolves class names dynamically from `model.names` at runtime.

---

## Running the Project

```bash
# Always use uv to manage the environment
uv sync                          # install / sync deps + register fast-detect command
uv run fast-detect --help        # verify CLI

# Run detection
uv run fast-detect --videos myvideo.mp4 --objects person car --compress 30

# List classes of a custom model
uv run fast-detect --list-classes --model my_model.pt

# Add a new dependency
uv add some-package
```

---

## Development Conventions

- **Branch strategy:** Git Flow (feature branches → `develop` → `main`; releases via `release/` branches)
- **Environment:** Always use `uv` — never call `pip` directly
- **New modules:** add them inside `src/fast_detect/` — the package is auto-discovered via `[tool.setuptools.packages.find]`
- **No hardcoded paths:** all paths come through CLI arguments
- **Output:** always JSON to `results/`; never stdout-only

---

## Known Limitations / Future Work

1. **No deduplication of detections** — consecutive frames with the same object produce multiple entries; consider event-based grouping (e.g., merge detections within a time window)
2. **Single-threaded I/O** — within each batch, `cap.grab()` / `cap.read()` calls are sequential across all open video captures; parallelising with `ThreadPoolExecutor` could reduce the remaining I/O overhead
3. **No progress bar** — for large batches, a `tqdm` progress bar would improve UX
4. **No CSV/Excel export** — only JSON output is supported
