"""
cli.py — CLI entry point for fast_detect.

Detects objects in video files (including DVR .dav footage) and writes
a timestamped JSON report to the results/ folder. Optionally extracts a
merged highlight video containing only the segments with detected objects.

Usage:
    # Individual files
    uv run fast-detect --videos vid1.dav vid2.mp4 \\
                       --objects person car \\
                       --compress 60

    # Folder of videos — with highlight clip output
    uv run fast-detect --folder /path/to/videos \\
                       --objects person car \\
                       --compress 60 --recursive --clip

    # Or, after `uv sync`, use the installed command:
    fast-detect --videos clip.mp4 --objects person --compress 30 --clip

    # List all classes a model knows about (useful for custom models)
    fast-detect --list-classes --model my_custom_model.pt
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

# ── Ultralytics: redirect model downloads to ./models/ before any import ──────
try:
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    _models_dir = Path("models").resolve()
    _models_dir.mkdir(exist_ok=True)
    SETTINGS.update({"weights_dir": str(_models_dir)})

except ImportError:
    sys.exit(
        "Missing dependency: run  uv sync  then try again."
    )

from fast_detect.detector import detect_in_video, detect_videos_batched, probe_batch_size
from fast_detect.helpers import collect_video_files
from fast_detect.clipper import extract_highlights


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def resolve_model_path(model_arg: str) -> str:
    """
    Resolve a model filename to an absolute path.

    Search order:
    1. The argument itself (absolute path or CWD-relative path that exists)
    2. ./models/<model_arg>
    3. Fall back to the bare name and let Ultralytics download it to ./models/
    """
    p = Path(model_arg)
    if p.exists():
        return str(p)
    local = Path("models") / model_arg
    if local.exists():
        return str(local)
    return model_arg  # YOLO will download it into weights_dir (./models/)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Compute the run timestamp before parsing so it can be the default output name
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        prog="fast-detect",
        description="Detect objects in video files and export timestamped JSON reports.",
    )
    parser.add_argument(
        "--videos", nargs="+", default=[],
        help="One or more video file paths.",
    )
    parser.add_argument(
        "--folder", default=None,
        help="Folder containing video files to analyse.",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Search --folder recursively for video files.",
    )
    parser.add_argument(
        "--objects", nargs="+", default=None,
        help=(
            "Object class names to detect. "
            "For COCO-pretrained models: person, car, dog, etc. "
            "For fine-tuned models: use the class names defined during training. "
            "Use --list-classes to discover available names."
        ),
    )
    parser.add_argument(
        "--compress", type=int, default=1,
        help="Process 1 frame every N frames (default: 1 = every frame).",
    )
    parser.add_argument(
        "--output", default=None,
        help=(
            "Output JSON file path. "
            f"Defaults to results/{run_ts}.json"
        ),
    )
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help=(
            "YOLO model weights file (default: yolov8n.pt). "
            "Supports standard Ultralytics models (auto-downloaded if not found) "
            "and local fine-tuned .pt files with custom classes."
        ),
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Minimum detection confidence 0–1 (default: 0.5).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-frame progress.",
    )
    parser.add_argument(
        "--list-classes", action="store_true", dest="list_classes",
        help=(
            "Load the model, print all class names it recognises, and exit. "
            "Useful for discovering classes in fine-tuned custom models."
        ),
    )
    parser.add_argument(
        "--clip", action="store_true",
        help=(
            "After detection, extract a merged highlight video containing only "
            "the segments where target objects were detected. "
            f"Output: results/highlights_<run_ts>.mp4"
        ),
    )
    parser.add_argument(
        "--clip-buffer", type=float, default=2.0, dest="clip_buffer",
        metavar="SECONDS",
        help="Seconds of padding added before/after each detected frame (default: 2.0).",
    )
    parser.add_argument(
        "--clip-boxes", action="store_true", dest="clip_boxes",
        help=(
            "Draw YOLO bounding boxes on highlight frames (requires --clip). "
            "WARNING: this re-runs the YOLO model on every frame of the "
            "highlight video, which adds significant time and GPU/CPU compute "
            "on top of the initial detection pass."
        ),
    )
    parser.add_argument(
        "--clip-transition", type=float, default=1.0, dest="clip_transition",
        metavar="SECONDS",
        help=(
            "Seconds of black screen transition inserted between different "
            "source videos in the highlight clip (default: 1.0). "
            "Set to 0 to disable transitions."
        ),
    )
    args = parser.parse_args()

    # ── --list-classes: load model, print classes, exit ────────────────────────
    if args.list_classes:
        model_path = resolve_model_path(args.model)
        print(f"Loading model: {model_path}")
        model = YOLO(model_path)
        print(f"\nModel: {args.model}")
        print(f"Number of classes: {len(model.names)}")
        print(f"\nClass names:")
        for idx, name in sorted(model.names.items()):
            print(f"  {idx:3d}: {name}")
        return

    # ── Validate arguments ────────────────────────────────────────────────────
    if args.objects is None:
        parser.error("--objects is required (unless using --list-classes)")
    if args.compress < 1:
        parser.error("--compress must be >= 1")
    if not (0 < args.confidence <= 1):
        parser.error("--confidence must be between 0 and 1")
    if not args.videos and args.folder is None:
        parser.error("Provide at least one of --videos or --folder.")
    if args.clip_buffer <= 0:
        parser.error("--clip-buffer must be > 0")
    if args.clip_boxes and not args.clip:
        parser.error("--clip-boxes requires --clip")
    if args.clip_transition < 0:
        parser.error("--clip-transition must be >= 0")

    # ── Resolve output path ───────────────────────────────────────────────────
    out_path = Path(args.output) if args.output else Path("results") / f"{run_ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Collect all video paths ───────────────────────────────────────────────
    video_paths:  list[Path] = []
    skipped_files: list[dict] = []   # unreadable entries → go straight into JSON

    for p in args.videos:
        path = Path(p)
        try:
            if not path.exists():
                print(f"[skip] File not found: {p}")
            else:
                video_paths.append(path)
        except OSError as exc:
            print(f"[skip] Unreadable path '{p}': {exc}")
            skipped_files.append({"video": p, "error": str(exc)})

    if args.folder is not None:
        try:
            folder_files, folder_skipped = collect_video_files(
                args.folder, args.recursive
            )
            skipped_files.extend(folder_skipped)
            print(
                f"Found {len(folder_files)} video(s) in '{args.folder}'"
                + (" (recursive)" if args.recursive else "")
            )
            if folder_skipped:
                print(
                    f"  ({len(folder_skipped)} unreadable file(s) skipped"
                    f" — will be logged in results JSON)"
                )
            # Deduplicate: skip files already listed via --videos
            existing = {p.resolve() for p in video_paths}
            for f in folder_files:
                if f.resolve() not in existing:
                    video_paths.append(f)
        except NotADirectoryError as exc:
            parser.error(str(exc))

    if not video_paths and not skipped_files:
        sys.exit("No valid video files found. Exiting.")

    print(f"Total videos to process: {len(video_paths)}")

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = resolve_model_path(args.model)
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    target_classes = set(args.objects)

    # ── Probe GPU batch capacity ──────────────────────────────────────────────────
    print("\nProbing GPU batch capacity...")
    batch_size      = probe_batch_size(model, compress=args.compress)
    effective_batch = min(batch_size, len(video_paths))

    # ── Run detection ─────────────────────────────────────────────────────────
    # Pre-populate with any files that were unreadable at enumeration time so
    # they are recorded in the JSON output even though they were never processed.
    all_results: list[dict] = list(skipped_files)
    total_videos = len(video_paths)


    if effective_batch > 1:
        # ── Batched inference ──────────────────────────────────────────────
        print(
            f"\nBatched inference: {effective_batch} video(s) at a time "
            f"({total_videos} total — "
            f"{math.ceil(total_videos / effective_batch)} batch(es))"
        )
        for chunk_start in range(0, total_videos, effective_batch):
            chunk_paths = [
                str(p) for p in
                video_paths[chunk_start : chunk_start + effective_batch]
            ]
            lo = chunk_start + 1
            hi = min(chunk_start + effective_batch, total_videos)
            names = ", ".join(Path(p).name for p in chunk_paths)
            print(f"\n  Batch [{lo}–{hi} / {total_videos}]: {names}")

            try:
                chunk_results = detect_videos_batched(
                    video_paths=chunk_paths,
                    model=model,
                    target_classes=target_classes,
                    compress=args.compress,
                    confidence=args.confidence,
                    verbose=args.verbose,
                )
                all_results.extend(chunk_results)

                for r in chunk_results:
                    if "error" not in r:
                        for obj, hits in r["detections"].items():
                            print(
                                f"  {Path(r['video']).name} | "
                                f"{obj}: {len(hits)} occurrence(s)"
                            )
            except ValueError as exc:
                print(f"  [error] {exc}")
                for p in chunk_paths:
                    all_results.append({"video": p, "error": str(exc)})

    else:
        # ── Sequential fallback ──────────────────────────────────────────────
        for video_path in video_paths:
            print(f"\nAnalysing: {video_path}  (compress={args.compress}x)")
            try:
                result = detect_in_video(
                    video_path=str(video_path),
                    model=model,
                    target_classes=target_classes,
                    compress=args.compress,
                    confidence=args.confidence,
                    verbose=args.verbose,
                )
                all_results.append(result)

                for obj, hits in result["detections"].items():
                    print(f"  {obj}: {len(hits)} occurrence(s)")

            except (IOError, ValueError) as exc:
                print(f"  [error] {exc}")
                all_results.append({"video": str(video_path), "error": str(exc)})

    # ── Write JSON output ─────────────────────────────────────────────────────
    output = {
        "run_timestamp": run_ts,
        "model": args.model,
        "confidence_threshold": args.confidence,
        "compress_rate": args.compress,
        "target_objects": list(target_classes),
        "videos": all_results,
    }

    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults written to: {out_path.resolve()}")

    # ── Extract highlight clip (optional) ─────────────────────────────────────
    if args.clip:
        clip_path = out_path.parent / f"highlights_{run_ts}.mp4"
        print(f"\nExtracting highlight clip → {clip_path}")
        try:
            summary = extract_highlights(
                video_results=all_results,
                output_path=str(clip_path),
                buffer_secs=args.clip_buffer,
                transition_secs=args.clip_transition,
                draw_boxes=args.clip_boxes,
                model=model,
                confidence=args.confidence,
                target_classes=target_classes,
            )
            if summary["output"] is None:
                print("  No detections found in any video — highlight clip not created.")
            else:
                print(
                    f"  Done: {summary['total_clips']} segment(s) from "
                    f"{summary['sources_with_hits']} source(s), "
                    f"{summary['total_duration_secs']}s total."
                )
                print(f"  Clip saved to: {clip_path.resolve()}")
        except IOError as exc:
            print(f"  [error] Could not write highlight clip: {exc}")


if __name__ == "__main__":
    main()
