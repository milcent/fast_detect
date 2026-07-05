"""
detector.py — Core YOLO inference loop for fast_detect.

Public API:
    probe_batch_size(model, compress, safety_factor) -> int
    detect_in_video(...)           -> dict          # single-video, sequential
    detect_videos_batched(...)     -> list[dict]    # multi-video, GPU-batched
"""

import math
from pathlib import Path

import cv2
import numpy as np
import torch

from fast_detect.helpers import format_timestamp, seconds_to_dict


# ──────────────────────────────────────────────────────────────────────────────
# GPU batch-size probe
# ──────────────────────────────────────────────────────────────────────────────

def probe_batch_size(
    model,
    compress: int,
    safety_factor: float = 0.80,
) -> int:
    """
    Estimate the maximum number of videos that can be processed simultaneously
    in a lock-step batched inference loop, given the current GPU and model.

    Three limits are computed and the minimum is returned:

    1. **VRAM limit** — free GPU memory × safety_factor ÷ per-frame activation
       overhead.  Crucially, a **warmup pass** is run first so that model
       weights are fully transferred to the GPU before the measurement.  The
       second (probe) pass then captures only the incremental activation cost
       per additional frame, not the one-off weight-loading cost.

    2. **I/O cap** — ``max(8, compress // 2)``.  Higher compress values mean
       the GPU is called less often, so the script can afford more simultaneous
       video decoders without saturating disk I/O.

    3. **Always ≥ 1** — ensures a valid sequential fallback.

    Prints a one-line diagnostic and returns 1 on CPU or probe failure.
    """
    if not torch.cuda.is_available():
        print("  [batch] No CUDA device — running sequentially.")
        return 1

    try:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)

        # ── Warmup pass — forces model weights onto the GPU ───────────────────
        # Ultralytics uses lazy CUDA loading: weights are not resident until
        # the first forward call.  Without this pass, mem_before == 0 and the
        # per_frame measurement includes the one-off weight-transfer cost,
        # which massively under-estimates the true batch capacity.
        torch.cuda.empty_cache()
        _ = model([dummy], verbose=False)
        torch.cuda.synchronize()

        # ── Probe pass — model already resident, measure activations only ─────
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        model_vram = torch.cuda.memory_allocated()   # weights now fixed in VRAM

        _ = model([dummy], verbose=False)

        torch.cuda.synchronize()

        peak      = torch.cuda.max_memory_allocated()
        per_frame = peak - model_vram               # activations per frame only

        if per_frame <= 0:
            print("  [batch] VRAM probe returned zero — running sequentially.")
            return 1

        free_bytes, _ = torch.cuda.mem_get_info()
        available     = int(free_bytes * safety_factor)

        vram_batch = max(1, available // per_frame)

        # I/O cap: higher compress → more GPU idle time → more decoders affordable
        io_cap    = max(8, compress // 2)
        effective = min(vram_batch, io_cap)

        # ── Diagnostic ────────────────────────────────────────────────────────
        gpu_name     = torch.cuda.get_device_name(0)
        free_gb      = free_bytes  / 1024 ** 3
        model_gb     = model_vram  / 1024 ** 3
        per_frame_mb = per_frame   / 1024 ** 2

        print(
            f"  [GPU] {gpu_name} | "
            f"Free: {free_gb:.1f} GB | "
            f"Model: {model_gb:.2f} GB | "
            f"Per-frame: {per_frame_mb:.0f} MB | "
            f"VRAM limit: {vram_batch} | "
            f"I/O cap (compress={compress}): {io_cap} | "
            f"→ Effective batch: {effective}"
        )
        return effective

    except Exception as exc:
        print(f"  [batch] GPU probe failed ({exc}) — running sequentially.")
        return 1


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper: resolve class names → model indices
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_classes(model, target_classes: set[str]) -> dict[str, int]:
    """
    Map requested class-name strings to their integer indices in model.names.
    Prints a warning for any name not found.  Raises ValueError if none resolve.
    """
    class_name_to_id: dict[str, int] = {
        name.lower(): idx for idx, name in model.names.items()
    }
    resolved: dict[str, int] = {}
    for t in target_classes:
        t_lower = t.lower()
        if t_lower in class_name_to_id:
            resolved[t_lower] = class_name_to_id[t_lower]
        else:
            print(f"  [warn] '{t}' not found in model classes — skipping.")

    if not resolved:
        raise ValueError(
            "None of the requested objects are known to this model.\n"
            f"  Available classes: {sorted(class_name_to_id)}\n"
            "  Tip: run  fast-detect --list-classes --model <your_model.pt>  "
            "to see all classes the model supports."
        )
    return resolved


# ──────────────────────────────────────────────────────────────────────────────
# Single-video inference (sequential fallback)
# ──────────────────────────────────────────────────────────────────────────────

def detect_in_video(
    video_path: str,
    model,
    target_classes: set[str],
    compress: int,
    confidence: float,
    verbose: bool,
) -> dict:
    """
    Analyse a single video and return a dict of detections per object class.

    Uses ``cap.grab()`` for skipped frames (no pixel decode) and
    ``cap.read()`` only for the 1-in-N inference frames, reducing CPU I/O
    load proportionally to the compress ratio.

    Returns:
        {
            "video": "path/to/video.mp4",
            "fps": 29.97,
            "total_frames": 1800,
            "frames_analysed": 360,
            "detections": {
                "person": [{"seconds": 1.2, "timestamp": "00:00:01.200"}, ...],
                "car":    [...],
            }
        }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps             = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_analysed = 0

    resolved   = _resolve_classes(model, target_classes)
    detections: dict[str, list[dict]] = {name: [] for name in resolved}
    frame_idx  = 0

    while True:
        is_inference_frame = (frame_idx % compress == 0)

        if is_inference_frame:
            ret, frame = cap.read()     # decode pixel data
        else:
            ret   = cap.grab()          # advance stream position, no decode
            frame = None

        if not ret:
            break

        if is_inference_frame:
            frames_analysed += 1
            current_time = frame_idx / fps

            results = model(frame, verbose=False, conf=confidence)[0]

            detected_now: set[str] = set()
            for box in results.boxes:
                cls_id   = int(box.cls[0])
                cls_name = model.names[cls_id].lower()
                if cls_name in resolved:
                    detected_now.add(cls_name)

            for name in detected_now:
                detections[name].append(seconds_to_dict(current_time))

            if verbose:
                found = ", ".join(detected_now) if detected_now else "—"
                print(
                    f"  frame {frame_idx:6d} / {total_frames}"
                    f"  t={format_timestamp(current_time)}  detected: {found}"
                )

        frame_idx += 1

    cap.release()

    return {
        "video":           video_path,
        "fps":             round(fps, 3),
        "total_frames":    total_frames,
        "frames_analysed": frames_analysed,
        "detections":      detections,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Multi-video batched inference
# ──────────────────────────────────────────────────────────────────────────────

def detect_videos_batched(
    video_paths: list[str],
    model,
    target_classes: set[str],
    compress: int,
    confidence: float,
    verbose: bool,
) -> list[dict]:
    """
    Process multiple videos simultaneously in a lock-step batched inference loop.

    At each compress interval, one frame is collected from every active video,
    all frames are stacked into a single batch, and inference runs once.

    **I/O optimisation:** on non-inference ticks, each capture uses
    ``cap.grab()`` (stream-position advance only, no pixel decode) instead of
    ``cap.read()``.  At compress=60 this eliminates 59 of every 60 full frame
    decompresses per video, dramatically reducing CPU I/O time and letting the
    GPU stay closer to full utilisation.

    Returns a list of result dicts in the same schema as ``detect_in_video()``,
    one entry per input path (order preserved).
    """
    resolved = _resolve_classes(model, target_classes)

    # ── Open all captures ─────────────────────────────────────────────────────
    caps:    list[cv2.VideoCapture | None] = []
    results: list[dict]                    = []

    for path in video_paths:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"  [error] Cannot open video: {path} — skipping.")
            caps.append(None)
            results.append({"video": path, "error": "Cannot open video"})
            continue

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        caps.append(cap)
        results.append({
            "video":           path,
            "fps":             round(fps, 3),
            "total_frames":    total_frames,
            "frames_analysed": 0,
            "detections":      {name: [] for name in resolved},
        })

    # ── Lock-step reading loop ────────────────────────────────────────────────
    active:    list[int] = [i for i, cap in enumerate(caps) if cap is not None]
    frame_idx: int       = 0

    while active:
        batch_frames:      list      = []
        batch_indices:     list[int] = []   # batch slot → results[i]
        still_active:      list[int] = []

        # Decide once per tick whether this is an inference frame
        is_inference_tick = (frame_idx % compress == 0)

        for i in active:
            cap = caps[i]   # type: ignore[assignment]

            if is_inference_tick:
                ret, frame = cap.read()     # decode — frame needed for GPU
            else:
                ret   = cap.grab()          # advance only — no pixel decode
                frame = None

            if not ret:
                cap.release()
                caps[i] = None
                continue

            still_active.append(i)

            if is_inference_tick:
                batch_frames.append(frame)
                batch_indices.append(i)

        active = still_active

        # ── Batched GPU inference ─────────────────────────────────────────────
        if batch_frames:
            batch_preds = model(batch_frames, verbose=False, conf=confidence)

            for pred, vid_idx in zip(batch_preds, batch_indices):
                fps_i        = results[vid_idx]["fps"]
                current_time = frame_idx / fps_i
                results[vid_idx]["frames_analysed"] += 1

                detected_now: set[str] = set()
                for box in pred.boxes:
                    cls_id   = int(box.cls[0])
                    cls_name = model.names[cls_id].lower()
                    if cls_name in resolved:
                        detected_now.add(cls_name)

                for name in detected_now:
                    results[vid_idx]["detections"][name].append(
                        seconds_to_dict(current_time)
                    )

                if verbose:
                    found    = ", ".join(detected_now) if detected_now else "—"
                    vid_name = Path(results[vid_idx]["video"]).name
                    total_fr = results[vid_idx]["total_frames"]
                    print(
                        f"  [{vid_name}]  frame {frame_idx:6d} / {total_fr}"
                        f"  t={format_timestamp(current_time)}  detected: {found}"
                    )

        frame_idx += 1

    return results
