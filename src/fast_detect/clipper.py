"""
clipper.py — Highlight video extraction for fast_detect.

After detection, this module reads the detection results and writes a single
merged .mp4 file containing only the segments of each source video where one
or more target objects were found, with the source filename burned into every frame.

When segments from multiple source videos are joined, a short black screen
transition with a title card is inserted between them.
"""

from pathlib import Path

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_segments(
    detections: dict[str, list[dict]],
    fps: float,
    total_frames: int,
    buffer_secs: float,
) -> list[tuple[int, int]]:
    """
    Convert sparse detection timestamps into merged frame-range segments.

    Each detected timestamp is expanded by `buffer_secs` on both sides, then
    overlapping/adjacent intervals are merged.

    Returns a sorted list of (start_frame, end_frame) integer pairs.
    """
    total_duration = total_frames / fps

    # Collect all detection timestamps across every target class
    all_seconds: list[float] = []
    for hits in detections.values():
        for h in hits:
            all_seconds.append(h["seconds"])

    if not all_seconds:
        return []

    # Build raw (clamped) segments
    raw: list[tuple[float, float]] = []
    for t in all_seconds:
        start = max(0.0, t - buffer_secs)
        end = min(total_duration, t + buffer_secs)
        raw.append((start, end))

    # Sort and merge overlapping intervals
    raw.sort()
    merged: list[list[float]] = []
    for start, end in raw:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    # Convert to integer frame indices
    return [
        (int(s * fps), min(int(e * fps), total_frames))
        for s, e in merged
    ]


def _stamp_label(frame, label: str) -> None:
    """Burn a source-filename label into the bottom-left corner of *frame* in-place."""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    scale      = 0.60
    thickness  = 2
    margin     = 10
    pad        = 4  # padding around text inside the background rect

    h, w = frame.shape[:2]
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)

    rect_x1 = margin - pad
    rect_y1 = h - margin - text_h - baseline - pad
    rect_x2 = margin + text_w + pad
    rect_y2 = h - margin + pad

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # White text
    cv2.putText(
        frame, label,
        (margin, h - margin - baseline),
        font, scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )

def _draw_boxes(frame, model, confidence: float, target_classes: set[str]) -> None:
    """Run YOLO on *frame* and draw bounding boxes for target classes in-place."""
    results = model(frame, verbose=False, conf=confidence)[0]

    for box in results.boxes:
        cls_id   = int(box.cls[0])
        cls_name = model.names[cls_id].lower()
        if cls_name not in target_classes:
            continue

        conf  = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        label = f"{cls_name} {conf:.0%}"

        # Box colour: vivid green
        colour = (0, 255, 100)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)

        # Label background + text
        font      = cv2.FONT_HERSHEY_SIMPLEX
        scale     = 0.50
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - baseline - 2),
            font, scale, (0, 0, 0), thickness, cv2.LINE_AA,
        )


def _write_transition(
    writer: cv2.VideoWriter,
    width: int,
    height: int,
    fps: float,
    duration_secs: float,
    next_label: str,
) -> int:
    """
    Write a black-screen transition with a centered title card.

    The title card displays the filename of the *next* source video so the
    viewer knows which camera/file is coming next.

    Returns the number of frames written.
    """
    num_frames = max(1, int(fps * duration_secs))
    black = np.zeros((height, width, 3), dtype=np.uint8)

    # Render the title card text — auto-scale to fit the frame width
    font      = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    max_text_w = int(width * 0.80)  # leave 10% margin on each side

    # Start large and shrink until it fits
    scale = 1.20
    while scale > 0.30:
        (text_w, text_h), baseline = cv2.getTextSize(next_label, font, scale, thickness)
        if text_w <= max_text_w:
            break
        scale -= 0.05
    else:
        (text_w, text_h), baseline = cv2.getTextSize(next_label, font, scale, thickness)

    # Center the text
    x = (width  - text_w) // 2
    y = (height + text_h) // 2

    title_frame = black.copy()
    cv2.putText(
        title_frame, next_label,
        (x, y),
        font, scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )

    for _ in range(num_frames):
        writer.write(title_frame)

    return num_frames


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def extract_highlights(
    video_results: list[dict],
    output_path: str,
    buffer_secs: float = 2.0,
    transition_secs: float = 1.0,
    draw_boxes: bool = False,
    model=None,
    confidence: float = 0.5,
    target_classes: set[str] | None = None,
) -> dict:
    """
    Write a single merged highlight video from multiple video detection results.

    Iterates over each video result, extracts segments where detections occurred,
    and concatenates them into one output file. The source filename is burned into
    the bottom-left corner of every frame. All frames are resized to match the
    resolution of the first source that has detections.

    When moving from one source video to the next, a short black screen
    transition is inserted showing the filename of the next source as a
    centered title card.

    Args:
        video_results:    list of dicts returned by detect_in_video()
        output_path:      path of the output .mp4 file
        buffer_secs:      seconds of padding before/after each detected frame
        transition_secs:  seconds of black screen between different source videos
        draw_boxes:       re-run YOLO on each highlight frame and draw bounding boxes
        model:            YOLO model instance (required when draw_boxes is True)
        confidence:       detection confidence threshold (used with draw_boxes)
        target_classes:   set of class names to draw (used with draw_boxes)

    Returns:
        {
            "output":             "results/highlights_20260620_103200.mp4",
            "sources_with_hits":  2,
            "total_clips":        5,
            "total_duration_secs": 42.3,
        }
    """
    if draw_boxes and model is None:
        raise ValueError("--clip-boxes requires a YOLO model but none was provided.")

    out_writer: cv2.VideoWriter | None = None
    out_w = out_h = out_fps = None

    sources_with_hits = 0
    total_clips = 0
    total_frames_written = 0

    for result in video_results:
        # Skip results that have no detections or represent errors
        if "error" in result:
            continue
        detections: dict[str, list[dict]] = result.get("detections", {})
        if not any(detections.values()):
            continue

        video_path = result["video"]
        fps        = result["fps"]
        total_fr   = result["total_frames"]
        label      = Path(video_path).name

        segments = _build_segments(detections, fps, total_fr, buffer_secs)
        if not segments:
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [clip] Cannot open {video_path} — skipping.")
            continue

        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # First video with hits sets the output resolution & fps
        if out_writer is None:
            out_w, out_h, out_fps = src_w, src_h, fps
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))
            if not out_writer.isOpened():
                cap.release()
                raise IOError(f"Cannot create output video: {output_path}")
        else:
            # Insert black screen transition between different source videos
            if transition_secs > 0:
                transition_frames = _write_transition(
                    writer=out_writer,
                    width=out_w,
                    height=out_h,
                    fps=out_fps,
                    duration_secs=transition_secs,
                    next_label=label,
                )
                total_frames_written += transition_frames

        needs_resize = (src_w != out_w or src_h != out_h)
        sources_with_hits += 1

        for start_frame, end_frame in segments:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
            total_clips += 1
            for _ in range(end_frame - start_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                if needs_resize:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                if draw_boxes:
                    _draw_boxes(frame, model, confidence, target_classes or set())
                _stamp_label(frame, label)
                out_writer.write(frame)
                total_frames_written += 1

        cap.release()

    if out_writer is not None:
        out_writer.release()

    total_duration = round(total_frames_written / out_fps, 1) if out_fps else 0.0

    return {
        "output":              output_path if out_writer is not None else None,
        "sources_with_hits":   sources_with_hits,
        "total_clips":         total_clips,
        "total_duration_secs": total_duration,
    }
