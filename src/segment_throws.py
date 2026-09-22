#!/usr/bin/env python3
"""
segment_throws.py
-----------------
Automatically detects and segments throwing events from GoPro footage.
Works on both long videos (many throws) and short clips (trims dead space).

Usage:
    python3 src/segment_throws.py source/raw/input.MP4
    python3 src/segment_throws.py source/raw/input.MP4 --output-dir ./source/clips
    python3 src/segment_throws.py *.mp4           # batch process multiple files

Arguments:
    input               Input video file(s)
    --output-dir        Where to save clips (default: ./throws_output)
    --padding           Seconds to add before/after each throw (default: 0.5)
    --motion-threshold  Motion sensitivity (default: 1.4, lower = more sensitive)
    --min-throw-dur     Minimum throw duration in seconds (default: 0.3)
    --merge-gap         Merge events closer than this many seconds (default: 1.0)
    --preview           Print detected events without cutting video
    --scale             Downscale factor for motion analysis, higher = faster (default: 8)
"""

import cv2
import numpy as np
import subprocess
import argparse
import os
import sys
from pathlib import Path


def detect_throw_events(video_path, motion_threshold=1.4, min_throw_dur=0.3,
                        merge_gap=1.0, scale=8, verbose=True):
    """
    Analyze video for throw events using frame differencing.
    Returns list of (start_sec, end_sec) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    if verbose:
        print(f"\n  Video: {Path(video_path).name}")
        print(f"  Resolution: {width}x{height} | FPS: {fps:.2f} | Duration: {duration:.2f}s | Frames: {total_frames}")
        print(f"  Analyzing motion...")

    target_w = max(width // scale, 160)
    target_h = max(height // scale, 90)

    motion_scores = []
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (target_w, target_h))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_scores.append(float(diff.mean()))
        else:
            motion_scores.append(0.0)
        prev_gray = gray

    cap.release()

    # Smooth scores with a small window to reduce noise
    scores = np.array(motion_scores)
    kernel_size = max(3, int(fps * 0.05))  # smaller kernel to preserve peaks
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(scores, kernel, mode='same')

    # Find frames above threshold (use raw scores for detection, smoothed for display)
    active = scores > motion_threshold

    # Find contiguous active regions (raw events)
    raw_events = []
    in_event = False
    start_frame = 0
    for i, a in enumerate(active):
        if a and not in_event:
            in_event = True
            start_frame = i
        elif not a and in_event:
            in_event = False
            raw_events.append((start_frame / fps, i / fps))
    if in_event:
        raw_events.append((start_frame / fps, len(active) / fps))

    # Merge events that are close together FIRST
    merged = []
    for s, e in raw_events:
        if merged and (s - merged[-1][1]) < merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append([s, e])

    # Then filter by minimum duration
    events = [(s, e) for s, e in merged if (e - s) >= min_throw_dur]

    if verbose:
        print(f"  Detected {len(events)} throw event(s)")

    return events, fps, duration


def cut_clip(input_path, output_path, start_sec, end_sec, padding=0.5, total_duration=None):
    """Cut a clip using FFmpeg stream copy (no re-encoding, fast, lossless)."""
    actual_start = max(0.0, start_sec - padding)
    actual_end = end_sec + padding
    if total_duration:
        actual_end = min(actual_end, total_duration)
    duration = actual_end - actual_start

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{actual_start:.3f}",
        "-i", input_path,
        "-t", f"{duration:.3f}",
        "-c", "copy",        # stream copy = no re-encoding, very fast
        "-avoid_negative_ts", "make_zero",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    WARNING: ffmpeg error: {result.stderr[-200:]}")
        return False
    return True


def process_video(video_path, output_dir, padding=0.5, motion_threshold=1.4,
                  min_throw_dur=0.3, merge_gap=1.0, scale=8, preview=False):
    """Process a single video file."""
    video_path = str(video_path)
    stem = Path(video_path).stem
    suffix = Path(video_path).suffix

    events, fps, duration = detect_throw_events(
        video_path, motion_threshold, min_throw_dur, merge_gap, scale
    )

    if not events:
        print(f"  No throw events detected. Try lowering --motion-threshold.")
        return 0

    print(f"\n  Detected events:")
    for i, (s, e) in enumerate(events):
        print(f"    Throw {i+1}: {s:.2f}s → {e:.2f}s  (duration: {e-s:.2f}s)")

    if preview:
        print("  [Preview mode — no files written]")
        return len(events)

    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    for i, (s, e) in enumerate(events):
        out_name = f"{stem}_throw{i+1:02d}{suffix}"
        out_path = os.path.join(output_dir, out_name)
        print(f"  Cutting throw {i+1} → {out_name} ...", end=" ", flush=True)
        success = cut_clip(video_path, out_path, s, e, padding, duration)
        if success:
            print("✓")
            saved += 1
        else:
            print("✗ failed")

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Segment throwing events from video footage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("inputs", nargs="+", help="Input video file(s)")
    parser.add_argument("--output-dir", default="./source/clips",
                        help="Output directory (default: ./throws_output)")
    parser.add_argument("--padding", type=float, default=0.5,
                        help="Seconds of buffer before/after each throw (default: 0.5)")
    parser.add_argument("--motion-threshold", type=float, default=0.9,
                        help="Motion sensitivity threshold (default: 0.9, lower = more sensitive)")
    parser.add_argument("--min-throw-dur", type=float, default=0.3,
                        help="Minimum throw duration in seconds (default: 0.3)")
    parser.add_argument("--merge-gap", type=float, default=1.0,
                        help="Merge events within this many seconds (default: 1.0)")
    parser.add_argument("--scale", type=int, default=8,
                        help="Downscale factor for analysis (default: 8, higher = faster)")
    parser.add_argument("--preview", action="store_true",
                        help="Print detected events without cutting video")

    args = parser.parse_args()

    total_clips = 0
    for pattern in args.inputs:
        # Support glob patterns
        from glob import glob
        files = glob(pattern) if "*" in pattern else [pattern]
        for f in files:
            if not os.path.exists(f):
                print(f"WARNING: File not found: {f}")
                continue
            print(f"\nProcessing: {f}")
            n = process_video(
                f,
                output_dir=args.output_dir,
                padding=args.padding,
                motion_threshold=args.motion_threshold,
                min_throw_dur=args.min_throw_dur,
                merge_gap=args.merge_gap,
                scale=args.scale,
                preview=args.preview,
            )
            total_clips += n

    print(f"\n{'─'*40}")
    if args.preview:
        print(f"Total throw events found: {total_clips}")
    else:
        print(f"Total clips saved: {total_clips} → {args.output_dir}/")


if __name__ == "__main__":
    main()
