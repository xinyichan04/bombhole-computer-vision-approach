#!/usr/bin/env python3
"""
classify_holes.py
-----------------
Classifies each throw clip by which hole the ball entered.
Classes: Hole_1, Hole_2, Hole_3, Hole_4, ... or "no_hole"

Copies clips into subfolders:
    classified_holes/Hole_1/
    classified_holes/Hole_2/
    ...
    classified_holes/no_hole/

Usage:
    python3 src/classify_holes.py source/clips/*.MP4 --config config.json
    python3 src/classify_holes.py source/clips/*.MP4 --config config.json --copy-to ./test/results/classified_holes --output test/results/hole_results.csv
"""

import cv2
import numpy as np
import argparse
import os
import json
import csv
import shutil
from pathlib import Path
from glob import glob


# ─────────────────────────────────────────────
# Ball Tracker
# ─────────────────────────────────────────────
def create_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
    kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float32)
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
    kf.errorCovPost        = np.eye(4, dtype=np.float32)
    return kf


def find_ball_candidate(fg_mask, min_area=30, max_area=3000, min_circularity=0.3):
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < min_circularity:
            continue
        score = circularity * area
        if score > best_score:
            best_score = score
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            best = (int(x), int(y), int(radius))
    return best


# ─────────────────────────────────────────────
# Hole Classifier
# ─────────────────────────────────────────────
def classify_hole(video_path, holes, process_w=540, process_h=960,
                  tolerance=20, confirm_frames=1, cooldown_frames=30):
    """
    Returns the first hole the ball entered, or "no_hole".
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "no_hole", None, None

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    sx = orig_w / process_w
    sy = orig_h / process_h

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=20, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kf = create_kalman()
    kalman_initialized = False
    missed_frames = 0
    MAX_MISSED = 10

    hole_streak         = [0]     * len(holes)
    hole_was_in         = [False] * len(holes)
    hole_entry_logged   = [False] * len(holes)
    hole_last_entry     = [-999]  * len(holes)
    hole_hit_time       = [None]  * len(holes)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small   = cv2.resize(frame, (process_w, process_h))
        fg_mask = bg_sub.apply(small)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        candidate = find_ball_candidate(fg_mask)
        ball_pos  = None

        if candidate:
            cx, cy, r = candidate
            ox, oy = int(cx*sx), int(cy*sy)
            meas = np.array([[np.float32(cx)], [np.float32(cy)]])
            if not kalman_initialized:
                kf.statePre = np.array([[cx],[cy],[0],[0]], dtype=np.float32)
                kalman_initialized = True
            kf.correct(meas)
            kf.predict()
            missed_frames = 0
            ball_pos = (ox, oy, int(r * max(sx, sy)))
        else:
            missed_frames += 1
            if kalman_initialized and missed_frames < MAX_MISSED:
                pred = kf.predict()
                px = int(float(pred[0].flat[0]) * sx)
                py = int(float(pred[1].flat[0]) * sy)
                ball_pos = (px, py, 15)
            elif missed_frames >= MAX_MISSED:
                kalman_initialized = False
                hole_entry_logged = [False] * len(holes)
                hole_streak       = [0]     * len(holes)
                hole_was_in       = [False] * len(holes)

        if ball_pos:
            bx, by, _ = ball_pos
            for i, h in enumerate(holes):
                dist = float(np.sqrt((bx - h["cx"])**2 + (by - h["cy"])**2))
                in_hole = dist <= (h["r"] + tolerance)

                if in_hole:
                    hole_streak[i] += 1
                else:
                    if hole_was_in[i]:
                        hole_entry_logged[i] = False
                    hole_streak[i] = 0
                hole_was_in[i] = in_hole

                cooldown_ok = (frame_idx - hole_last_entry[i]) > cooldown_frames
                if hole_streak[i] >= confirm_frames and not hole_entry_logged[i] and cooldown_ok:
                    hole_entry_logged[i]  = True
                    hole_last_entry[i]    = frame_idx
                    if hole_hit_time[i] is None:
                        hole_hit_time[i] = frame_idx / fps

        frame_idx += 1

    cap.release()

    # Return earliest hole entry
    earliest_time, earliest_idx = None, None
    for i, t in enumerate(hole_hit_time):
        if t is not None and (earliest_time is None or t < earliest_time):
            earliest_time, earliest_idx = t, i

    if earliest_idx is not None:
        return holes[earliest_idx]["name"], earliest_time, earliest_idx
    return "no_hole", None, None


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Classify clips by hole entry.")
    parser.add_argument("inputs",       nargs="+", help="Clip file(s) or glob pattern")
    parser.add_argument("--config",     default="config.json")
    parser.add_argument("--copy-to",    default="./test/results/classified_holes",
                        help="Root folder for sorted subfolders (default: ./classified_holes)")
    parser.add_argument("--output",     default=None, help="Save results to CSV")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    holes = cfg.get("holes", [])
    if not holes:
        print("No holes found in config.")
        return

    classes = [h["name"] for h in holes] + ["no_hole"]
    print(f"  Holes: {[h['name'] for h in holes]}")

    # Collect files
    files = []
    for pattern in args.inputs:
        matched = glob(pattern) if "*" in pattern else [pattern]
        files.extend(matched)
    files = sorted(set(f for f in files if os.path.exists(f)))
    print(f"  Processing {len(files)} clips...\n")

    class_counts = {c: 0 for c in classes}
    results = []

    for idx, f in enumerate(files):
        name = Path(f).name
        print(f"  [{idx+1:3d}/{len(files)}] {name} ...", end=" ", flush=True)

        cls, time_sec, _ = classify_hole(f, holes)
        time_str = f"{time_sec:.2f}s" if time_sec else "—"
        print(f"→ {cls:12s}  entry@{time_str}")

        class_counts[cls] = class_counts.get(cls, 0) + 1
        results.append({"filename": name, "class": cls,
                        "entry_time_sec": time_sec if time_sec else ""})

        # Copy to subfolder
        dest_dir = os.path.join(args.copy_to, cls)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(f, os.path.join(dest_dir, name))

    # Summary
    print(f"\n{'─'*45}")
    print(f"  HOLE CLASSIFICATION SUMMARY ({len(files)} clips)")
    print(f"{'─'*45}")
    for cls in classes:
        count = class_counts.get(cls, 0)
        print(f"  {cls:12s}: {count:3d}  {'█' * count}")
    print(f"{'─'*45}")
    print(f"  Sorted into: {args.copy_to}/<hole>/")

    if args.output and results:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"  CSV saved to {args.output}")


if __name__ == "__main__":
    main()
