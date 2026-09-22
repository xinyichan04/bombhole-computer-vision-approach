#!/usr/bin/env python3
"""
classify_clips.py
-----------------
Classifies each throw clip into one of 6 classes:
  - Pedal_1 ... Pedal_5  (ball collides with that pedal)
  - miss                 (no collision detected)

Uses the same ball tracking + collision detection logic from collision_detector.py.
Reads pedal config from game_config.json (or --config).

Usage:
    python3 src/classify_clips.py source/clips/*.MP4 --config config.json
    python3 src/classify_clips.py source/clips/*.MP4 --config config.json --output test/results/results.csv
    python3 src/classify_clips.py source/clips/*.MP4 --config config.json --copy-to ./test/results/classified

Output:
    - Prints classification for each clip
    - Optionally saves CSV with: filename, class, pedal, time_sec, confidence
    - Optionally copies clips into subfolders by class (e.g. ./test/results/classified/Pedal_1/)
"""

import cv2
import numpy as np
import argparse
import os
import json
import csv
from pathlib import Path
from glob import glob


# ─────────────────────────────────────────────
# Ball Tracker (same as collision_detector.py)
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
# Classifier
# ─────────────────────────────────────────────
def classify_clip(video_path, pedals, process_w=540, process_h=960):
    """
    Run ball tracking on a clip and return the first pedal hit.

    Returns dict:
        {
            "class":      "Pedal_1" | ... | "Pedal_5" | "miss",
            "pedal":      pedal dict or None,
            "time_sec":   float or None,
            "min_dist":   closest approach distance to any pedal (px),
            "closest_pedal": name of closest pedal even if not hit,
        }
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"class": "miss", "pedal": None, "time_sec": None,
                "min_dist": 9999, "closest_pedal": None}

    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    sx = orig_w / process_w
    sy = orig_h / process_h

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=20, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kf = create_kalman()
    kalman_initialized = False
    missed_frames = 0
    MAX_MISSED = 10

    # Per-pedal: track consecutive frames inside ROI
    pedal_streak   = [0] * len(pedals)
    pedal_hit_time = [None] * len(pedals)
    CONFIRM_FRAMES = 1   # frames inside ROI to confirm hit

    # Track closest approach for diagnostics
    global_min_dist  = 9999.0
    global_closest   = None

    result_class     = "miss"
    result_pedal     = None
    result_time      = None
    frame_idx        = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        time_sec = frame_idx / fps

        # ── Ball tracking ──────────────────────────
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
                px = int(float(pred[0]) * sx)
                py = int(float(pred[1]) * sy)
                ball_pos = (px, py, 15)
            elif missed_frames >= MAX_MISSED:
                kalman_initialized = False
                pedal_streak = [0] * len(pedals)

        # ── Pedal collision check ──────────────────
        if ball_pos:
            bx, by, _ = ball_pos

            for i, p in enumerate(pedals):
                px1, py1 = p["x"], p["y"]
                px2, py2 = p["x"] + p["w"], p["y"] + p["h"]
                pcx = (px1 + px2) / 2
                pcy = (py1 + py2) / 2

                # Distance from ball to pedal center (for diagnostics)
                dist = float(np.sqrt((bx - pcx)**2 + (by - pcy)**2))
                if dist < global_min_dist:
                    global_min_dist = dist
                    global_closest  = p["name"]

                # ROI check
                in_roi = (px1 <= bx <= px2 and py1 <= by <= py2)
                if in_roi:
                    pedal_streak[i] += 1
                else:
                    pedal_streak[i] = 0

                if pedal_streak[i] >= CONFIRM_FRAMES and pedal_hit_time[i] is None:
                    pedal_hit_time[i] = time_sec

        frame_idx += 1

    cap.release()

    # ── Determine class ────────────────────────────
    # Pick the earliest confirmed hit
    earliest_time = None
    earliest_idx  = None
    for i, t in enumerate(pedal_hit_time):
        if t is not None:
            if earliest_time is None or t < earliest_time:
                earliest_time = t
                earliest_idx  = i

    if earliest_idx is not None:
        result_class = pedals[earliest_idx]["name"]
        result_pedal = pedals[earliest_idx]
        result_time  = earliest_time

    return {
        "class":          result_class,
        "pedal":          result_pedal,
        "time_sec":       result_time,
        "min_dist":       round(global_min_dist, 1),
        "closest_pedal":  global_closest,
    }


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Classify throw clips into pedal hit or miss.")
    parser.add_argument("inputs",       nargs="+", help="Clip file(s) or glob pattern")
    parser.add_argument("--config",     default="config.json",
                        help="Path to config JSON with pedal definitions (default: config.json)")
    parser.add_argument("--output",     default=None,
                        help="Save results to CSV file (e.g. results.csv)")
    parser.add_argument("--copy-to",    default=None,
                        help="Copy clips into class subfolders (e.g. ./test/results/classified)")
    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"ERROR: Config not found at {args.config}")
        return
    with open(args.config) as f:
        cfg = json.load(f)
    pedals = cfg.get("pedals", [])
    print(f"  Loaded {len(pedals)} pedals from {args.config}")
    print(f"  Classes: {[p['name'] for p in pedals]} + miss\n")

    # Collect files
    files = []
    for pattern in args.inputs:
        matched = glob(pattern) if "*" in pattern else [pattern]
        files.extend(matched)
    files = sorted(set(files))

    if not files:
        print("No files found.")
        return

    # Class counts for summary
    class_counts = {p["name"]: 0 for p in pedals}
    class_counts["miss"] = 0

    results = []

    for i, f in enumerate(files):
        if not os.path.exists(f):
            print(f"  SKIP: {f} not found")
            continue

        print(f"  [{i+1}/{len(files)}] {Path(f).name} ...", end=" ", flush=True)
        result = classify_clip(f, pedals)

        cls       = result["class"]
        time_str  = f"{result['time_sec']:.2f}s" if result["time_sec"] else "—"
        dist_str  = f"dist={result['min_dist']:.0f}px to {result['closest_pedal']}"

        print(f"→ {cls:12s}  hit@{time_str}  ({dist_str})")

        class_counts[cls] = class_counts.get(cls, 0) + 1
        results.append({
            "filename":       Path(f).name,
            "class":          cls,
            "hit_time_sec":   result["time_sec"] if result["time_sec"] else "",
            "min_dist_px":    result["min_dist"],
            "closest_pedal":  result["closest_pedal"] or "",
        })

        # Copy to subfolder if requested
        if args.copy_to:
            dest_dir = os.path.join(args.copy_to, cls)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, Path(f).name)
            import shutil
            shutil.copy2(f, dest)

    # Summary
    print(f"\n{'─'*45}")
    print(f"  CLASSIFICATION SUMMARY ({len(results)} clips)")
    print(f"{'─'*45}")
    for cls, count in sorted(class_counts.items()):
        bar = "█" * count
        print(f"  {cls:12s}: {count:3d}  {bar}")
    print(f"{'─'*45}")

    # Save CSV
    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Results saved to {args.output}")

    if args.copy_to:
        print(f"  Clips copied to {args.copy_to}/<class>/")


if __name__ == "__main__":
    main()