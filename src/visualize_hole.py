#!/usr/bin/env python3
"""
visualize_holes.py
------------------
Runs ball tracking on a clip, detects which hole the ball enters,
and saves an annotated video showing:
  - Ball trajectory trail
  - Hole circles with labels and scores
  - Score HUD
  - Event banner when hole entry confirmed

Usage:
    python3 src/visualize_hole.py source/clips/GX019502_throw16.MP4 --config config.json
    python3 src/visualize_hole.py source/clips/*.MP4 --config config.json --output-dir ./test/results/hole_viz
"""

import cv2
import numpy as np
import argparse
import os
import json
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
# Colors
# ─────────────────────────────────────────────
HOLE_COLORS = [
    (0,   80,  255),   # blue
    (255, 80,  0  ),   # orange
    (80,  255, 80 ),   # green
    (200, 0,   200),   # violet
    (0,   200, 200),   # teal
    (255, 200, 0  ),   # gold
]


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────
def run(video_path, holes, output_dir="test/results/hole_viz",
        process_w=540, process_h=960,
        tolerance=20, confirm_frames=1, cooldown_frames=30):

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sx = orig_w / process_w
    sy = orig_h / process_h

    print(f"\n  {Path(video_path).name}  {orig_w}x{orig_h} @ {fps:.0f}fps  {total} frames")

    # Output video
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, Path(video_path).stem + "_holes.MP4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"),
                             fps, (orig_w, orig_h))
    print(f"  Saving to: {out_path}")

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=20, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kf = create_kalman()
    kalman_initialized = False
    missed_frames = 0
    MAX_MISSED = 10
    trajectory = []

    # Per-hole state
    hole_streak       = [0]     * len(holes)
    hole_was_in       = [False] * len(holes)
    hole_entry_logged = [False] * len(holes)
    hole_last_entry   = [-999]  * len(holes)
    hole_scores       = [0]     * len(holes)   # accumulated score per hole

    total_score = 0
    events      = []   # list of (frame_idx, hole_name, score)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        time_sec = frame_idx / fps

        # ── Ball tracking ──────────────────────────────────────
        small   = cv2.resize(frame, (process_w, process_h))
        fg_mask = bg_sub.apply(small)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        candidate = find_ball_candidate(fg_mask)
        ball_pos  = None
        detected  = False

        if candidate:
            cx, cy, r = candidate
            ox, oy = int(cx*sx), int(cy*sy)
            or_    = int(r * max(sx, sy))
            meas   = np.array([[np.float32(cx)], [np.float32(cy)]])
            if not kalman_initialized:
                kf.statePre = np.array([[cx],[cy],[0],[0]], dtype=np.float32)
                kalman_initialized = True
            kf.correct(meas)
            kf.predict()
            missed_frames = 0
            detected  = True
            ball_pos  = (ox, oy, or_)
            trajectory.append((ox, oy))
        else:
            missed_frames += 1
            if kalman_initialized and missed_frames < MAX_MISSED:
                pred = kf.predict()
                px = int(float(pred[0].flat[0]) * sx)
                py = int(float(pred[1].flat[0]) * sy)
                ball_pos = (px, py, 15)
                trajectory.append((px, py))
            elif missed_frames >= MAX_MISSED:
                kalman_initialized = False
                trajectory = []
                hole_entry_logged = [False] * len(holes)
                hole_streak       = [0]     * len(holes)
                hole_was_in       = [False] * len(holes)

        # ── Hole detection ─────────────────────────────────────
        frame_events = []
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
                    hole_entry_logged[i] = True
                    hole_last_entry[i]   = frame_idx
                    hole_scores[i]      += h["score"]
                    total_score         += h["score"]
                    ev = (frame_idx, h["name"], h["score"])
                    events.append(ev)
                    frame_events.append(ev)
                    print(f"  🕳️  {time_sec:.2f}s  {h['name']} +{h['score']}pts  (total:{total_score})")

        # ── Visualization ──────────────────────────────────────
        vis = frame.copy()

        # Trajectory trail
        for j in range(1, len(trajectory)):
            alpha = j / len(trajectory)
            cv2.line(vis, trajectory[j-1], trajectory[j],
                     (0, int(255*alpha), int(255*(1-alpha))), 2)

        # Ball
        if ball_pos:
            bx, by, br = ball_pos
            col = (0,255,0) if detected else (0,165,255)
            cv2.circle(vis, (bx, by), max(br, 12), col, 3)
            cv2.circle(vis, (bx, by), 5, col, -1)
            cv2.putText(vis, "DETECTED" if detected else "PREDICTED",
                        (bx+16, by), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        # Hole circles
        for i, h in enumerate(holes):
            color = HOLE_COLORS[i % len(HOLE_COLORS)]
            hcx, hcy, hr = h["cx"], h["cy"], h["r"]

            # Pulse ring when ball is inside
            if hole_streak[i] > 0:
                cv2.circle(vis, (hcx, hcy), hr + tolerance, color, 3)
                # flash fill
                ov = vis.copy()
                cv2.circle(ov, (hcx, hcy), hr + tolerance, color, -1)
                cv2.addWeighted(ov, 0.2, vis, 0.8, 0, vis)
            else:
                cv2.circle(vis, (hcx, hcy), hr + tolerance, color, 2)

            # Label
            lbl = f"{h['name']} [{h['score']}pt]"
            cv2.putText(vis, lbl, (hcx - 40, hcy - (hr + tolerance) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            # Score badge if confirmed
            if hole_entry_logged[i]:
                cv2.putText(vis, f"IN +{h['score']}", (hcx - 40, hcy + (hr + tolerance) + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 100), 2)

        # Top HUD bar
        cv2.rectangle(vis, (0, 0), (orig_w, 130), (0, 0, 0), -1)
        cv2.putText(vis, f"SCORE: {total_score}",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.8, (0, 255, 255), 5)
        ts = f"{time_sec:.2f}s | Frame {frame_idx}"
        (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(vis, ts, (orig_w - tw - 30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
        bs  = "● TRACKING" if detected else ("◌ PREDICTING" if ball_pos else "✗ LOST")
        bsc = (0,255,0) if detected else ((0,165,255) if ball_pos else (0,0,255))
        cv2.putText(vis, bs, (30, 122), cv2.FONT_HERSHEY_SIMPLEX, 1.0, bsc, 2)

        # Bottom event banner
        if frame_events:
            ev = frame_events[-1]
            banner = f"+{ev[2]}pts  {ev[1]} HOLE IN!"
            cv2.rectangle(vis, (0, orig_h - 80), (orig_w, orig_h), (0, 0, 0), -1)
            (tw, _), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)
            cv2.putText(vis, banner, ((orig_w - tw) // 2, orig_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (100, 255, 100), 3)

        writer.write(vis)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"\n  Done — {len(events)} hole event(s), final score: {total_score}")
    return events, total_score


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Visualize ball tracking + hole entry detection.")
    parser.add_argument("inputs",        nargs="+", help="Clip file(s)")
    parser.add_argument("--config",      default="config.json")
    parser.add_argument("--output-dir",  default="./test/results/hole_viz")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    holes = cfg.get("holes", [])
    print(f"  Loaded {len(holes)} holes from {args.config}")

    files = []
    for pattern in args.inputs:
        files.extend(glob(pattern) if "*" in pattern else [pattern])
    files = sorted(set(f for f in files if os.path.exists(f)))

    for f in files:
        run(f, holes, output_dir=args.output_dir)


if __name__ == "__main__":
    main()