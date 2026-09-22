#!/usr/bin/env python3
"""
confusion_matrix_holes.py
--------------------------
Runs hole classifier on clips in classified_holes/<Hole_N>/ folders,
compares predicted vs actual (folder = ground truth),
prints a confusion matrix.

Usage:
    python3 src/eval/confusion_matrix_holes.py --config config.json --classified-dir ./test/results/classified_holes
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


def classify_hole(video_path, holes, process_w=540, process_h=960,
                  tolerance=20, confirm_frames=1, cooldown_frames=30):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "no_hole"

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

    hole_streak       = [0]     * len(holes)
    hole_was_in       = [False] * len(holes)
    hole_entry_logged = [False] * len(holes)
    hole_last_entry   = [-999]  * len(holes)
    hole_hit_time     = [None]  * len(holes)

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
                    hole_entry_logged[i] = True
                    hole_last_entry[i]   = frame_idx
                    if hole_hit_time[i] is None:
                        hole_hit_time[i] = frame_idx / fps

        frame_idx += 1

    cap.release()

    earliest_time, earliest_idx = None, None
    for i, t in enumerate(hole_hit_time):
        if t is not None and (earliest_time is None or t < earliest_time):
            earliest_time, earliest_idx = t, i

    return holes[earliest_idx]["name"] if earliest_idx is not None else "Hole_1"  # fallback to closest


# ─────────────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────────────
def print_confusion_matrix(matrix, classes):
    col_w   = 10
    label_w = 12

    print("\n  Confusion Matrix  (rows=Predicted, cols=Actual)\n")
    header = " " * label_w + "".join(f"{c:>{col_w}}" for c in classes)
    print("  " + header)
    print("  " + "─" * len(header))

    for i, row_cls in enumerate(classes):
        row = f"  {row_cls:<{label_w}}"
        for j in range(len(classes)):
            val  = matrix[i][j]
            cell = f"[{val}]" if i == j else f" {val} "
            row += f"{cell:>{col_w}}"
        print(row)

    print("  " + "─" * len(header))

    print("\n  Per-class metrics:")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("  " + "─" * 52)
    n = len(classes)
    for i, cls in enumerate(classes):
        tp = matrix[i][i]
        fp = sum(matrix[i][j] for j in range(n) if j != i)
        fn = sum(matrix[j][i] for j in range(n) if j != i)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
        support   = tp + fn
        print(f"  {cls:<12} {precision:>10.2f} {recall:>10.2f} {f1:>10.2f} {support:>10}")

    total = sum(matrix[i][i] for i in range(n))
    grand = sum(matrix[i][j] for i in range(n) for j in range(n))
    print(f"\n  Overall accuracy: {total}/{grand} = {total/max(grand,1)*100:.1f}%")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Confusion matrix for hole detection.")
    parser.add_argument("--config",         default="config.json")
    parser.add_argument("--classified-dir", default="./test/results/classified_holes")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    holes   = cfg.get("holes", [])
    classes = [h["name"] for h in holes]
    print(f"  Classes: {classes}")

    # Collect clips from each subfolder
    all_clips = []
    for cls in classes:
        folder = os.path.join(args.classified_dir, cls)
        if not os.path.isdir(folder):
            print(f"  WARNING: folder not found: {folder}")
            continue
        clips = sorted(glob(os.path.join(folder, "*.MP4")) +
                       glob(os.path.join(folder, "*.mp4")))
        for c in clips:
            all_clips.append((c, cls))
        print(f"  {cls:12s}: {len(clips)} clips")

    if not all_clips:
        print("No clips found.")
        return

    print(f"\n  Running hole classifier on {len(all_clips)} clips...\n")

    cls_idx = {c: i for i, c in enumerate(classes)}
    n       = len(classes)
    matrix  = [[0]*n for _ in range(n)]
    wrong   = []

    for idx, (video_path, true_label) in enumerate(all_clips):
        name = Path(video_path).name
        print(f"  [{idx+1:3d}/{len(all_clips)}] {name:40s} (true={true_label:10s}) ...", end=" ", flush=True)
        predicted = classify_hole(video_path, holes)
        correct = predicted == true_label
        print(f"pred={predicted:10s}" + (" ✓" if correct else " ✗"))

        row = cls_idx.get(predicted, n-1)
        col = cls_idx.get(true_label, n-1)
        matrix[row][col] += 1

        if not correct:
            wrong.append((name, true_label, predicted))

    print_confusion_matrix(matrix, classes)

    if wrong:
        print(f"\n  Misclassified ({len(wrong)}):")
        for name, true, pred in wrong:
            print(f"    {name}: true={true}  pred={pred}")


if __name__ == "__main__":
    main()