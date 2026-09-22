#!/usr/bin/env python3
"""
confusion_matrix_v3.py
----------------------
v1 pipeline + three targeted fixes identified from debug traces:
  Fix 1: lower spark threshold 0.04->0.02, full-frame suppression check
  Fix 2: preserve collision_logged on ball loss (don't wipe hit record)
  Fix 3: freeze Kalman during pending spark confirmation window

Usage:
    python3 src/eval/confusion_matrix_v3.py --config config.json --classified-dir ./test/results/classifiedFlash
"""

import cv2
import numpy as np
import argparse
import os
import json
from pathlib import Path
from glob import glob


# ═══════════════════════════════════════════════════════════════
#  Full pipeline — copied from collision_detector.py
#  (no visualization, no video writing — eval-only mode)
# ═══════════════════════════════════════════════════════════════

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


def detect_spark(frame, roi, threshold=0.02):
    """Detect yellow/orange spark in pedal ROI (x, y, w, h)."""
    x, y, w, h = roi
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x+w), min(frame.shape[0], y+h)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return False, 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10,  100, 150])
    upper_yellow = np.array([45,  255, 255])
    lower_white  = np.array([0,   0,   220])
    upper_white  = np.array([180, 40,  255])
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_yellow, upper_yellow),
        cv2.inRange(hsv, lower_white,  upper_white)
    )
    intensity = cv2.countNonZero(mask) / max((x2-x1)*(y2-y1), 1)
    return intensity > threshold, round(intensity, 4)


def detect_hole_entry(frame, hole, ball_pos):
    """
    Detect if ball center is within the hole circle (+tolerance).
    hole: {cx, cy, r, score, name}
    """
    if ball_pos is None:
        return False, 0.0
    bx, by, _ = ball_pos
    dist = float(np.sqrt((bx - hole["cx"])**2 + (by - hole["cy"])**2))
    TOLERANCE = 20
    return dist <= (hole["r"] + TOLERANCE), round(dist, 1)


def global_spark_intensity(frame):
    """
    Measure yellow/white spark pixel ratio across the FULL frame.
    (v3 fix: was bottom-50% only, missed sparks at mid-frame pedal height)
    """
    region = frame
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10,  100, 150])
    upper_yellow = np.array([45,  255, 255])
    lower_white  = np.array([0,   0,   220])
    upper_white  = np.array([180, 40,  255])
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_yellow, upper_yellow),
        cv2.inRange(hsv, lower_white,  upper_white)
    )
    return cv2.countNonZero(mask) / max(region.shape[0] * region.shape[1], 1)


# v3 fix: lowered from 0.04 — observed spark frames read 0.009–0.060,
# full-frame check means threshold can be lower without false positives.
SPARK_SUPPRESS_THRESHOLD = 0.02


def classify_clip_full(video_path, pedals, holes,
                       process_w=540, process_h=960,
                       spark_confirm_frames=1,
                       hole_confirm_frames=1):
    """
    Run the full scoring pipeline on one clip (no video output).
    Returns the name of the FIRST scoring event target, or "miss".

    v3 fixes (on top of v1):

    Fix 1 — Lower thresholds + full-frame spark suppression:
        detect_spark threshold: 0.04 → 0.02
        global_spark_intensity: full frame instead of bottom-50%
        SPARK_SUPPRESS_THRESHOLD: 0.04 → 0.02
        Observed spark frames read 0.009–0.060, old threshold was too high.

    Fix 2 — Preserve collision_logged on ball loss:
        Old code reset collision_logged=False for ALL pedals on ball loss.
        Now only resets spark_streak for pedals not yet hit — a logged
        collision is never erased by tracker loss.

    Fix 3 — Freeze Kalman during pending spark confirmation:
        Once any pedal sets collision_logged=True, skip find_ball_candidate
        until spark is confirmed. Prevents spark blob from hijacking the
        tracker and corrupting ball_pos during the impact window.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "miss"

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sx = orig_w / process_w
    sy = orig_h / process_h

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=20, detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kf = create_kalman()
    kalman_initialized = False
    missed_frames = 0
    MAX_MISSED = 10

    COLLISION_FREEZE_FRAMES = 15  # frames to hold Kalman after collision

    # ── Pedal state ───────────────────────────────────────────
    pedal_state = [{
        "spark_streak":     0,
        "collision_logged": False,
        "collision_frame":  -1,
        "spark_logged":     False,
        "freeze_remaining": 0,    # v3: Kalman freeze counter post-collision
    } for _ in pedals]

    # ── Hole state (mirrors collision_detector.py exactly) ──
    hole_state = [{
        "entry_streak":     0,
        "entry_logged":     False,
        "was_in_hole":      False,
        "throw_count":      0,
        "last_entry_frame": -999,
    } for _ in holes]

    # Collect all scoring events: (frame_idx, target_name)
    scoring_events = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Spark suppression check (Fix 1) ───────────────────
        # If the bottom half of the frame is flooded with yellow/white
        # spark pixels, ball detection is unreliable this frame.
        # We force Kalman-prediction-only mode to keep position stable.
        spark_active = global_spark_intensity(frame) > SPARK_SUPPRESS_THRESHOLD

        # ── Ball tracking ──────────────────────────────────────
        small   = cv2.resize(frame, (process_w, process_h))
        fg_mask = bg_sub.apply(small)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # v3 fix: if any pedal collision is pending spark confirmation,
        # freeze Kalman — don't let spark blob corrupt ball position.
        any_collision_pending = any(
            s["collision_logged"] and not s["spark_logged"]
            for s in pedal_state
        )

        # Only attempt detection when spark is NOT saturating the frame
        # AND no collision is pending (freeze window)
        candidate = find_ball_candidate(fg_mask) \
            if not spark_active and not any_collision_pending else None
        ball_pos  = None

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
            ball_pos = (ox, oy, or_)
        else:
            # spark_active frames count as missed for Kalman but do NOT
            # increment the persistent missed_frames counter — we don't
            # want MAX_MISSED to trigger a full state reset mid-impact.
            if not spark_active:
                missed_frames += 1
            if kalman_initialized and (missed_frames < MAX_MISSED or spark_active):
                pred = kf.predict()
                px = int(float(pred[0].flat[0]) * sx)
                py = int(float(pred[1].flat[0]) * sy)
                ball_pos = (px, py, 15)
            elif missed_frames >= MAX_MISSED:
                kalman_initialized = False
                # v3 fix: on ball loss, only reset pedals that haven't
                # been hit yet. collision_logged=True means we already
                # know which pedal was hit — don't erase that on tracker loss.
                for s in pedal_state:
                    if not s["collision_logged"]:
                        s["spark_streak"] = 0
                    # never reset collision_logged — preserve hit record
                for s in hole_state:
                    s["entry_logged"] = False
                    s["entry_streak"] = 0
                    s["was_in_hole"]  = False

        # ── Pedal detection: collision + spark confirmation ────
        for i, (p, state) in enumerate(zip(pedals, pedal_state)):
            ball_in_roi = False
            if ball_pos:
                bx, by, _ = ball_pos
                ball_in_roi = (p["x"] <= bx <= p["x"]+p["w"] and
                               p["y"] <= by <= p["y"]+p["h"])

            if ball_in_roi and not state["collision_logged"]:
                state["collision_logged"] = True
                state["collision_frame"]  = frame_idx

            if state["collision_logged"] and not state["spark_logged"]:
                roi = (p["x"], p["y"], p["w"], p["h"])
                spark_ok, _ = detect_spark(frame, roi)
                state["spark_streak"] = state["spark_streak"]+1 if spark_ok else 0
                if state["spark_streak"] >= spark_confirm_frames:
                    state["spark_logged"] = True
                    scoring_events.append((frame_idx, p["name"]))
            else:
                if not state["collision_logged"]:
                    state["spark_streak"] = 0

        # ── Hole detection ─────────────────────────────────────
        # Run hole logic to keep tracking state consistent with collision_detector.py,
        # but do NOT add hole events to scoring_events — pedal-only evaluation.
        for i, (h, state) in enumerate(zip(holes, hole_state)):
            in_hole, _ = detect_hole_entry(frame, h, ball_pos)

            if in_hole:
                state["entry_streak"] += 1
            else:
                if state["was_in_hole"]:
                    state["entry_logged"] = False
                state["entry_streak"] = 0
            state["was_in_hole"] = in_hole

            COOLDOWN_FRAMES = 30
            cooldown_ok = (frame_idx - state["last_entry_frame"]) > COOLDOWN_FRAMES
            if (state["entry_streak"] >= hole_confirm_frames
                    and not state["entry_logged"]
                    and cooldown_ok):
                state["entry_logged"]     = True
                state["last_entry_frame"] = frame_idx
                state["throw_count"]     += 1
                # intentionally NOT appended to scoring_events

        frame_idx += 1

    cap.release()

    if not scoring_events:
        return "miss"

    # Return the target from the EARLIEST scoring event
    scoring_events.sort(key=lambda e: e[0])
    return scoring_events[0][1]


# ═══════════════════════════════════════════════════════════════
#  Confusion Matrix Printer
# ═══════════════════════════════════════════════════════════════

def print_confusion_matrix(matrix, classes):
    col_w   = 12
    label_w = 14

    print("\n  Confusion Matrix  (rows = Predicted, cols = Actual)\n")
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
    print(f"  {'Class':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("  " + "─" * 54)
    n = len(classes)
    for i, cls in enumerate(classes):
        tp = matrix[i][i]
        fp = sum(matrix[i][j] for j in range(n) if j != i)
        fn = sum(matrix[j][i] for j in range(n) if j != i)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        support   = tp + fn
        print(f"  {cls:<14} {precision:>10.2f} {recall:>10.2f} {f1:>10.2f} {support:>10}")

    total = sum(matrix[i][i] for i in range(n))
    grand = sum(matrix[i][j] for i in range(n) for j in range(n))
    print(f"\n  Overall accuracy: {total}/{grand} = {total/max(grand,1)*100:.1f}%")


# ═══════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Confusion matrix using FULL collision+spark+hole pipeline.")
    parser.add_argument("--config",         default="config.json",
                        help="Config JSON with pedal + hole definitions (default: config.json)")
    parser.add_argument("--classified-dir", default="./test/results/classified",
                        help="Root folder with one subfolder per class (default: ./classified)")
    parser.add_argument("--spark-frames",   type=int, default=1,
                        help="Spark confirmation frame count (default: 1; use 3 to restore old behaviour)")
    parser.add_argument("--hole-frames",    type=int, default=1,
                        help="Hole confirmation frame count (default: 1, same as detector)")
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────
    with open(args.config) as f:
        cfg = json.load(f)
    pedals  = cfg.get("pedals", [])
    holes   = cfg.get("holes",  [])

    # Classes = pedal names + "miss" only (no holes)
    classes = [p["name"] for p in pedals] + ["miss"]
    print(f"\n  Config loaded: {len(pedals)} pedal(s), {len(holes)} hole(s) "
          f"(holes used for tracking only, not classification)")
    print(f"  Classes ({len(classes)}): {classes}")

    # ── Collect clips ────────────────────────────────────────
    # Only scan pedal folders + miss — hole folders are not part of this evaluation
    all_clips = []   # list of (video_path, true_label)
    for cls in classes:
        folder = os.path.join(args.classified_dir, cls)
        if not os.path.isdir(folder):
            print(f"  WARNING: folder not found: {folder}")
            continue
        clips = sorted(
            glob(os.path.join(folder, "*.MP4")) +
            glob(os.path.join(folder, "*.mp4"))
        )
        for c in clips:
            all_clips.append((c, cls))
        print(f"  {cls:<16s}: {len(clips)} clips")

    if not all_clips:
        print("\n  No clips found. Check --classified-dir.")
        return

    print(f"\n  Running FULL pipeline on {len(all_clips)} clips...\n")

    # ── Run classifier + build matrix ────────────────────────
    cls_idx = {c: i for i, c in enumerate(classes)}
    n       = len(classes)
    matrix  = [[0]*n for _ in range(n)]
    wrong   = []

    for idx, (video_path, true_label) in enumerate(all_clips):
        name = Path(video_path).name
        print(f"  [{idx+1:3d}/{len(all_clips)}] {name}  (true={true_label}) ...",
              end=" ", flush=True)

        predicted = classify_clip_full(
            video_path, pedals, holes,
            spark_confirm_frames=args.spark_frames,
            hole_confirm_frames=args.hole_frames,
        )

        ok = predicted == true_label
        print(f"pred={predicted}" + (" ✓" if ok else " ✗"))

        # If classifier somehow returns a hole name (shouldn't happen),
        # treat it as "miss" since holes are not in the class list
        row = cls_idx.get(predicted, cls_idx["miss"])
        col = cls_idx.get(true_label, cls_idx["miss"])
        matrix[row][col] += 1

        if not ok:
            wrong.append((name, true_label, predicted))

    # ── Print results ─────────────────────────────────────────
    print_confusion_matrix(matrix, classes)

    if wrong:
        print(f"\n  Misclassified clips ({len(wrong)}):")
        for name, true, pred in wrong:
            print(f"    {name}  true={true}  pred={pred}")
    else:
        print("\n  All clips classified correctly!")


if __name__ == "__main__":
    main()